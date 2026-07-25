"""One-shot remote outer lifecycle for the MMQA P1 formal study.

The outer process is deliberately an aggregate-only orchestrator.  It never
opens a source payload, action pack, gold pack, trusted ledger, or API
credential.  Every successful stage is preceded by a durable one-shot marker
and followed by an outer completion receipt binding the stage's persisted
aggregate receipt.  A failure writes one sanitized outer terminal receipt and
stops; restart, retry, replay, resampling, and provider switching are absent.

The fixed order is:

1. verify the execution and implementation freezes;
2. complete and seal the public-synthetic local runtime preflight while every
   formal source-download artifact is still absent;
3. consume the authorized one-shot four-file acquisition;
4. build the prequalification source/implementation freeze;
5. run the unique aggregate-only source qualification;
6. run the unique private selection; and
7. start the only post-selection formal-action controller as one fixed user
   transient service under ``/usr/bin/systemd-run --user --wait --collect``.

The final controller is supplied as an exact executable plus module file and
both byte identities are checked against caller-provided frozen SHA-256
values.  The transient service is fixed to
``RestrictAddressFamilies=AF_UNIX``, ``NoNewPrivileges=yes``,
``PrivateTmp=yes``, ``ProtectSystem=strict``, ``ProtectHome=read-only``, one
project-root ``ReadWritePaths``, and ``UMask=0077``.  The frozen wrapper must
actively prove both AF_INET and AF_INET6 socket denial before formal action.
``/usr/bin/env -i`` supplies a literal child environment whitelist; the
parent environment is never copied or inspected, and no API, Ruoli, proxy,
credential, or token variable is forwarded.

Tests inject every stage and the subprocess runner.  They never download,
open formal source data, run a model, or launch bubblewrap.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Protocol


VERSION = "mmqa_p1_remote_outer_lifecycle_v1"
STUDY_ID = "MMQA_P1_LOCAL_PROOF_E5_V1"
SYSTEMD_RUN_PATH = Path("/usr/bin/systemd-run")
ENV_PATH = Path("/usr/bin/env")
FORMAL_ACTION_UNIT_NAME = "mmqa-p1-formal-action-v1.service"
SOURCE_ACQUISITION_UNIT_NAME = (
    "mmqa-p1-source-acquisition-v1.service"
)
EXECUTION_FREEZE_SELF_SHA256_PLACEHOLDER = (
    "__MMQA_P1_EXECUTION_FREEZE_SELF_SHA256__"
)
LOCAL_PREFLIGHT_SELF_SHA256_PLACEHOLDER = (
    "__MMQA_P1_LOCAL_PREFLIGHT_SELF_SHA256__"
)
SELECTION_ACQUISITION_SHA256_PLACEHOLDER = (
    "__MMQA_P1_SELECTION_ACQUISITION_SHA256__"
)
FORMAL_ACTION_UNIT_PROPERTY_TEMPLATES = (
    "NoNewPrivileges=yes",
    "PrivateTmp=yes",
    "ProtectHome=read-only",
    "ProtectSystem=strict",
    "ReadWritePaths={project_root}",
    "RestrictAddressFamilies=AF_UNIX",
    "UMask=0077",
)
SOURCE_ACQUISITION_UNIT_PROPERTY_TEMPLATES = (
    "NoNewPrivileges=yes",
    "PrivateTmp=yes",
    "ProtectHome=read-only",
    "ProtectSystem=strict",
    "ReadWritePaths={project_root}",
    "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
    "UMask=0077",
)
OUTER_NETWORK_ISOLATION_CONTRACT = {
    "AF_INET_and_AF_INET6_denial_probe_required": True,
    "private_network_namespace_claimed": False,
    "required_RestrictAddressFamilies": ["AF_UNIX"],
}
OFFICIAL_HIPPO_PREFLIGHT_SCHEMA = (
    "mmqa_p1_official_hipporag_block_v1_fresh_comparator_preflight_v1"
)
OFFICIAL_HIPPO_PREFLIGHT_STATUS = (
    "passed_public_synthetic_candidate_only_fresh_runtime"
)
EXPECTED_CUSTODY_SELF_SHA256 = (
    "e82cb94e54a3020d1f2e41f47ed4141d19b448db985479551b1d933b43bf15f5"
)
EXPECTED_DESIGN_SELF_SHA256 = (
    "eefa61986bd2f58efa26564dc0709728e0323660f23ae532819f4fa98f0601b3"
)
EXPECTED_AUTHORIZATION_SELF_SHA256 = (
    "08f4bbc25c7d15182b16da909d535a4492e80c302940742e1e92c2828d7360cb"
)
EXPECTED_RUNTIME_DISPOSITION_SELF_SHA256 = (
    "73e468b2bafda0595ab05097bff62bc34109318219670ac870a17081bce0317a"
)

EXECUTION_FREEZE_RELATIVE = Path(
    "manifests/mmqa_p1_execution_freeze_v1.json"
)
IMPLEMENTATION_FREEZE_RELATIVE = Path(
    "manifests/mmqa_p1_implementation_freeze_v1.json"
)
RUNTIME_DISPOSITION_RELATIVE = Path(
    "manifests/mmqa_p1_preexecution_runtime_disposition_v1.json"
)
PREFLIGHT_RECEIPT_RELATIVE = Path(
    "manifests/mmqa_p1_local_runtime_preflight_receipt_v1.json"
)
DOWNLOAD_RECEIPT_RELATIVE = Path(
    "manifests/mmqa_p1_source_download_receipt_v1.json"
)
QUALIFICATION_FREEZE_RELATIVE = Path(
    "manifests/mmqa_p1_source_qualification_freeze_v1.json"
)
QUALIFICATION_RESULT_RELATIVE = Path(
    "manifests/mmqa_p1_source_qualification_result_v1.json"
)
SELECTION_ROOT_RELATIVE = Path("artifacts/mmqa_p1_private_selection_v1")
SELECTION_RECEIPT_RELATIVE = (
    SELECTION_ROOT_RELATIVE / "selection_receipt.public.json"
)
SOURCE_ROOT_RELATIVE = Path("artifacts/mmqa_p1_official_source_v1")
SOURCE_ACQUISITION_ATTEMPT_RELATIVE = Path(
    "artifacts/mmqa_p1_source_acquisition_v1/download.one_shot_attempt.json"
)
SOURCE_ACQUISITION_FAILURE_RELATIVE = Path(
    "manifests/mmqa_p1_source_download_terminal_failure_v1.json"
)
SOURCE_ACQUISITION_MODULE_RELATIVE = Path(
    "assumption_agent/benchmarks/mmqa_p1_source_acquisition_v1.py"
)

OUTER_ROOT_RELATIVE = Path("artifacts/mmqa_p1_remote_outer_lifecycle_v1")
OUTER_MARKER_FILENAME = "outer.one_shot.attempt.json"
OUTER_SUCCESS_FILENAME = "outer.success.aggregate.json"
OUTER_FAILURE_FILENAME = "outer.terminal_failure.aggregate.json"
STAGE_DIRECTORY_NAME = "stages"

STAGE_ORDER = (
    "verify_execution_and_implementation_freezes",
    "public_synthetic_local_runtime_preflight",
    "authorized_source_acquisition",
    "source_qualification_freeze",
    "aggregate_source_qualification",
    "private_one_shot_selection",
    "post_selection_network_denied_formal_action",
)
REQUIRED_PREFLIGHT_BINDING_NAMES = frozenset(
    {
        "public_synthetic_local_runtime_preflight",
        "official_hipporag_runtime_binding_canary",
    }
)
REQUIRED_FREEZE_BINDING_NAMES = frozenset(
    {"execution_freeze", "implementation_freeze"}
)
EXPECTED_STAGE_BINDING_NAMES = {
    "verify_execution_and_implementation_freezes": (
        REQUIRED_FREEZE_BINDING_NAMES
    ),
    "public_synthetic_local_runtime_preflight": (
        REQUIRED_PREFLIGHT_BINDING_NAMES
    ),
    "authorized_source_acquisition": frozenset({"source_download"}),
    "source_qualification_freeze": frozenset(
        {"source_qualification_freeze"}
    ),
    "aggregate_source_qualification": frozenset(
        {"source_qualification_result"}
    ),
    "private_one_shot_selection": frozenset(
        {"private_selection_public_receipt"}
    ),
    "post_selection_network_denied_formal_action": frozenset(
        {"post_selection_formal_action_terminal"}
    ),
}
REQUIRED_IMPLEMENTATION_RELATIVES = frozenset(
    {
        "assumption_agent/__init__.py",
        "assumption_agent/benchmarks/__init__.py",
        "assumption_agent/benchmarks/eraser_evidence_inference_official_hipporag_v1/__init__.py",
        "assumption_agent/benchmarks/eraser_evidence_inference_official_hipporag_v1/adapter.py",
        "assumption_agent/benchmarks/eraser_evidence_inference_official_hipporag_v1/contract.py",
        "assumption_agent/benchmarks/eraser_evidence_inference_official_hipporag_v1/worker.py",
        "assumption_agent/benchmarks/mmqa_p1_action_integration_v1.py",
        "assumption_agent/benchmarks/mmqa_p1_block_coordinate_worker_v1.py",
        "assumption_agent/benchmarks/mmqa_p1_formal_action_runtime_v1.py",
        "assumption_agent/benchmarks/mmqa_p1_formal_controller_v1.py",
        "assumption_agent/benchmarks/mmqa_p1_local_action_executor_v1.py",
        "assumption_agent/benchmarks/mmqa_p1_local_runtime_preflight_v1.py",
        "assumption_agent/benchmarks/mmqa_p1_official_hipporag_block_v1.py",
        "assumption_agent/benchmarks/mmqa_p1_private_selection_v1.py",
        "assumption_agent/benchmarks/mmqa_p1_remote_outer_lifecycle_v1.py",
        "assumption_agent/benchmarks/mmqa_p1_source_acquisition_v1.py",
        "assumption_agent/benchmarks/mmqa_p1_source_qualification_freeze_v1.py",
        "assumption_agent/benchmarks/mmqa_p1_source_qualification_v1.py",
        "assumption_agent/benchmarks/mmqa_p1_typed_proof_e5_core_v1.py",
        "assumption_agent/models.py",
        "manifests/mmqa_p1_local_proof_e5_study_design_v1.json",
        "manifests/mmqa_p1_preexecution_runtime_disposition_v1.json",
        "manifests/mmqa_p1_source_custody_v1.json",
        "manifests/mmqa_p1_source_download_authorization_v1.json",
        "tests/test_eraser_evidence_inference_official_hipporag_v1.py",
        "tests/test_mmqa_p1_action_integration_v1.py",
        "tests/test_mmqa_p1_block_coordinate_worker_v1.py",
        "tests/test_mmqa_p1_formal_action_runtime_v1.py",
        "tests/test_mmqa_p1_formal_controller_v1.py",
        "tests/test_mmqa_p1_local_action_executor_v1.py",
        "tests/test_mmqa_p1_local_runtime_preflight_v1.py",
        "tests/test_mmqa_p1_official_hipporag_block_v1.py",
        "tests/test_mmqa_p1_private_selection_v1.py",
        "tests/test_mmqa_p1_remote_outer_lifecycle_v1.py",
        "tests/test_mmqa_p1_source_acquisition_v1.py",
        "tests/test_mmqa_p1_source_qualification_freeze_v1.py",
        "tests/test_mmqa_p1_source_qualification_v1.py",
        "tests/test_mmqa_p1_typed_proof_e5_core_v1.py",
    }
)
EXECUTION_POLICY = {
    "formal_online_evaluator_or_API_allowed": False,
    "network_after_source_acquisition_allowed": False,
    "outer_AF_INET_and_AF_INET6_denial_probe_required": True,
    "outer_RestrictAddressFamilies": ["AF_UNIX"],
    "outer_network_allowed": False,
    "outer_restart_retry_replay_resample_provider_or_model_switch_allowed": (
        False
    ),
    "parent_environment_inheritance_allowed": False,
    "post_selection_network_allowed": False,
    "post_selection_AF_INET_and_AF_INET6_denial_probe_required": True,
    "post_selection_transient_user_service_required": True,
    "post_selection_transient_user_service_property_templates": list(
        FORMAL_ACTION_UNIT_PROPERTY_TEMPLATES
    ),
    "post_selection_transient_user_service_unit": FORMAL_ACTION_UNIT_NAME,
    "public_synthetic_preflight_network_allowed": False,
    "source_acquisition_network_scope": (
        "four_authorized_fixed_HTTPS_GETs_only"
    ),
    "source_acquisition_sibling_transient_user_service_property_templates": (
        list(SOURCE_ACQUISITION_UNIT_PROPERTY_TEMPLATES)
    ),
    "source_acquisition_sibling_transient_user_service_unit": (
        SOURCE_ACQUISITION_UNIT_NAME
    ),
    "source_download_after_both_public_synthetic_preflights": True,
}

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_NAME = re.compile(r"[a-z][a-z0-9_]{0,95}\Z")
_SAFE_SCHEMA_STATUS = re.compile(r"[A-Za-z0-9_.:-]{1,192}\Z")
MAXIMUM_AGGREGATE_RECEIPT_BYTES = 32 << 20
DEFAULT_CONTROLLER_TIMEOUT_SECONDS = 7 * 24 * 60 * 60
SOURCE_ACQUISITION_TIMEOUT_SECONDS = 6 * 60 * 60

_FORBIDDEN_CHILD_ENV_FRAGMENTS = (
    "API_KEY",
    "OPENAI",
    "RUOLI",
    "BEARER",
    "CREDENTIAL",
    "SECRET",
    "ACCESS_TOKEN",
    "AUTH_TOKEN",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)
_FIXED_CHILD_ENV_NAMES = frozenset(
    {
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_OFFLINE",
        "HOME",
        "LANG",
        "LC_ALL",
        "OMP_NUM_THREADS",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "TOKENIZERS_PARALLELISM",
        "TRANSFORMERS_OFFLINE",
    }
)
_FIXED_SOURCE_ACQUISITION_ENV_NAMES = frozenset(
    {
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
    }
)
_FIXED_SYSTEMD_CLIENT_ENV_NAMES = frozenset(
    {
        "DBUS_SESSION_BUS_ADDRESS",
        "LANG",
        "LC_ALL",
        "PATH",
        "XDG_RUNTIME_DIR",
    }
)


def _resolved_unit_properties(project: Path) -> tuple[str, ...]:
    """Resolve the sole project-specific transient-service property."""

    project_text = str(Path(project).resolve(strict=True))
    return tuple(
        template.replace("{project_root}", project_text)
        for template in FORMAL_ACTION_UNIT_PROPERTY_TEMPLATES
    )


def _resolved_source_acquisition_unit_properties(
    project: Path,
) -> tuple[str, ...]:
    project_text = str(Path(project).resolve(strict=True))
    properties = tuple(
        template.replace("{project_root}", project_text)
        for template in SOURCE_ACQUISITION_UNIT_PROPERTY_TEMPLATES
    )
    if properties.count(
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6"
    ) != 1:
        raise MMQAP1RemoteOuterLifecycleError(
            "source acquisition address-family policy drifted"
        )
    return properties


def _transient_unit_contract(project: Path) -> dict[str, object]:
    return {
        "AF_INET_and_AF_INET6_denial_probe_required": True,
        "child_environment_clear_executable": str(ENV_PATH),
        "child_environment_clear_flag": "-i",
        "properties": list(_resolved_unit_properties(project)),
        "systemd_run_prefix": [
            str(SYSTEMD_RUN_PATH),
            "--user",
            "--wait",
            "--collect",
            "--quiet",
        ],
        "unit_name": FORMAL_ACTION_UNIT_NAME,
        "working_directory": str(Path(project).resolve(strict=True)),
    }


def _source_acquisition_transient_unit_contract(
    project: Path,
) -> dict[str, object]:
    return {
        "address_family_restriction_property_present": True,
        "allowed_address_families": ["AF_UNIX", "AF_INET", "AF_INET6"],
        "child_environment_clear_executable": str(ENV_PATH),
        "child_environment_clear_flag": "-i",
        "network_scope": "four_authorized_fixed_HTTPS_GETs_only",
        "properties": list(
            _resolved_source_acquisition_unit_properties(project)
        ),
        "systemd_run_prefix": [
            str(SYSTEMD_RUN_PATH),
            "--user",
            "--wait",
            "--collect",
            "--quiet",
        ],
        "timeout_seconds": SOURCE_ACQUISITION_TIMEOUT_SECONDS,
        "unit_name": SOURCE_ACQUISITION_UNIT_NAME,
        "working_directory": str(Path(project).resolve(strict=True)),
    }


class MMQAP1RemoteOuterLifecycleError(RuntimeError):
    """The one-shot outer lifecycle failed closed."""


@dataclass(frozen=True)
class OuterLifecycleConfig:
    project_root: Path
    execution_freeze_self_sha256: str
    implementation_freeze_self_sha256: str
    typed_python: Path
    minilm_model: Path
    cross_encoder_model: Path
    nvidia_smi: Path
    systemd_run_sha256: str
    env_executable_sha256: str
    controller_executable: Path
    controller_executable_sha256: str
    controller_module: Path
    controller_module_sha256: str
    controller_arguments: tuple[str, ...]
    official_hippo_receipt_relative: Path
    official_hippo_receipt_schema: str
    official_hippo_receipt_status: str
    official_hippo_receipt_self_sha256: str | None
    official_runtime_python: Path
    official_pyvenv_cfg: Path
    official_overlay_root: Path
    official_hipporag_source_root: Path
    official_p16_site_root: Path
    official_local_llm_model: Path
    official_local_embedding_model: Path
    official_expected_package_versions: Mapping[str, str]
    official_expected_module_import_roots: Mapping[str, str]
    controller_receipt_relative: Path
    controller_receipt_schema: str
    controller_receipt_status: str
    controller_timeout_seconds: int = DEFAULT_CONTROLLER_TIMEOUT_SECONDS


@dataclass(frozen=True)
class ReceiptSpec:
    name: str
    relative_path: Path
    expected_schema: str
    expected_status: str | None = None
    self_hash_field: str = "self_sha256"
    expected_self_sha256: str | None = None
    required_mode: int | None = None


@dataclass(frozen=True)
class ReceiptBinding:
    name: str
    relative_path: str
    schema: str
    status: str | None
    self_hash_field: str
    self_sha256: str
    file_sha256: str
    size_bytes: int
    mode_octal: str

    def payload(self) -> dict[str, object]:
        return {
            "file_sha256": self.file_sha256,
            "mode_octal": self.mode_octal,
            "name": self.name,
            "relative_path": self.relative_path,
            "schema": self.schema,
            "self_hash_field": self.self_hash_field,
            "self_sha256": self.self_sha256,
            "size_bytes": self.size_bytes,
            "status": self.status,
        }


@dataclass(frozen=True)
class StageContext:
    config: OuterLifecycleConfig
    bindings: Mapping[str, ReceiptBinding]


class StageRunner(Protocol):
    def __call__(self, context: StageContext) -> Sequence[ReceiptSpec]: ...


@dataclass(frozen=True)
class LifecycleStages:
    verify_execution_and_implementation_freezes: StageRunner
    public_synthetic_local_runtime_preflight: StageRunner
    authorized_source_acquisition: StageRunner
    source_qualification_freeze: StageRunner
    aggregate_source_qualification: StageRunner
    private_one_shot_selection: StageRunner
    post_selection_network_denied_formal_action: StageRunner

    def ordered(self) -> tuple[tuple[str, StageRunner], ...]:
        return tuple((stage, getattr(self, stage)) for stage in STAGE_ORDER)


@dataclass(frozen=True)
class _Written:
    value: Mapping[str, Any]
    file_sha256: str


ProcessRunner = Callable[..., Any]


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MMQAP1RemoteOuterLifecycleError(
            "outer aggregate metadata is invalid"
        ) from exc


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _self_hashed(body: Mapping[str, Any]) -> dict[str, Any]:
    if "self_sha256" in body:
        raise MMQAP1RemoteOuterLifecycleError(
            "outer aggregate metadata already has a self hash"
        )
    result = dict(body)
    result["self_sha256"] = _semantic_hash(result)
    return result


def _duplicate_rejecting_object(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MMQAP1RemoteOuterLifecycleError(
                "aggregate receipt contains a duplicate key"
            )
        result[key] = value
    return result


def _reject_nonfinite(_value: str) -> None:
    raise MMQAP1RemoteOuterLifecycleError(
        "aggregate receipt contains a non-finite number"
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise MMQAP1RemoteOuterLifecycleError(
                "outer lifecycle directory is unsafe"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path, *, mode: int = 0o700) -> None:
    missing: list[Path] = []
    cursor = path
    while True:
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            if cursor.parent == cursor:
                raise MMQAP1RemoteOuterLifecycleError(
                    "outer lifecycle directory parent is unavailable"
                )
            missing.append(cursor)
            cursor = cursor.parent
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise MMQAP1RemoteOuterLifecycleError(
                "outer lifecycle directory path is unsafe"
            )
        break
    for directory in reversed(missing):
        os.mkdir(directory, mode)
        os.chmod(directory, mode)
        _fsync_directory(directory)
        _fsync_directory(directory.parent)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> _Written:
    raw = _canonical_bytes(value)
    _ensure_directory(path.parent)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
            metadata = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_size != len(raw)
            ):
                raise MMQAP1RemoteOuterLifecycleError(
                    "outer lifecycle receipt metadata drifted"
                )
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return _Written(value, hashlib.sha256(raw).hexdigest())


def _create_outer_root(project: Path) -> Path:
    root = project / OUTER_ROOT_RELATIVE
    _ensure_directory(root.parent)
    try:
        os.mkdir(root, 0o700)
        os.chmod(root, 0o700)
        _fsync_directory(root)
        _fsync_directory(root.parent)
    except OSError as exc:
        raise MMQAP1RemoteOuterLifecycleError(
            "outer lifecycle attempt is already consumed"
        ) from exc
    return root


def _safe_relative(path: Path, *, label: str) -> str:
    value = PurePosixPath(path.as_posix())
    if (
        value.is_absolute()
        or not value.parts
        or any(part in {"", ".", ".."} for part in value.parts)
    ):
        raise MMQAP1RemoteOuterLifecycleError(f"{label} path is unsafe")
    return value.as_posix()


def _validate_receipt_spec(spec: ReceiptSpec) -> str:
    if not _SAFE_NAME.fullmatch(spec.name):
        raise MMQAP1RemoteOuterLifecycleError(
            "stage receipt name is invalid"
        )
    relative = _safe_relative(spec.relative_path, label="stage receipt")
    lowered = relative.casefold()
    if (
        lowered.endswith(".gz")
        or ".action.label_free." in lowered
        or ".gold.sealed." in lowered
        or "selection_secret" in lowered
        or "trusted.private" in lowered
        or "source_mapping" in lowered
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "outer lifecycle cannot open private or source payload paths"
        )
    if not _SAFE_SCHEMA_STATUS.fullmatch(spec.expected_schema):
        raise MMQAP1RemoteOuterLifecycleError(
            "stage receipt schema is invalid"
        )
    if (
        spec.expected_status is not None
        and not _SAFE_SCHEMA_STATUS.fullmatch(spec.expected_status)
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "stage receipt status is invalid"
        )
    if not _SAFE_NAME.fullmatch(spec.self_hash_field):
        raise MMQAP1RemoteOuterLifecycleError(
            "stage receipt self-hash field is invalid"
        )
    if (
        spec.expected_self_sha256 is not None
        and _HEX64.fullmatch(spec.expected_self_sha256) is None
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "stage receipt expected self hash is invalid"
        )
    if spec.required_mode not in {None, 0o600, 0o644}:
        raise MMQAP1RemoteOuterLifecycleError(
            "stage receipt required mode is invalid"
        )
    return relative


def _read_stable_regular(path: Path) -> tuple[bytes, os.stat_result]:
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
    except OSError as exc:
        raise MMQAP1RemoteOuterLifecycleError(
            "stage aggregate receipt is unavailable"
        ) from exc
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > MAXIMUM_AGGREGATE_RECEIPT_BYTES
        ):
            raise MMQAP1RemoteOuterLifecycleError(
                "stage aggregate receipt file is unsafe"
            )
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                raise MMQAP1RemoteOuterLifecycleError(
                    "stage aggregate receipt ended early"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise MMQAP1RemoteOuterLifecycleError(
                "stage aggregate receipt grew during read"
            )
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
            raise MMQAP1RemoteOuterLifecycleError(
                "stage aggregate receipt changed during read"
            )
    finally:
        os.close(descriptor)
    return b"".join(chunks), after


def _load_receipt_value_and_binding(
    project: Path, spec: ReceiptSpec
) -> tuple[dict[str, Any], ReceiptBinding]:
    relative = _validate_receipt_spec(spec)
    raw, metadata = _read_stable_regular(project / relative)
    if (
        spec.required_mode is not None
        and stat.S_IMODE(metadata.st_mode) != spec.required_mode
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "stage aggregate receipt mode drifted"
        )
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise MMQAP1RemoteOuterLifecycleError(
            "stage aggregate receipt JSON is invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise MMQAP1RemoteOuterLifecycleError(
            "stage aggregate receipt shape drifted"
        )
    value = dict(value)
    body = dict(value)
    claimed = body.pop(spec.self_hash_field, None)
    if (
        not isinstance(claimed, str)
        or _HEX64.fullmatch(claimed) is None
        or _semantic_hash(body) != claimed
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "stage aggregate receipt self hash drifted"
        )
    if (
        spec.expected_self_sha256 is not None
        and claimed != spec.expected_self_sha256
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "stage aggregate receipt frozen hash drifted"
        )
    if (
        value.get("schema") != spec.expected_schema
        or value.get("study_id") != STUDY_ID
        or (
            spec.expected_status is not None
            and value.get("status") != spec.expected_status
        )
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "stage aggregate receipt contract drifted"
        )
    binding = ReceiptBinding(
        name=spec.name,
        relative_path=relative,
        schema=spec.expected_schema,
        status=spec.expected_status,
        self_hash_field=spec.self_hash_field,
        self_sha256=claimed,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        mode_octal=f"{stat.S_IMODE(metadata.st_mode):04o}",
    )
    return value, binding


def _load_receipt_binding(
    project: Path, spec: ReceiptSpec
) -> ReceiptBinding:
    _value, binding = _load_receipt_value_and_binding(project, spec)
    return binding


def _sha256_regular_file(
    path: Path, *, resolve_symlink: bool = False
) -> str:
    candidate = path.resolve(strict=True) if resolve_symlink else path
    raw, _metadata = _read_stable_regular(candidate)
    return hashlib.sha256(raw).hexdigest()


def _verify_outer_inet_denial_once(
    *,
    socket_factory: Callable[[int, int], Any] = socket.socket,
) -> Mapping[str, object]:
    denied: dict[str, int] = {}
    for family, label in (
        (socket.AF_INET, "AF_INET"),
        (socket.AF_INET6, "AF_INET6"),
    ):
        candidate: Any | None = None
        try:
            candidate = socket_factory(family, socket.SOCK_STREAM)
        except OSError as exc:
            if exc.errno != errno.EAFNOSUPPORT:
                raise MMQAP1RemoteOuterLifecycleError(
                    "outer address-family denial probe failed closed"
                ) from exc
            denied[label] = exc.errno
        else:
            candidate.close()
            raise MMQAP1RemoteOuterLifecycleError(
                "outer address-family isolation is absent"
            )
        finally:
            if candidate is not None:
                candidate.close()
    if denied != {
        "AF_INET": errno.EAFNOSUPPORT,
        "AF_INET6": errno.EAFNOSUPPORT,
    }:
        raise MMQAP1RemoteOuterLifecycleError(
            "outer address-family denial probe is incomplete"
        )
    return MappingProxyType(
        {
            "AF_INET6_socket_creation_errno": "EAFNOSUPPORT",
            "AF_INET_socket_creation_errno": "EAFNOSUPPORT",
            "denied_family_count": 2,
            "outer_network_isolation_contract": (
                OUTER_NETWORK_ISOLATION_CONTRACT
            ),
            "probe_count": 2,
            "schema": f"{VERSION}_outer_address_family_probe_v1",
            "status": "AF_INET_and_AF_INET6_socket_creation_denied",
        }
    )


def _canonical_formal_runtime_arguments(
    config: OuterLifecycleConfig,
    *,
    execution_freeze_sha256: str,
) -> tuple[str, ...]:
    """Build the only accepted split argv for the frozen formal runtime."""

    from assumption_agent.benchmarks import (
        mmqa_p1_local_runtime_preflight_v1 as local_preflight,
    )

    try:
        project = Path(config.project_root).resolve(strict=True)
        paths = _official_runtime_paths(config)
    except Exception as exc:
        raise MMQAP1RemoteOuterLifecycleError(
            "formal runtime argv path contract is invalid"
        ) from exc
    receipt_sha256 = config.official_hippo_receipt_self_sha256
    if (
        not isinstance(receipt_sha256, str)
        or _HEX64.fullmatch(receipt_sha256) is None
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "formal runtime argv official receipt hash is invalid"
        )
    arguments = [
        "--project",
        str(project),
        "--execution-freeze-self-sha256",
        execution_freeze_sha256,
        "--implementation-freeze-self-sha256",
        config.implementation_freeze_self_sha256,
        "--local-preflight-receipt",
        str(project / PREFLIGHT_RECEIPT_RELATIVE),
        "--local-preflight-self-sha256",
        LOCAL_PREFLIGHT_SELF_SHA256_PLACEHOLDER,
        "--typed-python",
        str(config.typed_python),
        "--typed-python-resolved-sha256",
        config.controller_executable_sha256,
        "--minilm-model",
        str(config.minilm_model),
        "--minilm-required-tree-sha256",
        local_preflight.MINILM_REQUIRED_TREE_SHA256,
        "--cross-encoder-model",
        str(config.cross_encoder_model),
        "--cross-encoder-required-tree-sha256",
        local_preflight.CE_REQUIRED_TREE_SHA256,
        "--nvidia-smi",
        str(config.nvidia_smi),
        "--systemd-run",
        str(SYSTEMD_RUN_PATH),
        "--systemd-run-resolved-sha256",
        config.systemd_run_sha256,
        "--systemd-isolation-disposition-sha256",
        _semantic_hash(_transient_unit_contract(project)),
        "--runtime-module-sha256",
        config.controller_module_sha256,
        "--official-preflight-receipt",
        str(project / config.official_hippo_receipt_relative),
        "--official-preflight-receipt-sha256",
        receipt_sha256,
        "--official-runtime-python",
        paths.runtime_python,
        "--official-pyvenv-cfg",
        paths.pyvenv_cfg,
        "--official-overlay-root",
        paths.overlay_root,
        "--official-hipporag-source-root",
        paths.hipporag_source_root,
        "--official-p16-site-root",
        paths.p16_site_root,
        "--official-local-llm-model",
        paths.local_llm_model,
        "--official-local-embedding-model",
        paths.local_embedding_model,
    ]
    for name, value in sorted(
        config.official_expected_package_versions.items()
    ):
        arguments.extend(("--official-package-version", f"{name}={value}"))
    for name, value in sorted(
        config.official_expected_module_import_roots.items()
    ):
        arguments.extend(
            ("--official-module-import-root", f"{name}={value}")
        )
    arguments.extend(
        (
            "--selection-acquisition-sha256",
            SELECTION_ACQUISITION_SHA256_PLACEHOLDER,
        )
    )
    return tuple(arguments)


def _controller_argument_template(
    config: OuterLifecycleConfig,
) -> tuple[str, ...]:
    """Validate the full child argv and restore the execution hash marker."""

    expected = _canonical_formal_runtime_arguments(
        config,
        execution_freeze_sha256=config.execution_freeze_self_sha256,
    )
    if tuple(config.controller_arguments) != expected:
        raise MMQAP1RemoteOuterLifecycleError(
            "formal runtime exact split argv contract drifted"
        )
    template = _canonical_formal_runtime_arguments(
        config,
        execution_freeze_sha256=(
            EXECUTION_FREEZE_SELF_SHA256_PLACEHOLDER
        ),
    )
    observed_placeholders = {
        argument
        for argument in template
        if argument.startswith("__MMQA_P1_")
        or argument.endswith("_SHA256__")
    }
    if observed_placeholders != {
        EXECUTION_FREEZE_SELF_SHA256_PLACEHOLDER,
        LOCAL_PREFLIGHT_SELF_SHA256_PLACEHOLDER,
        SELECTION_ACQUISITION_SHA256_PLACEHOLDER,
    }:
        raise MMQAP1RemoteOuterLifecycleError(
            "formal controller argument placeholder registry drifted"
        )
    return template


def _official_runtime_paths(config: OuterLifecycleConfig) -> Any:
    from assumption_agent.benchmarks import (
        mmqa_p1_official_hipporag_block_v1 as official_hippo,
    )

    return official_hippo.FreshComparatorRuntimePaths(
        runtime_python=str(config.official_runtime_python),
        pyvenv_cfg=str(config.official_pyvenv_cfg),
        overlay_root=str(config.official_overlay_root),
        hipporag_source_root=str(
            config.official_hipporag_source_root
        ),
        p16_site_root=str(config.official_p16_site_root),
        local_llm_model=str(config.official_local_llm_model),
        local_embedding_model=str(
            config.official_local_embedding_model
        ),
    )


def _official_runtime_contract(
    config: OuterLifecycleConfig,
) -> dict[str, object]:
    paths = _official_runtime_paths(config)
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
            for field in (
                "runtime_python",
                "pyvenv_cfg",
                "overlay_root",
                "hipporag_source_root",
                "p16_site_root",
                "local_llm_model",
                "local_embedding_model",
            )
        },
    }


def _validate_config(config: OuterLifecycleConfig) -> Path:
    try:
        project = Path(config.project_root).resolve(strict=True)
    except OSError as exc:
        raise MMQAP1RemoteOuterLifecycleError(
            "remote project root is unavailable"
        ) from exc
    if not project.is_dir():
        raise MMQAP1RemoteOuterLifecycleError(
            "remote project root is invalid"
        )
    for runtime_path in (
        config.typed_python,
        config.minilm_model,
        config.cross_encoder_model,
        config.nvidia_smi,
        config.controller_executable,
        config.controller_module,
        config.official_runtime_python,
        config.official_pyvenv_cfg,
        config.official_overlay_root,
        config.official_hipporag_source_root,
        config.official_p16_site_root,
        config.official_local_llm_model,
        config.official_local_embedding_model,
    ):
        if not Path(runtime_path).is_absolute():
            raise MMQAP1RemoteOuterLifecycleError(
                "outer lifecycle runtime path is not absolute"
            )
    for digest in (
        config.execution_freeze_self_sha256,
        config.implementation_freeze_self_sha256,
        config.systemd_run_sha256,
        config.env_executable_sha256,
        config.controller_executable_sha256,
        config.controller_module_sha256,
    ):
        if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
            raise MMQAP1RemoteOuterLifecycleError(
            "outer lifecycle frozen SHA256 is invalid"
            )
    if (
        not isinstance(config.official_hippo_receipt_self_sha256, str)
        or _HEX64.fullmatch(
            config.official_hippo_receipt_self_sha256
        )
        is None
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "official HippoRAG receipt SHA256 is invalid"
        )
    for argument in config.controller_arguments:
        if (
            not isinstance(argument, str)
            or "\x00" in argument
            or len(argument) > 16_384
        ):
            raise MMQAP1RemoteOuterLifecycleError(
                "formal controller argument is invalid"
            )
    try:
        from assumption_agent.benchmarks import (
            mmqa_p1_official_hipporag_block_v1 as official_hippo,
        )

        official_hippo._validated_expected_maps(  # noqa: SLF001
            config.official_expected_package_versions,
            config.official_expected_module_import_roots,
        )
        _official_runtime_paths(config)
    except Exception as exc:
        raise MMQAP1RemoteOuterLifecycleError(
            "official HippoRAG runtime contract is invalid"
        ) from exc
    _controller_argument_template(config)
    if Path(config.controller_executable).absolute() != Path(
        config.typed_python
    ).absolute():
        raise MMQAP1RemoteOuterLifecycleError(
            "formal and acquisition typed executable binding drifted"
        )
    _safe_relative(
        config.controller_receipt_relative,
        label="formal controller receipt",
    )
    _safe_relative(
        config.official_hippo_receipt_relative,
        label="official HippoRAG preflight receipt",
    )
    if (
        config.controller_receipt_relative.parts[0] != "manifests"
        or not _SAFE_SCHEMA_STATUS.fullmatch(config.controller_receipt_schema)
        or not _SAFE_SCHEMA_STATUS.fullmatch(config.controller_receipt_status)
        or config.official_hippo_receipt_relative.parts[0] != "manifests"
        or not _SAFE_SCHEMA_STATUS.fullmatch(
            config.official_hippo_receipt_schema
        )
        or not _SAFE_SCHEMA_STATUS.fullmatch(
            config.official_hippo_receipt_status
        )
        or config.official_hippo_receipt_schema
        != OFFICIAL_HIPPO_PREFLIGHT_SCHEMA
        or config.official_hippo_receipt_status
        != OFFICIAL_HIPPO_PREFLIGHT_STATUS
        or config.controller_receipt_relative
        == config.official_hippo_receipt_relative
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "formal controller receipt contract is invalid"
        )
    if (
        type(config.controller_timeout_seconds) is not int
        or config.controller_timeout_seconds < 1
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "formal controller timeout is invalid"
        )
    return project


def _source_download_artifacts_absent(project: Path) -> bool:
    return not any(
        path.exists() or path.is_symlink()
        for path in (
            project / SOURCE_ROOT_RELATIVE,
            project / SOURCE_ACQUISITION_ATTEMPT_RELATIVE,
            project / DOWNLOAD_RECEIPT_RELATIVE,
            project / SOURCE_ACQUISITION_FAILURE_RELATIVE,
        )
    )


def _consume_outer_marker(
    root: Path,
    config: OuterLifecycleConfig,
    *,
    outer_network_probe: Mapping[str, object],
) -> _Written:
    body = {
        "api_ruoli_proxy_credential_or_token_environment_read_count": 0,
        "controller_argument_template_sha256": _semantic_hash(
            list(_controller_argument_template(config))
        ),
        "controller_executable_sha256": (
            config.controller_executable_sha256
        ),
        "controller_module_sha256": config.controller_module_sha256,
        "env_executable_sha256": config.env_executable_sha256,
        "execution_freeze_self_sha256": (
            config.execution_freeze_self_sha256
        ),
        "implementation_freeze_self_sha256": (
            config.implementation_freeze_self_sha256
        ),
        "preexecution_runtime_disposition_self_sha256": (
            EXPECTED_RUNTIME_DISPOSITION_SELF_SHA256
        ),
        "official_hipporag_preflight_receipt_self_sha256": (
            config.official_hippo_receipt_self_sha256
        ),
        "official_hipporag_runtime_contract_sha256": _semantic_hash(
            _official_runtime_contract(config)
        ),
        "outer_network_isolation_probe": dict(outer_network_probe),
        "outer_network_isolation_probe_sha256": _semantic_hash(
            dict(outer_network_probe)
        ),
        "outer_restart_retry_replay_or_resample_count": 0,
        "schema": f"{VERSION}_one_shot_marker_v1",
        "stage_order": list(STAGE_ORDER),
        "status": "consumed_before_first_freeze_verification",
        "study_id": STUDY_ID,
        "source_acquisition_transient_unit_contract": (
            _source_acquisition_transient_unit_contract(
                Path(config.project_root).resolve(strict=True)
            )
        ),
        "systemd_run_sha256": config.systemd_run_sha256,
        "transient_unit_name": FORMAL_ACTION_UNIT_NAME,
        "transient_unit_properties": list(
            _resolved_unit_properties(
                Path(config.project_root).resolve(strict=True)
            )
        ),
    }
    return _write_exclusive(
        root / OUTER_MARKER_FILENAME, _self_hashed(body)
    )


def _consume_stage_marker(
    root: Path,
    *,
    config: OuterLifecycleConfig,
    bindings: Mapping[str, ReceiptBinding],
    stage: str,
    ordinal: int,
    outer_marker: _Written,
    prior_completion: _Written | None,
) -> _Written:
    body = {
        "outer_marker_file_sha256": outer_marker.file_sha256,
        "outer_marker_self_sha256": outer_marker.value["self_sha256"],
        "prior_stage_completion_file_sha256": (
            None if prior_completion is None else prior_completion.file_sha256
        ),
        "prior_stage_completion_self_sha256": (
            None
            if prior_completion is None
            else prior_completion.value["self_sha256"]
        ),
        "restart_retry_replay_or_resample_count": 0,
        "schema": f"{VERSION}_stage_marker_v1",
        "stage": stage,
        "stage_ordinal": ordinal,
        "status": "consumed_immediately_before_unique_stage_call",
        "study_id": STUDY_ID,
    }
    if stage == "post_selection_network_denied_formal_action":
        materialized, substitutions = _materialize_controller_arguments(
            config, bindings
        )
        body.update(
            {
                "controller_dynamic_argument_substitution_count": 2,
                "controller_dynamic_argument_substitution_sha256": (
                    _semantic_hash(dict(substitutions))
                ),
                "controller_materialized_argument_registry_sha256": (
                    _semantic_hash(list(materialized))
                ),
            }
        )
    path = (
        root
        / STAGE_DIRECTORY_NAME
        / f"{ordinal:02d}.{stage}.attempt.json"
    )
    return _write_exclusive(path, _self_hashed(body))


def _write_stage_completion(
    root: Path,
    *,
    stage: str,
    ordinal: int,
    marker: _Written,
    bindings: Sequence[ReceiptBinding],
) -> _Written:
    body = {
        "aggregate_receipt_bindings": [
            binding.payload() for binding in bindings
        ],
        "api_or_ruoli_environment_read_or_forward_count": 0,
        "restart_retry_replay_or_resample_count": 0,
        "schema": f"{VERSION}_stage_completion_v1",
        "stage": stage,
        "stage_marker_file_sha256": marker.file_sha256,
        "stage_marker_self_sha256": marker.value["self_sha256"],
        "stage_ordinal": ordinal,
        "status": "unique_stage_complete_receipt_bound",
        "study_id": STUDY_ID,
    }
    path = (
        root
        / STAGE_DIRECTORY_NAME
        / f"{ordinal:02d}.{stage}.complete.json"
    )
    return _write_exclusive(path, _self_hashed(body))


def _write_terminal_failure(
    root: Path,
    *,
    stage: str,
    completed_stage_count: int,
    marker: _Written | None,
    exc: BaseException,
) -> None:
    exception_class = (
        f"{type(exc).__module__}.{type(exc).__qualname__}".encode(
            "utf-8", errors="replace"
        )
    )
    body = {
        "action_gold_ledger_source_or_exception_message_included": False,
        "api_or_ruoli_environment_read_or_forward_count": 0,
        "completed_stage_count": completed_stage_count,
        "exception_class_sha256": hashlib.sha256(exception_class).hexdigest(),
        "failure_stage": stage,
        "restart_retry_replay_resample_provider_or_model_switch_count": 0,
        "schema": f"{VERSION}_terminal_failure_v1",
        "stage_marker_file_sha256": (
            None if marker is None else marker.file_sha256
        ),
        "stage_marker_self_sha256": (
            None if marker is None else marker.value["self_sha256"]
        ),
        "status": "terminal_failure_outer_attempt_consumed_no_restart",
        "study_id": STUDY_ID,
    }
    try:
        _write_exclusive(
            root / OUTER_FAILURE_FILENAME, _self_hashed(body)
        )
    except BaseException:
        pass


def run_outer_lifecycle(
    config: OuterLifecycleConfig,
    stages: LifecycleStages,
    *,
    outer_network_probe: Callable[
        [], Mapping[str, object]
    ] = _verify_outer_inet_denial_once,
) -> Mapping[str, Any]:
    """Consume the one remote outer attempt and execute all stages in order."""

    project = _validate_config(config)
    observed_outer_network_probe = outer_network_probe()
    if (
        not isinstance(observed_outer_network_probe, Mapping)
        or observed_outer_network_probe.get("probe_count") != 2
        or observed_outer_network_probe.get("status")
        != "AF_INET_and_AF_INET6_socket_creation_denied"
        or observed_outer_network_probe.get(
            "outer_network_isolation_contract"
        )
        != OUTER_NETWORK_ISOLATION_CONTRACT
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "outer network isolation probe contract drifted"
        )
    root = _create_outer_root(project)
    outer_marker: _Written | None = None
    current_marker: _Written | None = None
    completed: list[tuple[str, _Written]] = []
    receipt_bindings: dict[str, ReceiptBinding] = {}
    stage_name = "consume_outer_one_shot_marker"
    try:
        outer_marker = _consume_outer_marker(
            root,
            config,
            outer_network_probe=observed_outer_network_probe,
        )
        prior_completion: _Written | None = None
        for ordinal, (stage_name, runner) in enumerate(
            stages.ordered(), start=1
        ):
            context = StageContext(
                config=config,
                bindings=MappingProxyType(dict(receipt_bindings)),
            )
            current_marker = _consume_stage_marker(
                root,
                config=config,
                bindings=context.bindings,
                stage=stage_name,
                ordinal=ordinal,
                outer_marker=outer_marker,
                prior_completion=prior_completion,
            )
            if (
                stage_name
                == "public_synthetic_local_runtime_preflight"
                and not _source_download_artifacts_absent(project)
            ):
                raise MMQAP1RemoteOuterLifecycleError(
                    "formal source download preceded public preflight"
                )
            if (
                stage_name == "authorized_source_acquisition"
                and not REQUIRED_PREFLIGHT_BINDING_NAMES.issubset(
                    receipt_bindings
                )
            ):
                raise MMQAP1RemoteOuterLifecycleError(
                    "source acquisition requires both frozen public preflights"
                )
            specs = tuple(runner(context))
            if not specs:
                raise MMQAP1RemoteOuterLifecycleError(
                    "stage emitted no aggregate receipt binding"
                )
            if {spec.name for spec in specs} != set(
                EXPECTED_STAGE_BINDING_NAMES[stage_name]
            ):
                raise MMQAP1RemoteOuterLifecycleError(
                    "stage aggregate receipt registry drifted"
                )
            if (
                stage_name
                == "public_synthetic_local_runtime_preflight"
                and not _source_download_artifacts_absent(project)
            ):
                raise MMQAP1RemoteOuterLifecycleError(
                    "public preflight touched formal source download state"
                )
            stage_bindings: list[ReceiptBinding] = []
            for spec in specs:
                binding = _load_receipt_binding(project, spec)
                if (
                    binding.name
                    == "official_hipporag_runtime_binding_canary"
                    and config.official_hippo_receipt_self_sha256
                    is not None
                    and binding.self_sha256
                    != config.official_hippo_receipt_self_sha256
                ):
                    raise MMQAP1RemoteOuterLifecycleError(
                        "official HippoRAG preflight receipt hash drifted"
                    )
                if binding.name in receipt_bindings:
                    raise MMQAP1RemoteOuterLifecycleError(
                        "stage aggregate receipt name was reused"
                    )
                receipt_bindings[binding.name] = binding
                stage_bindings.append(binding)
            prior_completion = _write_stage_completion(
                root,
                stage=stage_name,
                ordinal=ordinal,
                marker=current_marker,
                bindings=stage_bindings,
            )
            completed.append((stage_name, prior_completion))
            current_marker = None

        body = {
            "api_or_ruoli_environment_read_or_forward_count": 0,
            "outer_marker_file_sha256": outer_marker.file_sha256,
            "outer_marker_self_sha256": outer_marker.value["self_sha256"],
            "restart_retry_replay_resample_provider_or_model_switch_count": 0,
            "schema": f"{VERSION}_success_v1",
            "stage_completion_chain": [
                {
                    "completion_file_sha256": completion.file_sha256,
                    "completion_self_sha256": completion.value["self_sha256"],
                    "stage": stage,
                }
                for stage, completion in completed
            ],
            "stage_count": len(completed),
            "status": "outer_lifecycle_complete_formal_action_terminal_bound",
            "study_id": STUDY_ID,
        }
        success = _self_hashed(body)
        _write_exclusive(root / OUTER_SUCCESS_FILENAME, success)
        return success
    except BaseException as exc:
        _write_terminal_failure(
            root,
            stage=stage_name,
            completed_stage_count=len(completed),
            marker=current_marker,
            exc=exc,
        )
        raise MMQAP1RemoteOuterLifecycleError(
            "MMQA P1 outer lifecycle failed closed"
        ) from None


def _lexical_path_sha256(path: Path) -> str:
    lexical = Path(path).expanduser().absolute()
    return hashlib.sha256(os.fsencode(str(lexical))).hexdigest()


def _project_relative_no_symlink(project: Path, path: Path) -> str:
    lexical = Path(path).absolute()
    try:
        resolved = lexical.resolve(strict=True)
        relative = resolved.relative_to(project)
    except (OSError, ValueError) as exc:
        raise MMQAP1RemoteOuterLifecycleError(
            "frozen implementation path is outside the project"
        ) from exc
    if lexical != resolved:
        raise MMQAP1RemoteOuterLifecycleError(
            "frozen implementation path contains a symlink"
        )
    return _safe_relative(relative, label="frozen implementation")


def _verify_frozen_inventory(
    project: Path,
    manifest: Mapping[str, Any],
    *,
    required_relatives: Sequence[str] = (),
) -> dict[str, str]:
    inventory = manifest.get("implementation_files")
    if (
        not isinstance(inventory, Mapping)
        or not inventory
        or len(inventory) > 512
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "implementation freeze inventory is invalid"
        )
    normalized: dict[str, str] = {}
    resolved_registry: set[Path] = set()
    for raw_relative, expected_sha256 in inventory.items():
        if not isinstance(raw_relative, str):
            raise MMQAP1RemoteOuterLifecycleError(
                "implementation freeze path is invalid"
            )
        relative = _safe_relative(
            Path(raw_relative), label="implementation freeze"
        )
        if (
            not isinstance(expected_sha256, str)
            or _HEX64.fullmatch(expected_sha256) is None
            or relative in normalized
        ):
            raise MMQAP1RemoteOuterLifecycleError(
                "implementation freeze file binding is invalid"
            )
        lexical = project / relative
        try:
            resolved = lexical.resolve(strict=True)
            resolved.relative_to(project)
        except (OSError, ValueError) as exc:
            raise MMQAP1RemoteOuterLifecycleError(
                "implementation freeze file is unavailable"
            ) from exc
        if lexical != resolved or resolved in resolved_registry:
            raise MMQAP1RemoteOuterLifecycleError(
                "implementation freeze file path is aliased or symlinked"
            )
        observed = _sha256_regular_file(resolved)
        if observed != expected_sha256:
            raise MMQAP1RemoteOuterLifecycleError(
                "implementation freeze dependency file drifted"
            )
        resolved_registry.add(resolved)
        normalized[relative] = observed
    required = set(REQUIRED_IMPLEMENTATION_RELATIVES).union(
        required_relatives
    )
    if not required.issubset(normalized):
        raise MMQAP1RemoteOuterLifecycleError(
            "implementation freeze required dependency is absent"
        )
    return normalized


def _runtime_file_binding(
    path: Path,
    *,
    resolved_file_sha256: str,
) -> dict[str, str]:
    try:
        resolved = Path(path).expanduser().absolute().resolve(strict=True)
    except OSError as exc:
        raise MMQAP1RemoteOuterLifecycleError(
            "execution freeze runtime file is unavailable"
        ) from exc
    observed = _sha256_regular_file(resolved)
    if observed != resolved_file_sha256:
        raise MMQAP1RemoteOuterLifecycleError(
            "execution freeze runtime file identity drifted"
        )
    return {
        "lexical_path_sha256": _lexical_path_sha256(path),
        "resolved_file_sha256": observed,
    }


def _verify_execution_freeze(
    project: Path,
    value: Mapping[str, Any],
    config: OuterLifecycleConfig,
    *,
    implementation_inventory: Mapping[str, str],
) -> str:
    required = {
        "implementation_freeze_self_sha256": (
            config.implementation_freeze_self_sha256
        ),
        "preexecution_runtime_disposition_self_sha256": (
            EXPECTED_RUNTIME_DISPOSITION_SELF_SHA256
        ),
        "source_custody_self_sha256": EXPECTED_CUSTODY_SELF_SHA256,
        "study_design_self_sha256": EXPECTED_DESIGN_SELF_SHA256,
        "download_authorization_self_sha256": (
            EXPECTED_AUTHORIZATION_SELF_SHA256
        ),
        "stage_order": list(STAGE_ORDER),
        "execution_policy": EXECUTION_POLICY,
        "controller_argument_template_sha256": _semantic_hash(
            list(_controller_argument_template(config))
        ),
        "formal_child_environment_sha256": _semantic_hash(
            _formal_child_environment(project)
        ),
        "formal_action_transient_unit_contract": (
            _transient_unit_contract(project)
        ),
        "outer_network_isolation_contract": (
            OUTER_NETWORK_ISOLATION_CONTRACT
        ),
        "official_hipporag_runtime_contract": (
            _official_runtime_contract(config)
        ),
        "source_acquisition_child_environment_sha256": _semantic_hash(
            _source_acquisition_child_environment(project)
        ),
        "source_acquisition_transient_unit_contract": (
            _source_acquisition_transient_unit_contract(project)
        ),
        "systemd_client_environment_sha256": _semantic_hash(
            _systemd_client_environment()
        ),
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise MMQAP1RemoteOuterLifecycleError(
                "execution freeze policy or upstream binding drifted"
            )
    receipt_contract = value.get("formal_controller_receipt_contract")
    if receipt_contract != {
        "relative_path": _safe_relative(
            config.controller_receipt_relative,
            label="formal controller receipt",
        ),
        "schema": config.controller_receipt_schema,
        "status": config.controller_receipt_status,
    }:
        raise MMQAP1RemoteOuterLifecycleError(
            "execution freeze controller receipt contract drifted"
        )
    official_receipt_contract = value.get(
        "official_hipporag_preflight_receipt_contract"
    )
    if official_receipt_contract != {
        "relative_path": _safe_relative(
            config.official_hippo_receipt_relative,
            label="official HippoRAG preflight receipt",
        ),
        "schema": config.official_hippo_receipt_schema,
        "self_hash_field": "receipt_sha256",
        "self_sha256": config.official_hippo_receipt_self_sha256,
        "status": config.official_hippo_receipt_status,
    }:
        raise MMQAP1RemoteOuterLifecycleError(
            "execution freeze official HippoRAG receipt contract drifted"
        )
    official_adapter = value.get(
        "official_hipporag_adapter_relative_path"
    )
    if not isinstance(official_adapter, str):
        raise MMQAP1RemoteOuterLifecycleError(
            "execution freeze official HippoRAG adapter is absent"
        )
    official_relative = _safe_relative(
        Path(official_adapter), label="official HippoRAG adapter"
    )
    if official_relative not in implementation_inventory:
        raise MMQAP1RemoteOuterLifecycleError(
            "official HippoRAG adapter is not implementation-frozen"
        )

    module_relative = _project_relative_no_symlink(
        project, config.controller_module
    )
    runtime = value.get("runtime_path_bindings")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "controller_executable",
        "controller_module",
        "cross_encoder_model",
        "env_executable",
        "minilm_model",
        "nvidia_smi",
        "systemd_run",
        "typed_python",
    }:
        raise MMQAP1RemoteOuterLifecycleError(
            "execution freeze runtime registry drifted"
        )
    expected_rows: dict[str, Mapping[str, str]] = {
        "controller_executable": _runtime_file_binding(
            config.controller_executable,
            resolved_file_sha256=config.controller_executable_sha256,
        ),
        "controller_module": {
            "file_sha256": config.controller_module_sha256,
            "project_relative_path": module_relative,
        },
        "cross_encoder_model": {
            "lexical_path_sha256": _lexical_path_sha256(
                config.cross_encoder_model
            )
        },
        "env_executable": _runtime_file_binding(
            ENV_PATH,
            resolved_file_sha256=config.env_executable_sha256,
        ),
        "minilm_model": {
            "lexical_path_sha256": _lexical_path_sha256(
                config.minilm_model
            )
        },
        "nvidia_smi": _runtime_file_binding(
            config.nvidia_smi,
            resolved_file_sha256=str(
                runtime.get("nvidia_smi", {}).get(
                    "resolved_file_sha256", ""
                )
            )
            if isinstance(runtime.get("nvidia_smi"), Mapping)
            else "",
        ),
        "systemd_run": _runtime_file_binding(
            SYSTEMD_RUN_PATH,
            resolved_file_sha256=config.systemd_run_sha256,
        ),
        "typed_python": _runtime_file_binding(
            config.typed_python,
            resolved_file_sha256=str(
                runtime.get("typed_python", {}).get(
                    "resolved_file_sha256", ""
                )
            )
            if isinstance(runtime.get("typed_python"), Mapping)
            else "",
        ),
    }
    module_observed = _sha256_regular_file(project / module_relative)
    if (
        module_observed != config.controller_module_sha256
        or implementation_inventory.get(module_relative)
        != config.controller_module_sha256
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "formal controller module implementation binding drifted"
        )
    for role, expected in expected_rows.items():
        if runtime.get(role) != expected:
            raise MMQAP1RemoteOuterLifecycleError(
                "execution freeze runtime path identity drifted"
            )
    return official_relative


def _freeze_stage(context: StageContext) -> Sequence[ReceiptSpec]:
    config = context.config
    project = Path(config.project_root).resolve(strict=True)
    execution_spec = ReceiptSpec(
        name="execution_freeze",
        relative_path=EXECUTION_FREEZE_RELATIVE,
        expected_schema="mmqa_p1_execution_freeze_v1",
        expected_status="frozen_before_outer_one_shot",
        expected_self_sha256=config.execution_freeze_self_sha256,
    )
    implementation_spec = ReceiptSpec(
        name="implementation_freeze",
        relative_path=IMPLEMENTATION_FREEZE_RELATIVE,
        expected_schema="mmqa_p1_implementation_freeze_v1",
        expected_status="frozen_before_execution_freeze",
        expected_self_sha256=config.implementation_freeze_self_sha256,
    )
    disposition_spec = ReceiptSpec(
        name="preexecution_runtime_disposition",
        relative_path=RUNTIME_DISPOSITION_RELATIVE,
        expected_schema="mmqa_p1_preexecution_runtime_disposition_v1",
        expected_status=(
            "prospective_runtime_and_contract_clarification_frozen_before_formal_source_download_or_parse"
        ),
        expected_self_sha256=(
            EXPECTED_RUNTIME_DISPOSITION_SELF_SHA256
        ),
        required_mode=0o644,
    )
    _disposition, _disposition_binding = (
        _load_receipt_value_and_binding(project, disposition_spec)
    )
    implementation, _implementation_binding = (
        _load_receipt_value_and_binding(project, implementation_spec)
    )
    execution, _execution_binding = _load_receipt_value_and_binding(
        project, execution_spec
    )
    official_adapter = execution.get(
        "official_hipporag_adapter_relative_path"
    )
    if not isinstance(official_adapter, str):
        raise MMQAP1RemoteOuterLifecycleError(
            "execution freeze official HippoRAG adapter is absent"
        )
    module_relative = _project_relative_no_symlink(
        project, config.controller_module
    )
    inventory = _verify_frozen_inventory(
        project,
        implementation,
        required_relatives=(
            module_relative,
            _safe_relative(
                Path(official_adapter),
                label="official HippoRAG adapter",
            ),
        ),
    )
    _verify_execution_freeze(
        project,
        execution,
        config,
        implementation_inventory=inventory,
    )
    return execution_spec, implementation_spec


def _preflight_stage(context: StageContext) -> Sequence[ReceiptSpec]:
    from assumption_agent.benchmarks import (
        mmqa_p1_local_runtime_preflight_v1 as preflight,
    )

    project = Path(context.config.project_root).resolve(strict=True)
    receipt = preflight.run_preflight(
        typed_python=context.config.typed_python,
        minilm_model=context.config.minilm_model,
        cross_encoder_model=context.config.cross_encoder_model,
        nvidia_smi=context.config.nvidia_smi,
    )
    preflight._write_exclusive(project / PREFLIGHT_RECEIPT_RELATIVE, receipt)
    return (
        ReceiptSpec(
            name="public_synthetic_local_runtime_preflight",
            relative_path=PREFLIGHT_RECEIPT_RELATIVE,
            expected_schema=preflight.RECEIPT_SCHEMA,
            expected_status=(
                "passed_public_synthetic_non_scoring_runtime_action_preflight"
            ),
            required_mode=0o600,
        ),
    )


def _sealed_official_hippo_preflight_spec(
    context: StageContext,
) -> Sequence[ReceiptSpec]:
    config = context.config
    if config.official_hippo_receipt_self_sha256 is None:
        raise MMQAP1RemoteOuterLifecycleError(
            "official HippoRAG preflight runner or sealed hash is required"
        )
    return (
        ReceiptSpec(
            name="official_hipporag_runtime_binding_canary",
            relative_path=config.official_hippo_receipt_relative,
            expected_schema=config.official_hippo_receipt_schema,
            expected_status=config.official_hippo_receipt_status,
            self_hash_field="receipt_sha256",
            expected_self_sha256=(
                config.official_hippo_receipt_self_sha256
            ),
            required_mode=0o600,
        ),
    )


def _live_validate_official_hippo_preflight(
    context: StageContext,
    spec: ReceiptSpec,
) -> str:
    from assumption_agent.benchmarks import (
        mmqa_p1_official_hipporag_block_v1 as official_hippo,
    )

    project = Path(context.config.project_root).resolve(strict=True)
    value, receipt_binding = _load_receipt_value_and_binding(
        project, spec
    )
    if (
        spec.name != "official_hipporag_runtime_binding_canary"
        or spec.self_hash_field != "receipt_sha256"
        or value.get("expected_package_versions")
        != dict(context.config.official_expected_package_versions)
        or value.get("expected_module_import_roots")
        != dict(context.config.official_expected_module_import_roots)
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "official HippoRAG preflight frozen contract drifted"
        )
    try:
        binding = official_hippo.load_fresh_preflight_binding(
            project / spec.relative_path,
            paths=_official_runtime_paths(context.config),
            expected_receipt_sha256=receipt_binding.self_sha256,
            filesystem_inspector=(
                official_hippo.production_filesystem_inspector
            ),
            isolation_inspector=(
                official_hippo.production_address_family_isolation_probe
            ),
        )
    except official_hippo.MmqaP1OfficialHippoRAGBlockError as exc:
        raise MMQAP1RemoteOuterLifecycleError(
            "official HippoRAG live filesystem preflight failed"
        ) from exc
    if binding.receipt_sha256 != receipt_binding.self_sha256:
        raise MMQAP1RemoteOuterLifecycleError(
            "official HippoRAG live binding identity drifted"
        )
    return binding.binding_sha256


def _acquisition_stage(
    context: StageContext,
    *,
    process_runner: ProcessRunner = subprocess.run,
) -> Sequence[ReceiptSpec]:
    config = context.config
    project = Path(config.project_root).resolve(strict=True)
    executable, module = _verify_source_acquisition_capability(
        config, project
    )
    child_environment = _source_acquisition_child_environment(project)
    command = _build_source_acquisition_command(
        project=project,
        executable=executable,
        module=module,
        environment=child_environment,
    )
    result = process_runner(
        command,
        cwd=project,
        env=_systemd_client_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=SOURCE_ACQUISITION_TIMEOUT_SECONDS,
        check=False,
    )
    returncode = getattr(result, "returncode", None)
    if type(returncode) is not int or returncode != 0:
        raise MMQAP1RemoteOuterLifecycleError(
            "authorized source acquisition transient service failed"
        )
    return (
        ReceiptSpec(
            name="source_download",
            relative_path=DOWNLOAD_RECEIPT_RELATIVE,
            expected_schema="mmqa_p1_source_download_receipt_v1",
            expected_status=(
                "four_fixed_sources_downloaded_identity_verified_not_parsed"
            ),
            required_mode=0o600,
        ),
    )


def _qualification_freeze_stage(
    context: StageContext,
) -> Sequence[ReceiptSpec]:
    from assumption_agent.benchmarks import (
        mmqa_p1_source_qualification_freeze_v1,
    )

    mmqa_p1_source_qualification_freeze_v1.build_qualification_freeze(
        context.config.project_root
    )
    return (
        ReceiptSpec(
            name="source_qualification_freeze",
            relative_path=QUALIFICATION_FREEZE_RELATIVE,
            expected_schema="mmqa_p1_source_qualification_freeze_v1",
            expected_status="frozen_before_unique_formal_qualification",
            required_mode=0o600,
        ),
    )


def _qualification_stage(context: StageContext) -> Sequence[ReceiptSpec]:
    from assumption_agent.benchmarks import mmqa_p1_source_qualification_v1

    project = Path(context.config.project_root).resolve(strict=True)
    module_project = mmqa_p1_source_qualification_v1.PROJECT_ROOT.resolve(
        strict=True
    )
    if project != module_project:
        raise MMQAP1RemoteOuterLifecycleError(
            "qualifier module project root drifted"
        )
    mmqa_p1_source_qualification_v1.run_formal_qualification()
    return (
        ReceiptSpec(
            name="source_qualification_result",
            relative_path=QUALIFICATION_RESULT_RELATIVE,
            expected_schema="mmqa_p1_source_qualification_v1_result_v1",
            expected_status="qualified_aggregate_only",
            required_mode=0o600,
        ),
    )


def _selection_stage(context: StageContext) -> Sequence[ReceiptSpec]:
    from assumption_agent.benchmarks import mmqa_p1_private_selection_v1

    qualification = context.bindings.get("source_qualification_result")
    if qualification is None:
        raise MMQAP1RemoteOuterLifecycleError(
            "qualification receipt is not bound before selection"
        )
    mmqa_p1_private_selection_v1.run_formal_selection(
        context.config.project_root,
        expected_qualification_self_sha256=qualification.self_sha256,
    )
    return (
        ReceiptSpec(
            name="private_selection_public_receipt",
            relative_path=SELECTION_RECEIPT_RELATIVE,
            expected_schema="mmqa_p1_private_selection_v1_public_receipt_v1",
            expected_status="private_one_shot_selection_complete",
            self_hash_field="acquisition_sha256",
            required_mode=0o644,
        ),
    )


def _formal_child_environment(project: Path) -> dict[str, str]:
    environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "HOME": "/nonexistent-mmqa-p1-formal-home",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "OMP_NUM_THREADS": "4",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(project),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if set(environment) != _FIXED_CHILD_ENV_NAMES or any(
        fragment in key.upper()
        for key in environment
        for fragment in _FORBIDDEN_CHILD_ENV_FRAGMENTS
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "formal child environment whitelist drifted"
        )
    return environment


def _source_acquisition_child_environment(
    project: Path,
) -> dict[str, str]:
    environment = {
        "HOME": "/nonexistent-mmqa-p1-source-home",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(project),
    }
    if (
        set(environment) != _FIXED_SOURCE_ACQUISITION_ENV_NAMES
        or any(
            fragment in key.upper()
            for key in environment
            for fragment in _FORBIDDEN_CHILD_ENV_FRAGMENTS
        )
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "source acquisition child environment whitelist drifted"
        )
    return environment


def _systemd_client_environment() -> dict[str, str]:
    """Build the fixed environment needed only to reach the user manager."""

    runtime_directory = f"/run/user/{os.getuid()}"
    environment = {
        "DBUS_SESSION_BUS_ADDRESS": (
            f"unix:path={runtime_directory}/bus"
        ),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "XDG_RUNTIME_DIR": runtime_directory,
    }
    if set(environment) != _FIXED_SYSTEMD_CLIENT_ENV_NAMES or any(
        fragment in key.upper()
        for key in environment
        for fragment in _FORBIDDEN_CHILD_ENV_FRAGMENTS
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "systemd client environment whitelist drifted"
        )
    return environment


def _verify_controller_capability(
    config: OuterLifecycleConfig, project: Path
) -> tuple[Path, Path]:
    if SYSTEMD_RUN_PATH != Path("/usr/bin/systemd-run"):
        raise MMQAP1RemoteOuterLifecycleError("systemd-run path drifted")
    if ENV_PATH != Path("/usr/bin/env"):
        raise MMQAP1RemoteOuterLifecycleError("env executable path drifted")
    if (
        _sha256_regular_file(SYSTEMD_RUN_PATH)
        != config.systemd_run_sha256
        or _sha256_regular_file(ENV_PATH)
        != config.env_executable_sha256
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "transient-service launcher identity drifted"
        )
    executable_lexical = Path(config.controller_executable).absolute()
    try:
        executable_resolved = executable_lexical.resolve(strict=True)
        module_resolved = Path(config.controller_module).resolve(strict=True)
    except OSError as exc:
        raise MMQAP1RemoteOuterLifecycleError(
            "formal controller capability is unavailable"
        ) from exc
    if (
        _sha256_regular_file(executable_resolved)
        != config.controller_executable_sha256
        or _sha256_regular_file(module_resolved)
        != config.controller_module_sha256
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "formal controller capability identity drifted"
        )
    try:
        module_resolved.relative_to(project)
    except ValueError as exc:
        raise MMQAP1RemoteOuterLifecycleError(
            "formal controller module is outside the frozen project"
        ) from exc
    return executable_lexical, module_resolved


def _verify_source_acquisition_capability(
    config: OuterLifecycleConfig,
    project: Path,
) -> tuple[Path, Path]:
    executable, _formal_module = _verify_controller_capability(
        config, project
    )
    if executable != Path(config.typed_python).absolute():
        raise MMQAP1RemoteOuterLifecycleError(
            "source acquisition executable binding drifted"
        )
    implementation_spec = ReceiptSpec(
        name="implementation_freeze",
        relative_path=IMPLEMENTATION_FREEZE_RELATIVE,
        expected_schema="mmqa_p1_implementation_freeze_v1",
        expected_status="frozen_before_execution_freeze",
        expected_self_sha256=config.implementation_freeze_self_sha256,
    )
    implementation, _binding = _load_receipt_value_and_binding(
        project, implementation_spec
    )
    inventory = _verify_frozen_inventory(project, implementation)
    relative = SOURCE_ACQUISITION_MODULE_RELATIVE.as_posix()
    module = project / SOURCE_ACQUISITION_MODULE_RELATIVE
    if (
        relative not in inventory
        or _project_relative_no_symlink(project, module) != relative
        or _sha256_regular_file(module) != inventory[relative]
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "source acquisition module implementation drifted"
        )
    return executable, module


def _materialize_controller_arguments(
    config: OuterLifecycleConfig,
    bindings: Mapping[str, ReceiptBinding],
) -> tuple[tuple[str, ...], Mapping[str, str]]:
    local = bindings.get("public_synthetic_local_runtime_preflight")
    selection = bindings.get("private_selection_public_receipt")
    if (
        local is None
        or local.self_hash_field != "self_sha256"
        or _HEX64.fullmatch(local.self_sha256) is None
        or selection is None
        or selection.self_hash_field != "acquisition_sha256"
        or _HEX64.fullmatch(selection.self_sha256) is None
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "formal controller dynamic receipt binding drifted"
        )
    _controller_argument_template(config)
    substitutions = {
        LOCAL_PREFLIGHT_SELF_SHA256_PLACEHOLDER: local.self_sha256,
        SELECTION_ACQUISITION_SHA256_PLACEHOLDER: selection.self_sha256,
    }
    arguments = tuple(
        substitutions.get(argument, argument)
        for argument in config.controller_arguments
    )
    local_index = (
        config.controller_arguments.index(
            "--local-preflight-self-sha256"
        )
        + 1
    )
    selection_index = (
        config.controller_arguments.index(
            "--selection-acquisition-sha256"
        )
        + 1
    )
    execution_index = (
        config.controller_arguments.index(
            "--execution-freeze-self-sha256"
        )
        + 1
    )
    if (
        len(substitutions) != 2
        or any(
            placeholder in arguments
            for placeholder in substitutions
        )
        or arguments[local_index] != local.self_sha256
        or arguments[selection_index] != selection.self_sha256
        or arguments[execution_index]
        != config.execution_freeze_self_sha256
    ):
        raise MMQAP1RemoteOuterLifecycleError(
            "formal controller dynamic argument substitution drifted"
        )
    return arguments, MappingProxyType(substitutions)


def _build_source_acquisition_command(
    *,
    project: Path,
    executable: Path,
    module: Path,
    environment: Mapping[str, str],
) -> list[str]:
    command = [
        str(SYSTEMD_RUN_PATH),
        "--user",
        "--wait",
        "--collect",
        "--quiet",
        f"--unit={SOURCE_ACQUISITION_UNIT_NAME}",
        f"--working-directory={project}",
    ]
    command.extend(
        f"--property={property_value}"
        for property_value in (
            _resolved_source_acquisition_unit_properties(project)
        )
    )
    command.extend(("--", str(ENV_PATH), "-i"))
    command.extend(
        f"{name}={environment[name]}" for name in sorted(environment)
    )
    command.extend(
        (str(executable), str(module), "--project", str(project))
    )
    return command


def _build_systemd_run_command(
    config: OuterLifecycleConfig,
    *,
    project: Path,
    executable: Path,
    module: Path,
    environment: Mapping[str, str],
    controller_arguments: Sequence[str],
) -> list[str]:
    command = [
        str(SYSTEMD_RUN_PATH),
        "--user",
        "--wait",
        "--collect",
        "--quiet",
        f"--unit={FORMAL_ACTION_UNIT_NAME}",
        f"--working-directory={project}",
    ]
    command.extend(
        f"--property={property_value}"
        for property_value in _resolved_unit_properties(project)
    )
    command.extend(("--", str(ENV_PATH), "-i"))
    for name in sorted(environment):
        command.append(f"{name}={environment[name]}")
    command.extend(
        (
            str(executable),
            str(module),
            *controller_arguments,
        )
    )
    return command


def _post_selection_stage(
    context: StageContext,
    *,
    process_runner: ProcessRunner = subprocess.run,
) -> Sequence[ReceiptSpec]:
    if "private_selection_public_receipt" not in context.bindings:
        raise MMQAP1RemoteOuterLifecycleError(
            "private selection is not bound before formal action"
        )
    config = context.config
    project = Path(config.project_root).resolve(strict=True)
    executable, module = _verify_controller_capability(config, project)
    controller_arguments, _substitutions = (
        _materialize_controller_arguments(config, context.bindings)
    )
    child_environment = _formal_child_environment(project)
    client_environment = _systemd_client_environment()
    command = _build_systemd_run_command(
        config,
        project=project,
        executable=executable,
        module=module,
        environment=child_environment,
        controller_arguments=controller_arguments,
    )
    result = process_runner(
        command,
        cwd=project,
        env=client_environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=config.controller_timeout_seconds,
        check=False,
    )
    returncode = getattr(result, "returncode", None)
    if type(returncode) is not int or returncode != 0:
        raise MMQAP1RemoteOuterLifecycleError(
            "network-denied formal controller exited unsuccessfully"
        )
    return (
        ReceiptSpec(
            name="post_selection_formal_action_terminal",
            relative_path=config.controller_receipt_relative,
            expected_schema=config.controller_receipt_schema,
            expected_status=config.controller_receipt_status,
            required_mode=0o600,
        ),
    )


def production_stages(
    *,
    process_runner: ProcessRunner = subprocess.run,
    official_hippo_preflight: StageRunner | None = None,
) -> LifecycleStages:
    """Return production adapters with injectable official-Hippo canary.

    Until the fresh official adapter module is installed, the default binds a
    separately sealed, execution-freeze-bound aggregate canary receipt.  The
    injected runner replaces only that second preflight receipt producer; the
    local MiniLM/CE preflight always remains first.
    """

    def post_selection(context: StageContext) -> Sequence[ReceiptSpec]:
        return _post_selection_stage(
            context, process_runner=process_runner
        )

    def acquisition(context: StageContext) -> Sequence[ReceiptSpec]:
        return _acquisition_stage(
            context, process_runner=process_runner
        )

    def both_preflights(context: StageContext) -> Sequence[ReceiptSpec]:
        local_specs = tuple(_preflight_stage(context))
        if not _source_download_artifacts_absent(
            Path(context.config.project_root).resolve(strict=True)
        ):
            raise MMQAP1RemoteOuterLifecycleError(
                "local preflight touched formal source download state"
            )
        official_runner = (
            _sealed_official_hippo_preflight_spec
            if official_hippo_preflight is None
            else official_hippo_preflight
        )
        official_specs = tuple(official_runner(context))
        if (
            len(official_specs) != 1
            or official_specs[0].name
            != "official_hipporag_runtime_binding_canary"
        ):
            raise MMQAP1RemoteOuterLifecycleError(
                "official HippoRAG preflight receipt registry drifted"
            )
        _live_validate_official_hippo_preflight(
            context, official_specs[0]
        )
        if not _source_download_artifacts_absent(
            Path(context.config.project_root).resolve(strict=True)
        ):
            raise MMQAP1RemoteOuterLifecycleError(
                "official preflight touched formal source download state"
            )
        return local_specs + official_specs

    return LifecycleStages(
        verify_execution_and_implementation_freezes=_freeze_stage,
        public_synthetic_local_runtime_preflight=both_preflights,
        authorized_source_acquisition=acquisition,
        source_qualification_freeze=_qualification_freeze_stage,
        aggregate_source_qualification=_qualification_stage,
        private_one_shot_selection=_selection_stage,
        post_selection_network_denied_formal_action=post_selection,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--execution-freeze-self-sha256", required=True)
    parser.add_argument("--implementation-freeze-self-sha256", required=True)
    parser.add_argument("--typed-python", required=True, type=Path)
    parser.add_argument("--minilm-model", required=True, type=Path)
    parser.add_argument("--cross-encoder-model", required=True, type=Path)
    parser.add_argument(
        "--nvidia-smi", type=Path, default=Path("/usr/bin/nvidia-smi")
    )
    parser.add_argument("--systemd-run-sha256", required=True)
    parser.add_argument("--env-executable-sha256", required=True)
    parser.add_argument("--controller-executable", required=True, type=Path)
    parser.add_argument("--controller-executable-sha256", required=True)
    parser.add_argument("--controller-module", required=True, type=Path)
    parser.add_argument("--controller-module-sha256", required=True)
    parser.add_argument(
        "--controller-arg",
        action="append",
        default=[],
        help="one exact formal-controller argument; repeat to preserve order",
    )
    parser.add_argument(
        "--official-hippo-receipt-relative", required=True, type=Path
    )
    parser.add_argument("--official-hippo-receipt-schema", required=True)
    parser.add_argument("--official-hippo-receipt-status", required=True)
    parser.add_argument(
        "--official-hippo-receipt-self-sha256", required=True
    )
    parser.add_argument("--official-runtime-python", required=True, type=Path)
    parser.add_argument("--official-pyvenv-cfg", required=True, type=Path)
    parser.add_argument("--official-overlay-root", required=True, type=Path)
    parser.add_argument(
        "--official-hipporag-source-root", required=True, type=Path
    )
    parser.add_argument(
        "--official-p16-site-root", required=True, type=Path
    )
    parser.add_argument(
        "--official-local-llm-model", required=True, type=Path
    )
    parser.add_argument(
        "--official-local-embedding-model", required=True, type=Path
    )
    parser.add_argument(
        "--official-expected-package-versions-json", required=True
    )
    parser.add_argument(
        "--official-expected-module-import-roots-json", required=True
    )
    parser.add_argument(
        "--controller-receipt-relative", required=True, type=Path
    )
    parser.add_argument("--controller-receipt-schema", required=True)
    parser.add_argument("--controller-receipt-status", required=True)
    parser.add_argument(
        "--controller-timeout-seconds",
        type=int,
        default=DEFAULT_CONTROLLER_TIMEOUT_SECONDS,
    )
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        expected_versions = json.loads(
            arguments.official_expected_package_versions_json
        )
        expected_roots = json.loads(
            arguments.official_expected_module_import_roots_json
        )
    except json.JSONDecodeError:
        print("MMQA P1 official runtime map JSON is invalid.", file=sys.stderr)
        return 2
    if not isinstance(expected_versions, dict) or not isinstance(
        expected_roots, dict
    ):
        print("MMQA P1 official runtime maps are invalid.", file=sys.stderr)
        return 2
    config = OuterLifecycleConfig(
        project_root=arguments.project,
        execution_freeze_self_sha256=(
            arguments.execution_freeze_self_sha256
        ),
        implementation_freeze_self_sha256=(
            arguments.implementation_freeze_self_sha256
        ),
        typed_python=arguments.typed_python,
        minilm_model=arguments.minilm_model,
        cross_encoder_model=arguments.cross_encoder_model,
        nvidia_smi=arguments.nvidia_smi,
        systemd_run_sha256=arguments.systemd_run_sha256,
        env_executable_sha256=arguments.env_executable_sha256,
        controller_executable=arguments.controller_executable,
        controller_executable_sha256=(
            arguments.controller_executable_sha256
        ),
        controller_module=arguments.controller_module,
        controller_module_sha256=arguments.controller_module_sha256,
        controller_arguments=tuple(arguments.controller_arg),
        official_hippo_receipt_relative=(
            arguments.official_hippo_receipt_relative
        ),
        official_hippo_receipt_schema=(
            arguments.official_hippo_receipt_schema
        ),
        official_hippo_receipt_status=(
            arguments.official_hippo_receipt_status
        ),
        official_hippo_receipt_self_sha256=(
            arguments.official_hippo_receipt_self_sha256
        ),
        official_runtime_python=arguments.official_runtime_python,
        official_pyvenv_cfg=arguments.official_pyvenv_cfg,
        official_overlay_root=arguments.official_overlay_root,
        official_hipporag_source_root=(
            arguments.official_hipporag_source_root
        ),
        official_p16_site_root=arguments.official_p16_site_root,
        official_local_llm_model=arguments.official_local_llm_model,
        official_local_embedding_model=(
            arguments.official_local_embedding_model
        ),
        official_expected_package_versions=expected_versions,
        official_expected_module_import_roots=expected_roots,
        controller_receipt_relative=arguments.controller_receipt_relative,
        controller_receipt_schema=arguments.controller_receipt_schema,
        controller_receipt_status=arguments.controller_receipt_status,
        controller_timeout_seconds=arguments.controller_timeout_seconds,
    )
    try:
        result = run_outer_lifecycle(config, production_stages())
    except MMQAP1RemoteOuterLifecycleError:
        print(
            "MMQA P1 remote outer lifecycle failed closed; no restart is authorized.",
            file=sys.stderr,
        )
        return 2
    print(_canonical_bytes(result).decode("ascii"), end="")
    return 0


def main() -> int:
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ENV_PATH",
    "SYSTEMD_RUN_PATH",
    "LifecycleStages",
    "MMQAP1RemoteOuterLifecycleError",
    "OUTER_ROOT_RELATIVE",
    "OuterLifecycleConfig",
    "PREFLIGHT_RECEIPT_RELATIVE",
    "ReceiptBinding",
    "ReceiptSpec",
    "SOURCE_ROOT_RELATIVE",
    "STAGE_ORDER",
    "StageContext",
    "VERSION",
    "production_stages",
    "run_outer_lifecycle",
]
