#!/usr/bin/env python3
"""One-shot formal source-free qualification for the UAO v2 compiler.

The controller accepts only one self-hashed freeze, verifies the complete
implementation closure, and launches exactly two sequential isolated worker
processes with one exact registered Python executable.  Workers run the frozen synthetic
qualifier and may emit only one canonical ASCII JSON receipt.  PASS requires
byte-identical receipts and the frozen 5x2 known-mechanism, real active/no-op,
tamper-rejection, and zero-capability-access claims.

This controller has no real source, network, model, API, online evaluator,
retry, replay, resampling, repair, or alternative-candidate channel.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import socket
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


VERSION = "meta_assumption_source_free_qualification_controller_v2"
STUDY_ID = "UAO_P2_SPARSE_RELATIONAL_DECISION_COMPILER_V2"
FREEZE_SCHEMA = "meta_assumption_source_free_qualification_freeze_v2"
FREEZE_FILENAME = f"{FREEZE_SCHEMA}.json"
FORMAL_ROOT = Path(
    "/home/erzhu419/uao_p2_20260729/source_free_qualification_v2"
)
FROZEN_PYTHON = Path("/usr/bin/python3.12")
FORMAL_SERVICE_UNIT = (
    "meta-assumption-source-free-qualification-v2.service"
)
INSTALLED_USER_SERVICE_PATH = (
    Path("/home/erzhu419/.config/systemd/user") / FORMAL_SERVICE_UNIT
)
FROZEN_RUNTIME_IDENTITIES = {
    "/usr/bin/python3.10": {
        "python_executable_sha256": (
            "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
        ),
        "python_executable_size_bytes": 5917224,
        "python_executable_mode": "0755",
        "python_version": "3.10.12",
    },
    "/usr/bin/python3.12": {
        "python_executable_sha256": (
            "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
        ),
        "python_executable_size_bytes": 8020928,
        "python_executable_mode": "0755",
        "python_version": "3.12.3",
    },
}
FROZEN_RUNTIME_IDENTITY = FROZEN_RUNTIME_IDENTITIES[str(FROZEN_PYTHON)]
ARCHITECTURE_DECISION_SELF_SHA256 = (
    "7a5ab002828a6ce89940030c21f36a383dc8d0b1c02567f3cc4205495f6b84be"
)
EXPECTED_DEVELOPMENT_RECEIPT_SELF_SHA256 = (
    "6b02c7a9cf783886568abb366cd3f6d3516870662c2f5fdc2dba05f0a78b0ebc"
)
PASS_STATUS = "PASS_UAO_V2_SOURCE_FREE_QUALIFICATION"
STOP_STATUS = "STOP_UAO_V2_COMPILER_BEFORE_REALITY_SOURCE_SELECTION"
NONFORMAL_TEST_PASS_STATUS = "PASS_UAO_V2_NONFORMAL_CONTROLLER_TEST_ONLY"
NONFORMAL_TEST_STOP_STATUS = "STOP_UAO_V2_NONFORMAL_CONTROLLER_TEST_ONLY"
WORKER_TIMEOUT_SECONDS = 3600
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_IMPLEMENTATION_BYTES = 16 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

QUALIFIER_RELATIVE = "scripts/qualify_meta_assumption_synthetic_v1.py"
CONTROLLER_RELATIVE = (
    "scripts/run_meta_assumption_source_free_qualification_v2.py"
)
CONTROLLER_TEST_RELATIVE = (
    "tests/test_run_meta_assumption_source_free_qualification_v2.py"
)
SERVICE_RELATIVE = (
    "manifests/meta-assumption-source-free-qualification-v2.service"
)
CANARY_RELATIVE = "scripts/qualify_meta_assumption_sandbox_v2.py"
CANARY_SERVICE_RELATIVE = (
    "manifests/meta-assumption-source-free-sandbox-canary-v2.service"
)
ARCHITECTURE_RELATIVE = (
    "manifests/"
    "red_queen_universal_assumption_compiler_architecture_decision_v2.json"
)
REQUIRED_RELATIVE_FILES = (
    "assumption_agent/__init__.py",
    "assumption_agent/models.py",
    "assumption_agent/events.py",
    "assumption_agent/runtime.py",
    "assumption_agent/evaluation.py",
    "assumption_agent/meta_assumption.py",
    "assumption_agent/universal_assumption_ontology_v1.py",
    "assumption_agent/benchmarks/__init__.py",
    "assumption_agent/benchmarks/meta_assumption_synthetic_worlds_v1.py",
    QUALIFIER_RELATIVE,
    "tests/test_meta_assumption.py",
    "tests/test_universal_assumption_ontology_v1.py",
    "tests/test_meta_assumption_synthetic_worlds_v1.py",
    CONTROLLER_RELATIVE,
    CONTROLLER_TEST_RELATIVE,
    SERVICE_RELATIVE,
    CANARY_RELATIVE,
    CANARY_SERVICE_RELATIVE,
    ARCHITECTURE_RELATIVE,
)
QUALIFIER_IMPORT_CLOSURE_RELATIVE_FILES = (
    "assumption_agent/__init__.py",
    "assumption_agent/models.py",
    "assumption_agent/events.py",
    "assumption_agent/runtime.py",
    "assumption_agent/evaluation.py",
    "assumption_agent/meta_assumption.py",
    "assumption_agent/universal_assumption_ontology_v1.py",
    "assumption_agent/benchmarks/__init__.py",
    "assumption_agent/benchmarks/meta_assumption_synthetic_worlds_v1.py",
    QUALIFIER_RELATIVE,
)
BLOCKED_QUALIFIER_IMPORT_ROOTS = frozenset(
    {
        "anthropic",
        "aiohttp",
        "asyncio",
        "boto3",
        "builtins",
        "concurrent",
        "ctypes",
        "datasets",
        "ftplib",
        "google",
        "http",
        "huggingface_hub",
        "importlib",
        "multiprocessing",
        "openai",
        "os",
        "pickle",
        "requests",
        "shutil",
        "socket",
        "sqlite3",
        "ssl",
        "subprocess",
        "tempfile",
        "torch",
        "transformers",
        "urllib",
    }
)
BLOCKED_QUALIFIER_CALL_NAMES = frozenset(
    {"__import__", "compile", "eval", "exec", "input", "open"}
)
BLOCKED_QUALIFIER_FILE_METHODS = frozenset(
    {
        "open",
        "read_bytes",
        "read_text",
        "write_bytes",
        "write_text",
    }
)
ALLOWED_QUALIFIER_FILE_METHOD_SITES = frozenset(
    {
        (
            "assumption_agent/events.py",
            "open",
            61,
        ),
    }
)
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
LANDLOCK_MINIMUM_ABI = 3
LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
LANDLOCK_ACCESS_FS_REFER = 1 << 13
LANDLOCK_ACCESS_FS_TRUNCATE = 1 << 14
LANDLOCK_HANDLED_ACCESS_FS = (1 << 15) - 1
LANDLOCK_READ_EXECUTE_ACCESS = (
    LANDLOCK_ACCESS_FS_EXECUTE
    | LANDLOCK_ACCESS_FS_READ_FILE
    | LANDLOCK_ACCESS_FS_READ_DIR
)
LANDLOCK_WORK_ACCESS = (
    LANDLOCK_ACCESS_FS_WRITE_FILE
    | LANDLOCK_ACCESS_FS_READ_FILE
    | LANDLOCK_ACCESS_FS_READ_DIR
    | LANDLOCK_ACCESS_FS_MAKE_DIR
    | LANDLOCK_ACCESS_FS_MAKE_REG
)
PR_SET_NO_NEW_PRIVS = 38
SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446
OUTPUT_FILENAMES = (
    "attempt.json",
    "worker_1.receipt.json",
    "worker_2.receipt.json",
    "result.safe.json",
    "formal_terminal.json",
)
COMMON_OFFLINE_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
FREEZE_KEYS = {
    "architecture_decision_self_sha256",
    "expected_development_receipt_self_sha256",
    "formal_attempt_limit",
    "formal_root",
    "formal_source_access_count_before_qualification",
    "implementation_commit",
    "network_call_count_before_qualification",
    "online_or_API_evaluation_count_before_qualification",
    "ontology_hash",
    "project_root",
    "python_executable",
    "python_executable_mode",
    "python_executable_sha256",
    "python_executable_size_bytes",
    "python_version",
    "qualification_worker_count",
    "required_file_sha256s",
    "retry_replay_resample_or_repair_count",
    "schema",
    "self_sha256",
    "source_payload_access_count_before_qualification",
    "study_id",
    "version",
    "work_root",
    "worker_launch_policy",
    "worker_timeout_seconds",
}
DEVELOPMENT_RECEIPT_KEYS = {
    "all_compilation_receipts_valid",
    "all_known_mechanisms_identified",
    "all_probe_evidence_commitment",
    "all_probe_receipts_trusted_recomputed",
    "all_tampers_rejected",
    "all_wrong_claims_counterevidenced",
    "all_wrong_operators_harmful",
    "api_call_count",
    "claim_order_fixed_perturbation",
    "compiler_hash",
    "compiler_id",
    "compiler_trust_anchor_hash",
    "correct_identification_count",
    "efficacy_evidence",
    "expected_label_invariant",
    "fixture_provenance",
    "formal_result",
    "formal_source_access_count",
    "integer_ratio_decision_contract",
    "mechanism_families",
    "metamorphic_trials",
    "minimum_commitment_stage_order",
    "minimum_commitment_two_stage",
    "model_asset_access_count",
    "network_call_count",
    "no_op_disposition",
    "numeric_payload_contract",
    "numeric_payload_shape",
    "online_evaluator_call_count",
    "ontology_hash",
    "oracle_utility_contract",
    "prediction_signatures_distinct",
    "probe_rule_order_invariant",
    "probe_evidence_bundle_count",
    "probe_matrix_rows",
    "probe_statistic_commitments",
    "probe_verifier_trust_anchors",
    "runtime_active_differential_count",
    "runtime_active_trial_count",
    "runtime_noop_semantic_equivalence_count",
    "runtime_noop_trial_count",
    "safe_recomputed_counts",
    "schema",
    "selection_policy_hash",
    "selector_input_contract",
    "self_sha256",
    "source_payload_access_count",
    "status",
    "structural_variant_commitments",
    "structural_variant_nonisomorphism_verified",
    "structural_variants",
    "tamper_case_count",
    "tamper_case_ids",
    "tamper_rejected_count",
    "tamper_rejections",
    "test_access_count",
    "validation_access_count",
    "version",
    "world_compilations",
    "world_count",
    "world_id_invariant",
    "wrong_claim_count",
    "wrong_claims_with_counterevidence_count",
    "wrong_operator_harm_count",
    "wrong_operator_harm_world_count",
    "wrong_operator_trial_count",
}
MECHANISM_FAMILIES = (
    "sparse",
    "set_interaction",
    "local",
    "contamination",
    "no_op",
)
STRUCTURAL_VARIANTS = ("a", "b")
TAMPER_CASE_IDS = (
    "receipt_id",
    "ontology",
    "template",
    "claim",
    "probe",
    "probe_trust_anchor",
    "probe_evidence_bundle",
    "probe_statistic_commitment",
    "compiler_id",
    "compiler_version",
    "compiler_hash",
    "trust_anchor_hash",
    "primary_metric",
    "compiler_target",
    "treatment_disposition",
    "recipe",
    "recipe_action_binding",
    "behavior",
    "no_op",
)
EXPECTED_OPERATOR_BY_TEMPLATE = {
    "uao.v1.t02_sparsity": "SPARSE_SUPPORT_SELECT",
    "uao.v1.t05_low_order_interaction": "SET_INTERACTION_SCORE",
    "uao.v1.t08_locality_markov_blanket": "LOCAL_NEIGHBORHOOD",
    "uao.v1.t18_sparse_contamination": "ROBUST_TRIM_OR_MIXTURE",
    "uao.v1.t19_minimum_commitment": "PRESERVE_BASELINE",
}


class QualificationControllerError(RuntimeError):
    """The frozen one-shot qualification contract drifted."""


class OneShotRefusal(QualificationControllerError):
    """The formal root is not pristine and cannot be consumed again."""


class WorkerFailure(QualificationControllerError):
    """A frozen qualification worker failed closed."""

    def __init__(
        self,
        message: str,
        *,
        worker_exit_code: int | None = None,
        worker_stderr_sha256: str | None = None,
    ) -> None:
        super().__init__(message)
        issue_id = re.sub(r"[^a-z0-9]+", "_", message.lower()).strip("_")
        self.issue_id = issue_id[:160] or "worker_failure"
        self.worker_exit_code = worker_exit_code
        self.worker_stderr_sha256 = worker_stderr_sha256


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
        raise QualificationControllerError(
            "value is not canonical ASCII JSON"
        ) from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)[:-1]).hexdigest()


def _self_hashed(
    body: Mapping[str, Any], field_name: str
) -> dict[str, Any]:
    output = dict(body)
    output[field_name] = stable_hash(output)
    return output


def _object_without_duplicate_keys(
    pairs: Sequence[tuple[str, object]],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise QualificationControllerError(
                "JSON object contains a duplicate key"
            )
        output[key] = value
    return output


def _decode_json(raw: bytes, *, description: str) -> Mapping[str, Any]:
    if not raw or len(raw) > MAX_JSON_BYTES:
        raise QualificationControllerError(
            f"{description} size is outside the frozen range"
        )
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        QualificationControllerError,
    ) as exc:
        raise QualificationControllerError(
            f"{description} is not strict ASCII JSON"
        ) from exc
    if not isinstance(value, dict):
        raise QualificationControllerError(
            f"{description} root must be an object"
        )
    return value


def _regular_file_bytes(path: Path, *, description: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise QualificationControllerError(
            f"{description} is unavailable"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size <= 0
        or metadata.st_size > MAX_IMPLEMENTATION_BYTES
    ):
        raise QualificationControllerError(
            f"{description} is not a bounded regular file"
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise QualificationControllerError(
            f"{description} cannot be read"
        ) from exc
    if len(raw) != metadata.st_size:
        raise QualificationControllerError(
            f"{description} changed while it was read"
        )
    return raw


def _canonical_absolute_path(value: object, *, field: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or not Path(value).is_absolute()
        or str(Path(value)) != value
        or os.path.realpath(value) != value
    ):
        raise QualificationControllerError(
            f"{field} is not a canonical absolute path"
        )
    return Path(value)


def _validate_architecture_manifest(path: Path) -> None:
    value = _decode_json(
        _regular_file_bytes(
            path, description="UAO architecture decision"
        ),
        description="UAO architecture decision",
    )
    body = dict(value)
    declared = body.pop("self_sha256", None)
    if (
        value.get("schema")
        != "red_queen_universal_assumption_compiler_architecture_decision_v2"
        or value.get("version") != "v2"
        or declared != ARCHITECTURE_DECISION_SELF_SHA256
        or declared != stable_hash(body)
        or value.get("decision")
        != (
            "GO_FRESH_UAO_V2_INFRASTRUCTURE_REBIND_ONLY_AFTER_"
            "ONE_SOURCE_FREE_SANDBOX_CANARY"
        )
    ):
        raise QualificationControllerError(
            "UAO architecture decision binding drifted"
        )


def _audit_exact_frozen_project_tree(
    project_root: Path,
) -> Mapping[str, Any]:
    """Reject every unregistered file, directory, symlink, and import asset."""

    expected_files = set(REQUIRED_RELATIVE_FILES) | {
        f"manifests/{FREEZE_FILENAME}"
    }
    expected_directories = {""}
    for relative in expected_files:
        parent = Path(relative).parent
        while str(parent) != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    observed_files: set[str] = set()
    observed_directories = {""}
    try:
        entries = sorted(
            project_root.rglob("*"),
            key=lambda path: path.relative_to(project_root).as_posix(),
        )
    except OSError as exc:
        raise QualificationControllerError(
            "frozen project tree cannot be enumerated"
        ) from exc
    for path in entries:
        relative = path.relative_to(project_root).as_posix()
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise QualificationControllerError(
                "frozen project tree changed during enumeration"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise QualificationControllerError(
                f"frozen project tree contains a symlink: {relative}"
            )
        if stat.S_ISDIR(metadata.st_mode):
            observed_directories.add(relative)
        elif stat.S_ISREG(metadata.st_mode):
            observed_files.add(relative)
        else:
            raise QualificationControllerError(
                f"frozen project tree contains a special file: {relative}"
            )
    if (
        observed_files != expected_files
        or observed_directories != expected_directories
    ):
        raise QualificationControllerError(
            "frozen project tree exact allowlist drifted"
        )
    body = {
        "schema": "meta_assumption_exact_frozen_project_tree_audit_v1",
        "registered_relative_files": sorted(expected_files),
        "registered_relative_directories": sorted(expected_directories),
        "unregistered_file_count": 0,
        "unregistered_directory_count": 0,
        "symlink_or_special_file_count": 0,
    }
    return {**body, "audit_self_sha256": stable_hash(body)}


def _audit_qualifier_capability_closure(
    project_root: Path,
) -> Mapping[str, Any]:
    """Reject external capability channels in the qualifier import closure."""

    file_hashes: dict[str, str] = {}
    for relative in QUALIFIER_IMPORT_CLOSURE_RELATIVE_FILES:
        raw = _regular_file_bytes(
            project_root / relative,
            description=f"qualifier import closure {relative}",
        )
        try:
            tree = ast.parse(raw.decode("utf-8"), filename=relative)
        except (UnicodeDecodeError, SyntaxError) as exc:
                raise QualificationControllerError(
                    f"qualifier import closure is not parseable: {relative}"
                ) from exc
        for node in ast.walk(tree):
            imported_roots: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                imported_roots = tuple(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots = (node.module.split(".", 1)[0],)
            if any(
                root in BLOCKED_QUALIFIER_IMPORT_ROOTS
                for root in imported_roots
            ):
                raise QualificationControllerError(
                    f"qualifier capability import is forbidden: {relative}"
                )
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and (
                    node.func.id in BLOCKED_QUALIFIER_CALL_NAMES
                ):
                    raise QualificationControllerError(
                        "qualifier dynamic capability call is forbidden: "
                        f"{relative}"
                    )
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in BLOCKED_QUALIFIER_FILE_METHODS
                    and (
                        relative,
                        node.func.attr,
                        node.lineno,
                    )
                    not in ALLOWED_QUALIFIER_FILE_METHOD_SITES
                ):
                    raise QualificationControllerError(
                        f"qualifier file method is forbidden: {relative}"
                    )
        file_hashes[relative] = hashlib.sha256(raw).hexdigest()
    body = {
        "schema": "meta_assumption_qualifier_capability_closure_audit_v1",
        "audited_relative_files": list(
            QUALIFIER_IMPORT_CLOSURE_RELATIVE_FILES
        ),
        "audited_file_sha256s": file_hashes,
        "blocked_import_roots": sorted(BLOCKED_QUALIFIER_IMPORT_ROOTS),
        "blocked_call_names": sorted(BLOCKED_QUALIFIER_CALL_NAMES),
        "blocked_file_methods": sorted(BLOCKED_QUALIFIER_FILE_METHODS),
        "allowed_frozen_file_method_sites": [
            list(row) for row in sorted(ALLOWED_QUALIFIER_FILE_METHOD_SITES)
        ],
        "external_source_network_model_API_or_process_channel_found": False,
    }
    return {**body, "audit_self_sha256": stable_hash(body)}


def _attest_formal_service_sandbox() -> Mapping[str, Any]:
    """Prove the formal controller is inside the frozen network-denied unit."""

    try:
        raw_cgroup = Path("/proc/self/cgroup").read_bytes()
    except OSError as exc:
        raise QualificationControllerError(
            "formal service cgroup cannot be observed"
        ) from exc
    try:
        cgroup_text = raw_cgroup.decode("ascii")
    except UnicodeDecodeError as exc:
        raise QualificationControllerError(
            "formal service cgroup is not ASCII"
        ) from exc
    components = {
        component
        for line in cgroup_text.splitlines()
        for component in line.partition("::")[2].split("/")
        if component
    }
    if FORMAL_SERVICE_UNIT not in components:
        raise QualificationControllerError(
            "controller is not running in the frozen formal service unit"
        )
    denied_errnos: dict[str, int] = {}
    for label, family in (
        ("AF_UNIX", socket.AF_UNIX),
        ("AF_INET", socket.AF_INET),
        ("AF_INET6", socket.AF_INET6),
    ):
        denied_errno: int | None = None
        internet_socket: socket.socket | None = None
        try:
            internet_socket = socket.socket(family, socket.SOCK_STREAM)
        except OSError as exc:
            denied_errno = exc.errno
        finally:
            if internet_socket is not None:
                internet_socket.close()
        if denied_errno not in {
            errno.EACCES,
            errno.EAFNOSUPPORT,
            errno.EPERM,
        }:
            raise QualificationControllerError(
                f"{label} socket creation was not denied by the service sandbox"
            )
        denied_errnos[label] = int(denied_errno)
    return {
        "formal_service_unit": FORMAL_SERVICE_UNIT,
        "formal_service_cgroup_sha256": hashlib.sha256(
            raw_cgroup
        ).hexdigest(),
        "AF_UNIX_socket_creation_denied": True,
        "AF_UNIX_socket_denial_errno": denied_errnos["AF_UNIX"],
        "AF_INET_socket_creation_denied": True,
        "AF_INET_socket_denial_errno": denied_errnos["AF_INET"],
        "AF_INET6_socket_creation_denied": True,
        "AF_INET6_socket_denial_errno": denied_errnos["AF_INET6"],
    }


def _attest_installed_service_binding(
    *,
    project_root: Path,
    installed_service_path: Path = INSTALLED_USER_SERVICE_PATH,
) -> Mapping[str, Any]:
    expected_source = project_root / SERVICE_RELATIVE
    try:
        metadata = installed_service_path.lstat()
    except OSError as exc:
        raise QualificationControllerError(
            "installed formal service binding is unavailable"
        ) from exc
    if not stat.S_ISLNK(metadata.st_mode):
        raise QualificationControllerError(
            "installed formal service binding is not an exact symlink"
        )
    try:
        target = os.readlink(installed_service_path)
    except OSError as exc:
        raise QualificationControllerError(
            "installed formal service target cannot be read"
        ) from exc
    if target != str(expected_source):
        raise QualificationControllerError(
            "installed formal service target drifted"
        )
    raw = _regular_file_bytes(
        expected_source, description="frozen formal service source"
    )
    return {
        "installed_formal_service_path": str(installed_service_path),
        "installed_formal_service_target": target,
        "installed_formal_service_source_sha256": hashlib.sha256(
            raw
        ).hexdigest(),
        "installed_formal_service_binding_attested": True,
    }


class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _LandlockPathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


def _landlock_syscall(
    libc: Any, number: int, *arguments: object
) -> int:
    result = int(libc.syscall(number, *arguments))
    if result < 0:
        error_number = ctypes.get_errno()
        raise QualificationControllerError(
            f"Landlock syscall {number} failed with errno {error_number}"
        )
    return result


def _add_landlock_path_rule(
    *,
    libc: Any,
    ruleset_fd: int,
    path: Path,
    allowed_access: int,
) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_PATH | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise QualificationControllerError(
            f"Landlock allowlisted path is unavailable: {path}"
        ) from exc
    try:
        attribute = _LandlockPathBeneathAttr(
            allowed_access=allowed_access,
            parent_fd=descriptor,
        )
        _landlock_syscall(
            libc,
            SYS_LANDLOCK_ADD_RULE,
            ruleset_fd,
            LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(attribute),
            0,
        )
    finally:
        os.close(descriptor)


def _apply_landlock_filesystem_sandbox(
    *,
    python_executable: Path,
    project_root: Path,
    work_root: Path,
) -> Mapping[str, Any]:
    """Irreversibly restrict this process and descendants to frozen paths."""

    if platform.machine() not in {"x86_64", "AMD64"}:
        raise QualificationControllerError(
            "Landlock syscall registry is not frozen for this architecture"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    abi = _landlock_syscall(
        libc,
        SYS_LANDLOCK_CREATE_RULESET,
        0,
        0,
        LANDLOCK_CREATE_RULESET_VERSION,
    )
    if abi < LANDLOCK_MINIMUM_ABI:
        raise QualificationControllerError(
            "Landlock ABI is below the frozen minimum"
        )
    ruleset_attribute = _LandlockRulesetAttr(
        handled_access_fs=LANDLOCK_HANDLED_ACCESS_FS
    )
    ruleset_fd = _landlock_syscall(
        libc,
        SYS_LANDLOCK_CREATE_RULESET,
        ctypes.byref(ruleset_attribute),
        ctypes.sizeof(ruleset_attribute),
        0,
    )
    try:
        python_standard_library = Path(
            f"/usr/lib/{python_executable.name}"
        )
        read_execute_directories = (
            python_standard_library,
            Path("/usr/lib/x86_64-linux-gnu"),
            Path("/usr/lib/locale"),
            project_root,
        )
        _add_landlock_path_rule(
            libc=libc,
            ruleset_fd=ruleset_fd,
            path=python_executable,
            allowed_access=(
                LANDLOCK_ACCESS_FS_EXECUTE
                | LANDLOCK_ACCESS_FS_READ_FILE
            ),
        )
        for path in read_execute_directories:
            _add_landlock_path_rule(
                libc=libc,
                ruleset_fd=ruleset_fd,
                path=path,
                allowed_access=LANDLOCK_READ_EXECUTE_ACCESS,
            )
        _add_landlock_path_rule(
            libc=libc,
            ruleset_fd=ruleset_fd,
            path=work_root,
            allowed_access=LANDLOCK_WORK_ACCESS,
        )
        _add_landlock_path_rule(
            libc=libc,
            ruleset_fd=ruleset_fd,
            path=Path("/dev/null"),
            allowed_access=(
                LANDLOCK_ACCESS_FS_READ_FILE
                | LANDLOCK_ACCESS_FS_WRITE_FILE
            ),
        )
        if int(libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) < 0:
            error_number = ctypes.get_errno()
            raise QualificationControllerError(
                "Landlock no-new-privileges setup failed with errno "
                f"{error_number}"
            )
        _landlock_syscall(
            libc,
            SYS_LANDLOCK_RESTRICT_SELF,
            ruleset_fd,
            0,
        )
    finally:
        os.close(ruleset_fd)

    allowed_probe = project_root / QUALIFIER_RELATIVE
    try:
        descriptor = os.open(
            allowed_probe,
            os.O_RDONLY
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise QualificationControllerError(
            "Landlock denied the frozen project tree"
        ) from exc
    else:
        os.close(descriptor)

    denial_errnos: dict[str, int] = {}
    for label, path, flags in (
        (
            "home_directory",
            Path("/home/erzhu419"),
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        ),
        (
            "outside_direct_file",
            Path("/etc/passwd"),
            os.O_RDONLY | os.O_CLOEXEC,
        ),
    ):
        descriptor = -1
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EPERM}:
                raise QualificationControllerError(
                    f"Landlock denial probe drifted: {label}"
                ) from exc
            denial_errnos[label] = int(exc.errno)
        else:
            raise QualificationControllerError(
                f"Landlock did not deny the frozen probe: {label}"
            )
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    allowlist_body = {
        "read_execute_files": [str(python_executable)],
        "read_execute_roots": [
            str(python_standard_library),
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/locale",
            str(project_root),
        ],
        "read_write_root": str(work_root),
        "read_write_files": ["/dev/null"],
        "handled_access_fs": LANDLOCK_HANDLED_ACCESS_FS,
    }
    return {
        "landlock_filesystem_restriction_attested": True,
        "landlock_abi": abi,
        "landlock_allowlist_self_sha256": stable_hash(allowlist_body),
        "landlock_home_directory_denial_errno": denial_errnos[
            "home_directory"
        ],
        "landlock_outside_direct_file_denial_errno": denial_errnos[
            "outside_direct_file"
        ],
    }


def _outer_environment(formal_root: Path) -> Mapping[str, str]:
    return {
        **COMMON_OFFLINE_ENVIRONMENT,
        "HOME": str(formal_root),
        "TEMP": "/tmp",
        "TMP": "/tmp",
        "TMPDIR": "/tmp",
    }


def _expected_worker_environment(
    *, work_root: Path, ordinal: int
) -> Mapping[str, str]:
    if ordinal not in (1, 2):
        raise QualificationControllerError(
            "worker ordinal is outside the frozen registry"
        )
    worker_root = work_root / "sandbox" / f"worker_{ordinal}"
    return {
        **COMMON_OFFLINE_ENVIRONMENT,
        "HOME": str(worker_root / "home"),
        "TEMP": str(worker_root / "tmp"),
        "TMP": str(worker_root / "tmp"),
        "TMPDIR": str(worker_root / "tmp"),
    }


def _validate_isolated_launch(
    expected_environment: Mapping[str, str],
) -> None:
    if sys.flags.isolated != 1 or sys.dont_write_bytecode is not True:
        raise QualificationControllerError(
            "process was not launched with exact -I -B isolation"
        )
    if dict(os.environ) != dict(expected_environment):
        raise QualificationControllerError(
            "blank offline process environment drifted"
        )


def load_and_validate_freeze(
    freeze_path: Path,
    *,
    expected_formal_root: Path | None,
    expected_python: Path | None,
    enforce_invocation_path: bool,
    expected_environment: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    """Validate the self hash, roots, runtime, policy, and every frozen byte."""

    raw = _regular_file_bytes(
        freeze_path, description="meta-assumption qualification freeze"
    )
    value = _decode_json(
        raw, description="meta-assumption qualification freeze"
    )
    if set(value) != FREEZE_KEYS:
        raise QualificationControllerError("freeze keyset drifted")
    body = dict(value)
    declared = body.pop("self_sha256", None)
    if (
        value.get("schema") != FREEZE_SCHEMA
        or value.get("version") != "v2"
        or value.get("study_id") != STUDY_ID
        or not isinstance(declared, str)
        or _SHA256.fullmatch(declared) is None
        or declared != stable_hash(body)
    ):
        raise QualificationControllerError("freeze self binding drifted")

    formal_root = _canonical_absolute_path(
        value.get("formal_root"), field="formal root"
    )
    project_root = _canonical_absolute_path(
        value.get("project_root"), field="project root"
    )
    work_root = _canonical_absolute_path(
        value.get("work_root"), field="work root"
    )
    python_executable = _canonical_absolute_path(
        value.get("python_executable"), field="Python executable"
    )
    if expected_formal_root is not None and formal_root != expected_formal_root:
        raise QualificationControllerError("formal root drifted")
    if expected_python is not None and python_executable != expected_python:
        raise QualificationControllerError("Python executable drifted")
    if (
        project_root != formal_root / "reconstruction_v2"
        or work_root != formal_root / "work"
        or freeze_path
        != project_root / "manifests" / FREEZE_FILENAME
    ):
        raise QualificationControllerError(
            "freeze absolute root topology drifted"
        )
    if (
        not isinstance(value.get("implementation_commit"), str)
        or re.fullmatch(
            r"[0-9a-f]{40}", str(value.get("implementation_commit"))
        )
        is None
    ):
        raise QualificationControllerError(
            "implementation commit metadata drifted"
        )
    runtime_identity = FROZEN_RUNTIME_IDENTITIES.get(
        str(python_executable)
    )
    if runtime_identity is None:
        raise QualificationControllerError(
            "Python executable is outside the frozen runtime registry"
        )
    for field, expected in runtime_identity.items():
        if value.get(field) != expected:
            raise QualificationControllerError(
                f"freeze runtime identity drifted: {field}"
            )
    executable = _regular_file_bytes(
        python_executable, description="frozen Python executable"
    )
    mode = stat.S_IMODE(python_executable.stat().st_mode)
    if (
        hashlib.sha256(executable).hexdigest()
        != runtime_identity["python_executable_sha256"]
        or len(executable)
        != runtime_identity["python_executable_size_bytes"]
        or f"{mode:04o}"
        != runtime_identity["python_executable_mode"]
        or platform.python_version()
        != runtime_identity["python_version"]
    ):
        raise QualificationControllerError(
            "observed Python runtime identity drifted"
        )
    if (
        value.get("architecture_decision_self_sha256")
        != ARCHITECTURE_DECISION_SELF_SHA256
        or value.get("expected_development_receipt_self_sha256")
        != EXPECTED_DEVELOPMENT_RECEIPT_SELF_SHA256
        or not isinstance(value.get("ontology_hash"), str)
        or _SHA256.fullmatch(str(value.get("ontology_hash"))) is None
        or value.get("formal_attempt_limit") != 1
        or value.get("qualification_worker_count") != 2
        or value.get("worker_launch_policy")
        != "same_frozen_python_sequential_distinct_processes"
        or value.get("worker_timeout_seconds") != WORKER_TIMEOUT_SECONDS
        or value.get("retry_replay_resample_or_repair_count") != 0
        or value.get("formal_source_access_count_before_qualification") != 0
        or value.get("source_payload_access_count_before_qualification") != 0
        or value.get("network_call_count_before_qualification") != 0
        or value.get(
            "online_or_API_evaluation_count_before_qualification"
        )
        != 0
    ):
        raise QualificationControllerError(
            "freeze source-free qualification policy drifted"
        )

    hashes = value.get("required_file_sha256s")
    if not isinstance(hashes, dict) or set(hashes) != set(
        REQUIRED_RELATIVE_FILES
    ):
        raise QualificationControllerError(
            "freeze implementation registry drifted"
        )
    for relative in REQUIRED_RELATIVE_FILES:
        expected = hashes.get(relative)
        if (
            not isinstance(expected, str)
            or _SHA256.fullmatch(expected) is None
        ):
            raise QualificationControllerError(
                "freeze contains an invalid implementation hash"
            )
        observed = hashlib.sha256(
            _regular_file_bytes(
                project_root / relative, description=relative
            )
        ).hexdigest()
        if observed != expected:
            raise QualificationControllerError(
                f"frozen implementation hash drifted: {relative}"
            )
    _validate_architecture_manifest(
        project_root / ARCHITECTURE_RELATIVE
    )
    if enforce_invocation_path:
        invoked = Path(__file__)
        if not invoked.is_absolute():
            invoked = invoked.resolve()
        if invoked != project_root / CONTROLLER_RELATIVE:
            raise QualificationControllerError(
                "controller invocation path drifted"
            )
        if Path(sys.executable) != python_executable:
            raise QualificationControllerError(
                "controller did not use the freeze-bound Python executable"
            )
        if expected_environment is None:
            raise QualificationControllerError(
                "expected isolated environment was not supplied"
            )
        _validate_isolated_launch(expected_environment)
    return {
        **value,
        "_freeze_file_sha256": hashlib.sha256(raw).hexdigest(),
        "_formal_root_path": formal_root,
        "_project_root_path": project_root,
        "_work_root_path": work_root,
        "_python_path": python_executable,
    }


def _exclusive_write_bytes(path: Path, raw: bytes) -> str:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise QualificationControllerError(
            "formal artifact type or mode drifted"
        )
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(raw).hexdigest()


def _exclusive_write_json(
    path: Path, value: Mapping[str, Any]
) -> str:
    return _exclusive_write_bytes(path, _canonical_bytes(value))


def _assert_pristine_work_root(work_root: Path) -> None:
    if work_root.is_symlink():
        raise OneShotRefusal("formal work root is a symlink")
    if work_root.exists() and (
        not work_root.is_dir() or any(work_root.iterdir())
    ):
        raise OneShotRefusal("formal work root is not pristine")


def _prepare_pristine_work_root(work_root: Path) -> None:
    _assert_pristine_work_root(work_root)
    if not work_root.exists():
        work_root.mkdir(mode=0o700)
    os.chmod(work_root, 0o700)
    if any((work_root / name).exists() for name in OUTPUT_FILENAMES):
        raise OneShotRefusal("a formal artifact already exists")


def validate_semantic_receipt(
    value: Mapping[str, Any],
    *,
    expected_ontology_hash: str,
    expected_self_sha256: str,
) -> Mapping[str, Any]:
    """Validate the exact frozen development qualification declaration."""

    if set(value) != DEVELOPMENT_RECEIPT_KEYS:
        raise WorkerFailure("worker receipt keyset drifted")
    body = dict(value)
    declared = body.pop("self_sha256", None)
    zero_fields = (
        "formal_source_access_count",
        "source_payload_access_count",
        "network_call_count",
        "model_asset_access_count",
        "api_call_count",
        "online_evaluator_call_count",
        "validation_access_count",
        "test_access_count",
    )
    expected_payload_shape = {
        "action_fold_utilities": [4, 6],
        "adjacency": [6, 6],
        "decision_payoffs": [4, 4],
        "node_effect_folds": [4, 6],
        "observation_folds": [4, 8],
        "subset_utility_folds": [2, 64],
    }
    if (
        value.get("schema")
        != "meta_assumption_synthetic_worlds_v1_development_receipt"
        or value.get("version") != "meta_assumption_synthetic_worlds_v1"
        or value.get("status")
        != "passed_nonformal_source_free_development_qualification"
        or value.get("formal_result") is not False
        or value.get("efficacy_evidence") is not False
        or value.get("fixture_provenance")
        != "hand_authored_fixed_schema_numeric_mechanisms_only"
        or value.get("ontology_hash") != expected_ontology_hash
        or value.get("mechanism_families") != list(MECHANISM_FAMILIES)
        or value.get("structural_variants")
        != list(STRUCTURAL_VARIANTS)
        or value.get("world_count") != 10
        or value.get("correct_identification_count") != 10
        or value.get("all_known_mechanisms_identified") is not True
        or value.get("prediction_signatures_distinct") is not True
        or value.get("wrong_claim_count") != 40
        or value.get("wrong_claims_with_counterevidence_count") != 40
        or value.get("all_wrong_claims_counterevidenced") is not True
        or value.get("claim_order_fixed_perturbation") is not True
        or value.get("probe_rule_order_invariant") is not True
        or value.get("world_id_invariant") is not True
        or value.get("expected_label_invariant") is not True
        or value.get("selector_input_contract")
        != "opaque_world_id_and_fixed_schema_numeric_payload_only"
        or value.get("numeric_payload_contract")
        != "same_shape_integer_panels_without_family_or_expected_label"
        or value.get("numeric_payload_shape") != expected_payload_shape
        or value.get("integer_ratio_decision_contract")
        != "all_threshold_decisions_use_committed_cross_product_margins"
        or value.get("oracle_utility_contract")
        != "recipe_typed_numeric_panel_without_expected_family_lookup"
        or value.get("minimum_commitment_two_stage") is not True
        or value.get("minimum_commitment_stage_order")
        != "four_active_probes_then_context_bound_noop_probe"
        or value.get("compiler_id")
        != "synthetic_closed_treatment_compiler_v1"
        or value.get("all_compilation_receipts_valid") is not True
        or value.get("all_probe_receipts_trusted_recomputed") is not True
        or value.get("probe_evidence_bundle_count") != 50
        or value.get("no_op_disposition")
        != "preserve_baseline_program_none"
        or value.get("runtime_active_trial_count") != 8
        or value.get("runtime_active_differential_count") != 8
        or value.get("runtime_noop_trial_count") != 2
        or value.get("runtime_noop_semantic_equivalence_count") != 2
        or value.get("wrong_operator_trial_count") != 32
        or value.get("wrong_operator_harm_count") != 32
        or value.get("wrong_operator_harm_world_count") != 10
        or value.get("all_wrong_operators_harmful") is not True
        or value.get("structural_variant_nonisomorphism_verified")
        is not True
        or value.get("tamper_case_count") != len(TAMPER_CASE_IDS)
        or value.get("tamper_rejected_count") != len(TAMPER_CASE_IDS)
        or value.get("tamper_case_ids") != list(TAMPER_CASE_IDS)
        or value.get("all_tampers_rejected") is not True
        or any(value.get(field) != 0 for field in zero_fields)
        or any(
            not isinstance(value.get(field), str)
            or _SHA256.fullmatch(str(value.get(field))) is None
            for field in (
                "all_probe_evidence_commitment",
                "compiler_hash",
                "compiler_trust_anchor_hash",
                "selection_policy_hash",
            )
        )
        or declared != expected_self_sha256
        or declared != EXPECTED_DEVELOPMENT_RECEIPT_SELF_SHA256
        or declared != stable_hash(body)
    ):
        raise WorkerFailure("worker semantic receipt binding drifted")

    metamorphic = value.get("metamorphic_trials")
    metamorphic_ids = (
        "claim_order",
        "probe_rule_order",
        "world_id",
        "expected_label",
    )
    if (
        not isinstance(metamorphic, dict)
        or set(metamorphic) != set(metamorphic_ids)
    ):
        raise WorkerFailure("worker metamorphic trial registry drifted")
    for trial_id in metamorphic_ids:
        trial = metamorphic.get(trial_id)
        if (
            not isinstance(trial, dict)
            or set(trial)
            != {
                "all_invariant",
                "content_commitment",
                "invariant_count",
                "trial_count",
            }
            or trial.get("all_invariant") is not True
            or trial.get("trial_count") != 10
            or trial.get("invariant_count") != 10
            or not isinstance(trial.get("content_commitment"), str)
            or _SHA256.fullmatch(str(trial.get("content_commitment")))
            is None
        ):
            raise WorkerFailure("worker metamorphic trial binding drifted")

    anchors = value.get("probe_verifier_trust_anchors")
    anchor_keys = {
        "counter_rule_id",
        "implementation_hash",
        "probe_id",
        "support_rule_id",
        "verifier_id",
        "verifier_version",
    }
    if not isinstance(anchors, list) or len(anchors) != 5:
        raise WorkerFailure("worker probe trust-anchor count drifted")
    anchor_hashes: set[str] = set()
    probe_ids: set[str] = set()
    verifier_ids: set[str] = set()
    for anchor in anchors:
        if (
            not isinstance(anchor, dict)
            or set(anchor) != anchor_keys
            or any(
                not isinstance(anchor.get(field), str)
                or not anchor.get(field)
                for field in anchor_keys
            )
            or _SHA256.fullmatch(
                str(anchor.get("implementation_hash"))
            )
            is None
            or anchor["probe_id"] in probe_ids
            or anchor["verifier_id"] in verifier_ids
        ):
            raise WorkerFailure("worker probe trust-anchor binding drifted")
        probe_ids.add(str(anchor["probe_id"]))
        verifier_ids.add(str(anchor["verifier_id"]))
        anchor_hashes.add(stable_hash(anchor))
    if len(anchor_hashes) != 5:
        raise WorkerFailure("worker probe trust anchors are not distinct")

    structures = value.get("structural_variant_commitments")
    if not isinstance(structures, list) or len(structures) != 5:
        raise WorkerFailure("worker structural commitment count drifted")
    structure_hashes: set[str] = set()
    for expected_family, row in zip(MECHANISM_FAMILIES, structures):
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "family",
                "variant_a_structure_hash",
                "variant_b_structure_hash",
            }
            or row.get("family") != expected_family
            or any(
                not isinstance(row.get(field), str)
                or _SHA256.fullmatch(str(row.get(field))) is None
                for field in (
                    "variant_a_structure_hash",
                    "variant_b_structure_hash",
                )
            )
            or row["variant_a_structure_hash"]
            == row["variant_b_structure_hash"]
        ):
            raise WorkerFailure("worker structural commitment drifted")
        structure_hashes.update(
            {
                str(row["variant_a_structure_hash"]),
                str(row["variant_b_structure_hash"]),
            }
        )
    if len(structure_hashes) != 10:
        raise WorkerFailure("worker structural variants are not distinct")

    tamper_rows = value.get("tamper_rejections")
    if (
        not isinstance(tamper_rows, list)
        or len(tamper_rows) != len(TAMPER_CASE_IDS)
    ):
        raise WorkerFailure("worker tamper rejection count drifted")
    for case_id, row in zip(TAMPER_CASE_IDS, tamper_rows):
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "case_id",
                "cause_type",
                "expected_issue_ids",
                "observed_issue_ids",
                "rejected",
            }
            or row.get("case_id") != case_id
            or row.get("cause_type") != "PermissionError"
            or row.get("rejected") is not True
            or not isinstance(row.get("expected_issue_ids"), list)
            or not row["expected_issue_ids"]
            or any(
                not isinstance(issue_id, str) or not issue_id
                for issue_id in row["expected_issue_ids"]
            )
            or row.get("observed_issue_ids") != row["expected_issue_ids"]
        ):
            raise WorkerFailure("worker tamper rejection binding drifted")

    rows = value.get("world_compilations")
    if not isinstance(rows, list) or len(rows) != 10:
        raise WorkerFailure("worker compilation row count drifted")
    observed_pairs: set[tuple[str, str]] = set()
    world_ids: set[str] = set()
    active_count = 0
    no_op_count = 0
    wrong_operator_trial_count = 0
    wrong_operator_harm_count = 0
    wrong_operator_harm_world_count = 0
    world_commitments: dict[str, tuple[object, ...]] = {}
    world_row_keys = {
        "active_runtime_differential",
        "baseline_oracle_utility",
        "baseline_plan_hash",
        "candidate_oracle_utility",
        "candidate_plan_hash",
        "compilation_receipt_hash",
        "compiled_operator",
        "expected_template_id",
        "noop_runtime_equivalent",
        "numeric_payload_hash",
        "observed_counter_signature_ids",
        "observed_support_signature_ids",
        "probe_evidence_bundle_hash",
        "probe_statistic_commitment",
        "probe_trust_anchor_hash",
        "runtime_selected_lane",
        "runtime_semantic_commitment",
        "selected_probe_receipt_hash",
        "selected_template_id",
        "structural_variant",
        "treatment_disposition",
        "world_id",
        "wrong_operator_harm_count",
        "wrong_operator_trial_count",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != world_row_keys:
            raise WorkerFailure("worker compilation row shape drifted")
        template_id = row.get("selected_template_id")
        expected_template_id = row.get("expected_template_id")
        operator = row.get("compiled_operator")
        variant = row.get("structural_variant")
        world_id = row.get("world_id")
        receipt_hash = row.get("compilation_receipt_hash")
        hash_fields = (
            "baseline_plan_hash",
            "candidate_plan_hash",
            "compilation_receipt_hash",
            "numeric_payload_hash",
            "probe_evidence_bundle_hash",
            "probe_statistic_commitment",
            "probe_trust_anchor_hash",
            "runtime_semantic_commitment",
            "selected_probe_receipt_hash",
        )
        if (
            not isinstance(template_id, str)
            or expected_template_id != template_id
            or EXPECTED_OPERATOR_BY_TEMPLATE.get(template_id) != operator
            or variant not in STRUCTURAL_VARIANTS
            or not isinstance(world_id, str)
            or not world_id.startswith("omega.")
            or world_id in world_ids
            or any(
                not isinstance(row.get(field), str)
                or _SHA256.fullmatch(str(row.get(field))) is None
                for field in hash_fields
            )
            or receipt_hash != row.get("compilation_receipt_hash")
            or row.get("probe_trust_anchor_hash") not in anchor_hashes
            or row.get("observed_counter_signature_ids") != []
            or not isinstance(
                row.get("observed_support_signature_ids"), list
            )
            or len(row["observed_support_signature_ids"]) != 1
            or not isinstance(row["observed_support_signature_ids"][0], str)
            or not row["observed_support_signature_ids"][0]
            or type(row.get("baseline_oracle_utility")) is not int
            or type(row.get("candidate_oracle_utility")) is not int
            or type(row.get("wrong_operator_harm_count")) is not int
            or type(row.get("wrong_operator_trial_count")) is not int
        ):
            raise WorkerFailure("worker compilation binding drifted")
        pair = (template_id, str(variant))
        if pair in observed_pairs:
            raise WorkerFailure("worker compilation pair duplicated")
        observed_pairs.add(pair)
        world_ids.add(world_id)
        if operator == "PRESERVE_BASELINE":
            if (
                row.get("treatment_disposition") != "preserve_baseline"
                or row.get("active_runtime_differential") is not False
                or row.get("noop_runtime_equivalent") is not True
                or row.get("runtime_selected_lane")
                != "synthetic.baseline"
                or row.get("candidate_oracle_utility")
                != row.get("baseline_oracle_utility")
                or row.get("candidate_plan_hash")
                != row.get("baseline_plan_hash")
                or row.get("wrong_operator_trial_count") != 4
                or row.get("wrong_operator_harm_count") != 4
            ):
                raise WorkerFailure(
                    "worker no-op runtime semantics drifted"
                )
            no_op_count += 1
        else:
            if (
                row.get("treatment_disposition") != "active_program"
                or row.get("active_runtime_differential") is not True
                or row.get("noop_runtime_equivalent") is not False
                or row.get("runtime_selected_lane")
                != "synthetic.closed_operator"
                or row.get("candidate_oracle_utility")
                <= row.get("baseline_oracle_utility")
                or row.get("candidate_plan_hash")
                == row.get("baseline_plan_hash")
                or row.get("wrong_operator_trial_count") != 3
                or row.get("wrong_operator_harm_count") != 3
            ):
                raise WorkerFailure(
                    "worker active runtime semantics drifted"
                )
            active_count += 1
        wrong_operator_trial_count += row["wrong_operator_trial_count"]
        wrong_operator_harm_count += row["wrong_operator_harm_count"]
        if row["wrong_operator_harm_count"] > 0:
            wrong_operator_harm_world_count += 1
        world_commitments[world_id] = (
            template_id,
            row["selected_probe_receipt_hash"],
            row["probe_evidence_bundle_hash"],
            row["probe_statistic_commitment"],
            row["probe_trust_anchor_hash"],
        )
    expected_pairs = {
        (template_id, variant)
        for template_id in EXPECTED_OPERATOR_BY_TEMPLATE
        for variant in STRUCTURAL_VARIANTS
    }
    if (
        observed_pairs != expected_pairs
        or active_count != 8
        or no_op_count != 2
        or wrong_operator_trial_count != 32
        or wrong_operator_harm_count != 32
        or wrong_operator_harm_world_count != 10
    ):
        raise WorkerFailure(
            "worker did not compile real active and no-op treatments"
        )

    commitments = value.get("probe_statistic_commitments")
    commitment_keys = {
        "probe_evidence_bundle_hash",
        "probe_statistic_commitment",
        "probe_trust_anchor_hash",
        "selected_probe_receipt_hash",
        "selected_template_id",
        "world_id",
    }
    if not isinstance(commitments, list) or len(commitments) != 10:
        raise WorkerFailure("worker probe commitment count drifted")
    commitment_world_ids: set[str] = set()
    for row in commitments:
        if (
            not isinstance(row, dict)
            or set(row) != commitment_keys
            or not isinstance(row.get("world_id"), str)
            or row["world_id"] in commitment_world_ids
            or any(
                not isinstance(row.get(field), str)
                or _SHA256.fullmatch(str(row.get(field))) is None
                for field in (
                    "probe_evidence_bundle_hash",
                    "probe_statistic_commitment",
                    "probe_trust_anchor_hash",
                    "selected_probe_receipt_hash",
                )
            )
            or world_commitments.get(str(row["world_id"]))
            != (
                row.get("selected_template_id"),
                row.get("selected_probe_receipt_hash"),
                row.get("probe_evidence_bundle_hash"),
                row.get("probe_statistic_commitment"),
                row.get("probe_trust_anchor_hash"),
            )
        ):
            raise WorkerFailure("worker probe commitment binding drifted")
        commitment_world_ids.add(str(row["world_id"]))
    if commitment_world_ids != world_ids:
        raise WorkerFailure("worker probe commitment coverage drifted")

    probe_rows = value.get("probe_matrix_rows")
    probe_row_keys = {
        "claim_hash",
        "claim_id",
        "disposition",
        "evidence_bundle_hash",
        "observed_counter_signature_ids",
        "observed_support_signature_ids",
        "probe_receipt_hash",
        "probe_trust_anchor_hash",
        "statistic_commitment_hash",
        "template_id",
        "world_id",
    }
    if not isinstance(probe_rows, list) or len(probe_rows) != 50:
        raise WorkerFailure("worker probe matrix row count drifted")
    observed_probe_pairs: set[tuple[str, str]] = set()
    claim_id_by_template: dict[str, str] = {}
    claim_hash_by_template: dict[str, str] = {}
    anchor_hash_by_template: dict[str, str] = {}
    evidence_hashes: list[str] = []
    receipt_hashes: set[str] = set()
    statistic_hashes: set[str] = set()
    wrong_claim_count = 0
    wrong_claims_with_counterevidence_count = 0
    for row in probe_rows:
        if (
            not isinstance(row, dict)
            or set(row) != probe_row_keys
            or not isinstance(row.get("world_id"), str)
            or row.get("world_id") not in world_ids
            or not isinstance(row.get("template_id"), str)
            or row.get("template_id") not in EXPECTED_OPERATOR_BY_TEMPLATE
            or not isinstance(row.get("claim_id"), str)
            or not row.get("claim_id")
            or any(
                not isinstance(row.get(field), str)
                or _SHA256.fullmatch(str(row.get(field))) is None
                for field in (
                    "claim_hash",
                    "evidence_bundle_hash",
                    "probe_receipt_hash",
                    "probe_trust_anchor_hash",
                    "statistic_commitment_hash",
                )
            )
            or row.get("probe_trust_anchor_hash") not in anchor_hashes
        ):
            raise WorkerFailure("worker probe matrix binding drifted")
        pair = (str(row["world_id"]), str(row["template_id"]))
        if pair in observed_probe_pairs:
            raise WorkerFailure("worker probe matrix pair duplicated")
        observed_probe_pairs.add(pair)
        template_id = str(row["template_id"])
        claim_id = str(row["claim_id"])
        claim_hash = str(row["claim_hash"])
        anchor_hash = str(row["probe_trust_anchor_hash"])
        if (
            template_id in claim_id_by_template
            and claim_id_by_template[template_id] != claim_id
        ) or (
            template_id in claim_hash_by_template
            and claim_hash_by_template[template_id] != claim_hash
        ) or (
            template_id in anchor_hash_by_template
            and anchor_hash_by_template[template_id] != anchor_hash
        ):
            raise WorkerFailure("worker probe template binding drifted")
        claim_id_by_template[template_id] = claim_id
        claim_hash_by_template[template_id] = claim_hash
        anchor_hash_by_template[template_id] = anchor_hash

        expected_template = str(
            world_commitments[str(row["world_id"])][0]
        )
        support_ids = row.get("observed_support_signature_ids")
        counter_ids = row.get("observed_counter_signature_ids")
        if template_id == expected_template:
            if (
                row.get("disposition") != "supported"
                or not isinstance(support_ids, list)
                or len(support_ids) != 1
                or not isinstance(support_ids[0], str)
                or not support_ids[0]
                or counter_ids != []
                or (
                    row.get("probe_receipt_hash"),
                    row.get("evidence_bundle_hash"),
                    row.get("statistic_commitment_hash"),
                    row.get("probe_trust_anchor_hash"),
                )
                != (
                    world_commitments[str(row["world_id"])][1],
                    world_commitments[str(row["world_id"])][2],
                    world_commitments[str(row["world_id"])][3],
                    world_commitments[str(row["world_id"])][4],
                )
            ):
                raise WorkerFailure(
                    "worker selected probe matrix row drifted"
                )
        else:
            wrong_claim_count += 1
            if (
                row.get("disposition") != "falsified"
                or support_ids != []
                or not isinstance(counter_ids, list)
                or len(counter_ids) != 1
                or not isinstance(counter_ids[0], str)
                or not counter_ids[0]
            ):
                raise WorkerFailure(
                    "worker counterevidence probe matrix row drifted"
                )
            wrong_claims_with_counterevidence_count += 1
        evidence_hashes.append(str(row["evidence_bundle_hash"]))
        receipt_hashes.add(str(row["probe_receipt_hash"]))
        statistic_hashes.add(str(row["statistic_commitment_hash"]))

    expected_probe_pairs = {
        (world_id, template_id)
        for world_id in world_ids
        for template_id in EXPECTED_OPERATOR_BY_TEMPLATE
    }
    if (
        observed_probe_pairs != expected_probe_pairs
        or len(claim_id_by_template) != 5
        or len(set(claim_id_by_template.values())) != 5
        or len(claim_hash_by_template) != 5
        or len(set(claim_hash_by_template.values())) != 5
        or len(anchor_hash_by_template) != 5
        or len(set(anchor_hash_by_template.values())) != 5
        or len(set(evidence_hashes)) != 50
        or len(receipt_hashes) != 50
        or len(statistic_hashes) != 50
        or value.get("all_probe_evidence_commitment")
        != stable_hash(sorted(evidence_hashes))
    ):
        raise WorkerFailure("worker probe matrix coverage drifted")

    recomputed_counts = {
        "correct_identification_count": sum(
            row["selected_template_id"] == row["expected_template_id"]
            for row in rows
        ),
        "runtime_active_differential_count": sum(
            row["active_runtime_differential"] is True for row in rows
        ),
        "runtime_active_trial_count": sum(
            row["compiled_operator"] != "PRESERVE_BASELINE"
            for row in rows
        ),
        "runtime_noop_semantic_equivalence_count": sum(
            row["noop_runtime_equivalent"] is True for row in rows
        ),
        "runtime_noop_trial_count": sum(
            row["compiled_operator"] == "PRESERVE_BASELINE"
            for row in rows
        ),
        "tamper_case_count": len(tamper_rows),
        "tamper_rejected_count": sum(
            row["rejected"] is True for row in tamper_rows
        ),
        "world_count": len(rows),
        "wrong_claim_count": wrong_claim_count,
        "wrong_claims_with_counterevidence_count": (
            wrong_claims_with_counterevidence_count
        ),
        "wrong_operator_harm_count": wrong_operator_harm_count,
        "wrong_operator_harm_world_count": wrong_operator_harm_world_count,
        "wrong_operator_trial_count": wrong_operator_trial_count,
    }
    if value.get("safe_recomputed_counts") != recomputed_counts:
        raise WorkerFailure("worker safe recomputed counts drifted")
    for field, recomputed in recomputed_counts.items():
        if value.get(field) != recomputed:
            raise WorkerFailure(
                f"worker top-level recomputed count drifted: {field}"
            )
    return value


def _worker_environment(
    *, work_root: Path, ordinal: int
) -> Mapping[str, str]:
    environment = _expected_worker_environment(
        work_root=work_root, ordinal=ordinal
    )
    sandbox = work_root / "sandbox"
    if sandbox.is_symlink():
        raise WorkerFailure("worker sandbox root is a symlink")
    sandbox.mkdir(mode=0o700, exist_ok=True)
    worker_root = sandbox / f"worker_{ordinal}"
    worker_root.mkdir(mode=0o700, exist_ok=False)
    for name in ("home", "tmp"):
        (worker_root / name).mkdir(mode=0o700, exist_ok=False)
    return environment


def _launch_worker(
    *,
    freeze: Mapping[str, Any],
    environment: Mapping[str, str],
    ordinal: int,
) -> tuple[int, bytes]:
    project_root = freeze["_project_root_path"]
    command = (
        str(freeze["_python_path"]),
        "-I",
        "-B",
        str(project_root / CONTROLLER_RELATIVE),
        "--worker",
        "--worker-ordinal",
        str(ordinal),
        "--freeze",
        str(project_root / "manifests" / FREEZE_FILENAME),
    )
    process = subprocess.Popen(
        command,
        cwd=project_root,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
    )
    try:
        stdout, stderr = process.communicate(
            timeout=WORKER_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired as exc:
        process.kill()
        _stdout_after_kill, stderr_after_kill = process.communicate()
        raise WorkerFailure(
            "qualification worker timed out",
            worker_exit_code=process.returncode,
            worker_stderr_sha256=(
                hashlib.sha256(stderr_after_kill).hexdigest()
                if stderr_after_kill
                else None
            ),
        ) from exc
    stderr_sha256 = hashlib.sha256(stderr).hexdigest() if stderr else None
    if process.returncode != 0:
        raise WorkerFailure(
            "qualification worker exited nonzero",
            worker_exit_code=process.returncode,
            worker_stderr_sha256=stderr_sha256,
        )
    if stderr != b"":
        raise WorkerFailure(
            "qualification worker emitted stderr",
            worker_exit_code=process.returncode,
            worker_stderr_sha256=stderr_sha256,
        )
    if not stdout:
        raise WorkerFailure(
            "qualification worker stdout was empty",
            worker_exit_code=process.returncode,
        )
    if len(stdout) > MAX_JSON_BYTES:
        raise WorkerFailure(
            "qualification worker stdout exceeded limit",
            worker_exit_code=process.returncode,
        )
    value = _decode_json(stdout, description="worker stdout receipt")
    validate_semantic_receipt(
        value,
        expected_ontology_hash=str(freeze["ontology_hash"]),
        expected_self_sha256=str(
            freeze["expected_development_receipt_self_sha256"]
        ),
    )
    if stdout != _canonical_bytes(value):
        raise WorkerFailure("worker stdout is not canonical receipt bytes")
    return process.pid, stdout


WorkerLauncher = Callable[
    [Mapping[str, Any], Mapping[str, str], int], tuple[int, bytes]
]


def _default_launcher(
    freeze: Mapping[str, Any],
    environment: Mapping[str, str],
    ordinal: int,
) -> tuple[int, bytes]:
    return _launch_worker(
        freeze=freeze, environment=environment, ordinal=ordinal
    )


def _safe_failure_result(
    *,
    freeze: Mapping[str, Any],
    attempt: Mapping[str, Any],
    attempt_file_sha256: str,
    worker_file_sha256s: Sequence[str],
    project_tree_audit: Mapping[str, Any],
    capability_audit: Mapping[str, Any],
    sandbox_attestation: Mapping[str, Any],
    filesystem_attestation: Mapping[str, Any],
    failure_stage: str,
    failure_exception_type: str,
    failure_issue_id: str,
    worker_exit_code: int | None,
    worker_stderr_sha256: str | None,
    formal_mode: bool,
) -> Mapping[str, Any]:
    return _self_hashed(
        {
            "schema": (
                f"{VERSION}_safe_result"
                if formal_mode
                else f"{VERSION}_nonformal_test_safe_result"
            ),
            "version": "v2",
            "study_id": STUDY_ID,
            "status": (
                STOP_STATUS
                if formal_mode
                else NONFORMAL_TEST_STOP_STATUS
            ),
            "formal_result": formal_mode,
            "efficacy_evidence": False,
            "qualification_passed": False,
            "nonformal_controller_test_passed": False,
            "failure_code": "frozen_source_free_qualification_failed_closed",
            "failure_stage": failure_stage,
            "failure_exception_type": failure_exception_type,
            "failure_issue_id": failure_issue_id,
            "worker_exit_code": worker_exit_code,
            "worker_stderr_sha256": worker_stderr_sha256,
            "freeze_file_sha256": freeze["_freeze_file_sha256"],
            "freeze_self_sha256": freeze["self_sha256"],
            "attempt_file_sha256": attempt_file_sha256,
            "attempt_self_sha256": attempt["attempt_self_sha256"],
            "completed_worker_receipt_count": len(worker_file_sha256s),
            "worker_receipt_file_sha256s": list(worker_file_sha256s),
            "same_host_two_process_receipt_byte_exact": False,
            "qualifier_capability_closure_audit_self_sha256": (
                capability_audit["audit_self_sha256"]
            ),
            "exact_frozen_project_tree_audit_self_sha256": (
                project_tree_audit["audit_self_sha256"]
            ),
            "unregistered_frozen_project_entry_count": 0,
            "qualifier_import_closure_external_channel_found": False,
            "formal_service_unit_attested": bool(
                sandbox_attestation.get("formal_service_unit_attested")
            ),
            "installed_formal_service_binding_attested": bool(
                sandbox_attestation.get(
                    "installed_formal_service_binding_attested"
                )
            ),
            "installed_formal_service_source_sha256": (
                sandbox_attestation.get(
                    "installed_formal_service_source_sha256"
                )
            ),
            "AF_UNIX_socket_creation_denied": bool(
                sandbox_attestation.get(
                    "AF_UNIX_socket_creation_denied"
                )
            ),
            "AF_INET_socket_creation_denied": bool(
                sandbox_attestation.get(
                    "AF_INET_socket_creation_denied"
                )
            ),
            "AF_INET6_socket_creation_denied": bool(
                sandbox_attestation.get(
                    "AF_INET6_socket_creation_denied"
                )
            ),
            "landlock_filesystem_restriction_attested": bool(
                filesystem_attestation.get(
                    "landlock_filesystem_restriction_attested"
                )
            ),
            "landlock_abi": filesystem_attestation.get("landlock_abi"),
            "landlock_allowlist_self_sha256": (
                filesystem_attestation.get(
                    "landlock_allowlist_self_sha256"
                )
            ),
            "retry_replay_resample_or_repair_count": 0,
            "formal_source_access_count": 0,
            "source_payload_access_count": 0,
            "network_call_count": 0,
            "model_asset_access_count": 0,
            "api_call_count": 0,
            "online_evaluator_call_count": 0,
            "validation_access_count": 0,
            "test_access_count": 0,
            "online_or_API_evaluation_count": 0,
        },
        "result_self_sha256",
    )


def run_controller(
    freeze_path: Path,
    *,
    expected_formal_root: Path,
    expected_python: Path,
    enforce_invocation_path: bool = True,
    launcher: WorkerLauncher = _default_launcher,
) -> Mapping[str, Any]:
    """Consume the only formal attempt and write one durable terminal."""

    freeze = load_and_validate_freeze(
        freeze_path,
        expected_formal_root=expected_formal_root,
        expected_python=expected_python,
        enforce_invocation_path=enforce_invocation_path,
        expected_environment=_outer_environment(expected_formal_root),
    )
    project_tree_audit = _audit_exact_frozen_project_tree(
        freeze["_project_root_path"]
    )
    capability_audit = _audit_qualifier_capability_closure(
        freeze["_project_root_path"]
    )
    if enforce_invocation_path:
        observed_sandbox = _attest_formal_service_sandbox()
        observed_service_binding = _attest_installed_service_binding(
            project_root=freeze["_project_root_path"]
        )
        sandbox_attestation = {
            **observed_sandbox,
            **observed_service_binding,
            "formal_service_unit_attested": True,
        }
    else:
        sandbox_attestation = {
            "formal_service_unit": None,
            "formal_service_cgroup_sha256": None,
            "AF_UNIX_socket_creation_denied": False,
            "AF_UNIX_socket_denial_errno": None,
            "AF_INET_socket_creation_denied": False,
            "AF_INET_socket_denial_errno": None,
            "AF_INET6_socket_creation_denied": False,
            "AF_INET6_socket_denial_errno": None,
            "installed_formal_service_path": None,
            "installed_formal_service_target": None,
            "installed_formal_service_source_sha256": None,
            "installed_formal_service_binding_attested": False,
            "formal_service_unit_attested": False,
        }
    work_root = freeze["_work_root_path"]
    _prepare_pristine_work_root(work_root)
    if enforce_invocation_path:
        filesystem_attestation = _apply_landlock_filesystem_sandbox(
            python_executable=freeze["_python_path"],
            project_root=freeze["_project_root_path"],
            work_root=work_root,
        )
    else:
        filesystem_attestation = {
            "landlock_filesystem_restriction_attested": False,
            "landlock_abi": None,
            "landlock_allowlist_self_sha256": None,
            "landlock_home_directory_denial_errno": None,
            "landlock_outside_direct_file_denial_errno": None,
        }
    attempt = _self_hashed(
        {
            "schema": f"{VERSION}_attempt",
            "version": "v2",
            "study_id": STUDY_ID,
            "status": "formal_attempt_consumed_once",
            "attempt_ordinal": 1,
            "controller_pid": os.getpid(),
            "freeze_path": str(freeze_path),
            "freeze_file_sha256": freeze["_freeze_file_sha256"],
            "freeze_self_sha256": freeze["self_sha256"],
            "architecture_decision_self_sha256": (
                ARCHITECTURE_DECISION_SELF_SHA256
            ),
            "qualification_worker_count": 2,
            "worker_launch_policy": (
                "same_frozen_python_sequential_distinct_processes"
            ),
            "qualifier_capability_closure_audit_self_sha256": (
                capability_audit["audit_self_sha256"]
            ),
            "exact_frozen_project_tree_audit_self_sha256": (
                project_tree_audit["audit_self_sha256"]
            ),
            "unregistered_frozen_project_entry_count": 0,
            "qualifier_import_closure_external_channel_found": False,
            "formal_service_unit_attested": sandbox_attestation[
                "formal_service_unit_attested"
            ],
            "formal_service_cgroup_sha256": sandbox_attestation[
                "formal_service_cgroup_sha256"
            ],
            "installed_formal_service_binding_attested": (
                sandbox_attestation[
                    "installed_formal_service_binding_attested"
                ]
            ),
            "installed_formal_service_source_sha256": (
                sandbox_attestation[
                    "installed_formal_service_source_sha256"
                ]
            ),
            "AF_UNIX_socket_creation_denied": sandbox_attestation[
                "AF_UNIX_socket_creation_denied"
            ],
            "AF_UNIX_socket_denial_errno": sandbox_attestation[
                "AF_UNIX_socket_denial_errno"
            ],
            "AF_INET_socket_creation_denied": sandbox_attestation[
                "AF_INET_socket_creation_denied"
            ],
            "AF_INET_socket_denial_errno": sandbox_attestation[
                "AF_INET_socket_denial_errno"
            ],
            "AF_INET6_socket_creation_denied": sandbox_attestation[
                "AF_INET6_socket_creation_denied"
            ],
            "AF_INET6_socket_denial_errno": sandbox_attestation[
                "AF_INET6_socket_denial_errno"
            ],
            "landlock_filesystem_restriction_attested": (
                filesystem_attestation[
                    "landlock_filesystem_restriction_attested"
                ]
            ),
            "landlock_abi": filesystem_attestation["landlock_abi"],
            "landlock_allowlist_self_sha256": filesystem_attestation[
                "landlock_allowlist_self_sha256"
            ],
            "landlock_home_directory_denial_errno": (
                filesystem_attestation[
                    "landlock_home_directory_denial_errno"
                ]
            ),
            "landlock_outside_direct_file_denial_errno": (
                filesystem_attestation[
                    "landlock_outside_direct_file_denial_errno"
                ]
            ),
            "retry_replay_resample_or_repair_count": 0,
            "formal_source_access_count": 0,
            "source_payload_access_count": 0,
            "network_call_count": 0,
            "model_asset_access_count": 0,
            "api_call_count": 0,
            "online_evaluator_call_count": 0,
            "validation_access_count": 0,
            "test_access_count": 0,
            "online_or_API_evaluation_count": 0,
        },
        "attempt_self_sha256",
    )
    attempt_file_sha256 = _exclusive_write_json(
        work_root / "attempt.json", attempt
    )

    worker_file_sha256s: list[str] = []
    worker_pids: list[int] = []
    worker_receipts: list[bytes] = []
    failure_stage = "worker_loop_not_started"
    try:
        for ordinal in (1, 2):
            failure_stage = f"worker_{ordinal}_environment"
            environment = _worker_environment(
                work_root=work_root, ordinal=ordinal
            )
            failure_stage = f"worker_{ordinal}_launch_and_validation"
            pid, raw = launcher(freeze, environment, ordinal)
            if type(pid) is not int or pid <= 0 or pid in worker_pids:
                raise WorkerFailure(
                    "qualification workers were not distinct processes"
                )
            value = _decode_json(
                raw, description=f"worker {ordinal} receipt"
            )
            validate_semantic_receipt(
                value,
                expected_ontology_hash=str(freeze["ontology_hash"]),
                expected_self_sha256=str(
                    freeze[
                        "expected_development_receipt_self_sha256"
                    ]
                ),
            )
            if raw != _canonical_bytes(value):
                raise WorkerFailure(
                    "worker receipt is not canonical bytes"
                )
            worker_pids.append(pid)
            worker_receipts.append(raw)
            worker_file_sha256s.append(
                _exclusive_write_bytes(
                    work_root / f"worker_{ordinal}.receipt.json",
                    raw,
                )
            )
        failure_stage = "cross_worker_byte_identity"
        if worker_receipts[0] != worker_receipts[1]:
            raise WorkerFailure(
                "two worker receipts are not byte identical"
            )
        semantic = _decode_json(
            worker_receipts[0], description="common worker receipt"
        )
        result = _self_hashed(
            {
                "schema": (
                    f"{VERSION}_safe_result"
                    if enforce_invocation_path
                    else f"{VERSION}_nonformal_test_safe_result"
                ),
                "version": "v2",
                "study_id": STUDY_ID,
                "status": (
                    PASS_STATUS
                    if enforce_invocation_path
                    else NONFORMAL_TEST_PASS_STATUS
                ),
                "formal_result": enforce_invocation_path,
                "efficacy_evidence": False,
                "qualification_passed": enforce_invocation_path,
                "nonformal_controller_test_passed": (
                    not enforce_invocation_path
                ),
                "freeze_file_sha256": freeze["_freeze_file_sha256"],
                "freeze_self_sha256": freeze["self_sha256"],
                "attempt_file_sha256": attempt_file_sha256,
                "attempt_self_sha256": attempt["attempt_self_sha256"],
                "worker_process_count": 2,
                "worker_pids_distinct": True,
                "worker_receipt_file_sha256s": worker_file_sha256s,
                "worker_semantic_receipt_self_sha256": (
                    semantic["self_sha256"]
                ),
                "same_host_two_process_receipt_byte_exact": True,
                "known_mechanism_identification": "10/10",
                "wrong_claim_counterevidence": "40/40",
                "active_program_compile_count": 8,
                "preserve_baseline_compile_count": 2,
                "tamper_rejection": "19/19",
                "probe_evidence_bundle_count": 50,
                "probe_receipts_trusted_recomputed": True,
                "minimum_commitment_two_stage": True,
                "active_runtime_differential": "8/8",
                "noop_runtime_equivalence": "2/2",
                "wrong_operator_harm": "32/32_across_10/10_worlds",
                "claim_order_invariance": "10/10",
                "probe_rule_order_invariance": "10/10",
                "world_id_invariance": "10/10",
                "expected_label_invariance": "10/10",
                "safe_recomputed_counts": semantic[
                    "safe_recomputed_counts"
                ],
                "all_probe_evidence_commitment": semantic[
                    "all_probe_evidence_commitment"
                ],
                "oracle_utility_contract": (
                    "recipe_typed_numeric_panel_without_expected_family_lookup"
                ),
                "qualifier_capability_closure_audit_self_sha256": (
                    capability_audit["audit_self_sha256"]
                ),
                "exact_frozen_project_tree_audit_self_sha256": (
                    project_tree_audit["audit_self_sha256"]
                ),
                "unregistered_frozen_project_entry_count": 0,
                "qualifier_import_closure_external_channel_found": False,
                "formal_service_unit_attested": sandbox_attestation[
                    "formal_service_unit_attested"
                ],
                "formal_service_cgroup_sha256": sandbox_attestation[
                    "formal_service_cgroup_sha256"
                ],
                "installed_formal_service_binding_attested": (
                    sandbox_attestation[
                        "installed_formal_service_binding_attested"
                    ]
                ),
                "installed_formal_service_source_sha256": (
                    sandbox_attestation[
                        "installed_formal_service_source_sha256"
                    ]
                ),
                "AF_UNIX_socket_creation_denied": sandbox_attestation[
                    "AF_UNIX_socket_creation_denied"
                ],
                "AF_UNIX_socket_denial_errno": sandbox_attestation[
                    "AF_UNIX_socket_denial_errno"
                ],
                "AF_INET_socket_creation_denied": sandbox_attestation[
                    "AF_INET_socket_creation_denied"
                ],
                "AF_INET_socket_denial_errno": sandbox_attestation[
                    "AF_INET_socket_denial_errno"
                ],
                "AF_INET6_socket_creation_denied": sandbox_attestation[
                    "AF_INET6_socket_creation_denied"
                ],
                "AF_INET6_socket_denial_errno": sandbox_attestation[
                    "AF_INET6_socket_denial_errno"
                ],
                "landlock_filesystem_restriction_attested": (
                    filesystem_attestation[
                        "landlock_filesystem_restriction_attested"
                    ]
                ),
                "landlock_abi": filesystem_attestation["landlock_abi"],
                "landlock_allowlist_self_sha256": (
                    filesystem_attestation[
                        "landlock_allowlist_self_sha256"
                    ]
                ),
                "landlock_home_directory_denial_errno": (
                    filesystem_attestation[
                        "landlock_home_directory_denial_errno"
                    ]
                ),
                "landlock_outside_direct_file_denial_errno": (
                    filesystem_attestation[
                        "landlock_outside_direct_file_denial_errno"
                    ]
                ),
                "retry_replay_resample_or_repair_count": 0,
                "formal_source_access_count": 0,
                "source_payload_access_count": 0,
                "network_call_count": 0,
                "model_asset_access_count": 0,
                "api_call_count": 0,
                "online_evaluator_call_count": 0,
                "validation_access_count": 0,
                "test_access_count": 0,
                "online_or_API_evaluation_count": 0,
            },
            "result_self_sha256",
        )
    except Exception as exc:
        failure_issue_id = getattr(
            exc,
            "issue_id",
            (
                "unexpected_"
                + re.sub(
                    r"[^a-z0-9]+",
                    "_",
                    type(exc).__name__.lower(),
                ).strip("_")
            ),
        )
        result = _safe_failure_result(
            freeze=freeze,
            attempt=attempt,
            attempt_file_sha256=attempt_file_sha256,
            worker_file_sha256s=worker_file_sha256s,
            project_tree_audit=project_tree_audit,
            capability_audit=capability_audit,
            sandbox_attestation=sandbox_attestation,
            filesystem_attestation=filesystem_attestation,
            failure_stage=failure_stage,
            failure_exception_type=type(exc).__name__,
            failure_issue_id=str(failure_issue_id),
            worker_exit_code=getattr(exc, "worker_exit_code", None),
            worker_stderr_sha256=getattr(
                exc, "worker_stderr_sha256", None
            ),
            formal_mode=enforce_invocation_path,
        )

    result_file_sha256 = _exclusive_write_json(
        work_root / "result.safe.json", result
    )
    formal_passed = result["status"] == PASS_STATUS
    nonformal_passed = (
        result["status"] == NONFORMAL_TEST_PASS_STATUS
    )
    run_succeeded = formal_passed or nonformal_passed
    terminal = _self_hashed(
        {
            "schema": (
                f"{VERSION}_formal_terminal"
                if enforce_invocation_path
                else f"{VERSION}_nonformal_test_terminal"
            ),
            "version": "v2",
            "study_id": STUDY_ID,
            "status": (
                PASS_STATUS
                if formal_passed
                else (
                    NONFORMAL_TEST_PASS_STATUS
                    if nonformal_passed
                    else (
                        STOP_STATUS
                        if enforce_invocation_path
                        else NONFORMAL_TEST_STOP_STATUS
                    )
                )
            ),
            "formal_complete": enforce_invocation_path,
            "formal_result": enforce_invocation_path,
            "efficacy_evidence": False,
            "qualification_passed": formal_passed,
            "nonformal_controller_test_complete": (
                not enforce_invocation_path
            ),
            "nonformal_controller_test_passed": nonformal_passed,
            "freeze_file_sha256": freeze["_freeze_file_sha256"],
            "freeze_self_sha256": freeze["self_sha256"],
            "attempt_file_sha256": attempt_file_sha256,
            "attempt_self_sha256": attempt["attempt_self_sha256"],
            "result_safe_file_sha256": result_file_sha256,
            "result_safe_self_sha256": result["result_self_sha256"],
            "worker_receipt_file_sha256s": worker_file_sha256s,
            "same_host_two_process_receipt_byte_exact": run_succeeded,
            "qualifier_capability_closure_audit_self_sha256": (
                capability_audit["audit_self_sha256"]
            ),
            "exact_frozen_project_tree_audit_self_sha256": (
                project_tree_audit["audit_self_sha256"]
            ),
            "unregistered_frozen_project_entry_count": 0,
            "qualifier_import_closure_external_channel_found": False,
            "formal_service_unit_attested": sandbox_attestation[
                "formal_service_unit_attested"
            ],
            "formal_service_cgroup_sha256": sandbox_attestation[
                "formal_service_cgroup_sha256"
            ],
            "installed_formal_service_binding_attested": (
                sandbox_attestation[
                    "installed_formal_service_binding_attested"
                ]
            ),
            "installed_formal_service_source_sha256": (
                sandbox_attestation[
                    "installed_formal_service_source_sha256"
                ]
            ),
            "AF_UNIX_socket_creation_denied": sandbox_attestation[
                "AF_UNIX_socket_creation_denied"
            ],
            "AF_UNIX_socket_denial_errno": sandbox_attestation[
                "AF_UNIX_socket_denial_errno"
            ],
            "AF_INET_socket_creation_denied": sandbox_attestation[
                "AF_INET_socket_creation_denied"
            ],
            "AF_INET_socket_denial_errno": sandbox_attestation[
                "AF_INET_socket_denial_errno"
            ],
            "AF_INET6_socket_creation_denied": sandbox_attestation[
                "AF_INET6_socket_creation_denied"
            ],
            "AF_INET6_socket_denial_errno": sandbox_attestation[
                "AF_INET6_socket_denial_errno"
            ],
            "landlock_filesystem_restriction_attested": (
                filesystem_attestation[
                    "landlock_filesystem_restriction_attested"
                ]
            ),
            "landlock_abi": filesystem_attestation["landlock_abi"],
            "landlock_allowlist_self_sha256": filesystem_attestation[
                "landlock_allowlist_self_sha256"
            ],
            "landlock_home_directory_denial_errno": (
                filesystem_attestation[
                    "landlock_home_directory_denial_errno"
                ]
            ),
            "landlock_outside_direct_file_denial_errno": (
                filesystem_attestation[
                    "landlock_outside_direct_file_denial_errno"
                ]
            ),
            "retry_replay_resample_or_repair_count": 0,
            "formal_source_access_count": 0,
            "source_payload_access_count": 0,
            "network_call_count": 0,
            "model_asset_access_count": 0,
            "api_call_count": 0,
            "online_evaluator_call_count": 0,
            "validation_access_count": 0,
            "test_access_count": 0,
            "online_or_API_evaluation_count": 0,
            "next_action": (
                "freeze_at_most_one_fresh_reality_study_separately"
                if formal_passed
                else (
                    "close_UAO_v2_compiler_without_real_source"
                    if enforce_invocation_path
                    else "nonformal_test_complete_no_reality_authorization"
                )
            ),
        },
        "terminal_self_sha256",
    )
    _exclusive_write_json(work_root / "formal_terminal.json", terminal)
    return terminal


def _load_qualifier(project_root: Path) -> Any:
    path = project_root / QUALIFIER_RELATIVE
    specification = importlib.util.spec_from_file_location(
        "_frozen_meta_assumption_source_free_qualifier_v1", path
    )
    if specification is None or specification.loader is None:
        raise WorkerFailure("frozen qualifier cannot be imported")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    qualification = getattr(module, "qualification", None)
    if qualification is None or not callable(
        getattr(qualification, "qualify", None)
    ):
        raise WorkerFailure("frozen qualifier entry point drifted")
    return qualification


def worker_main(
    freeze_path: Path,
    *,
    ordinal: int,
    stdout_buffer: Any | None = None,
) -> int:
    """Emit exactly one canonical qualification receipt and no other output."""

    freeze = load_and_validate_freeze(
        freeze_path,
        expected_formal_root=None,
        expected_python=None,
        enforce_invocation_path=True,
        expected_environment=_expected_worker_environment(
            work_root=freeze_path.parents[2] / "work",
            ordinal=ordinal,
        ),
    )
    qualifier = _load_qualifier(freeze["_project_root_path"])
    receipt = qualifier.qualify()
    if not isinstance(receipt, dict):
        raise WorkerFailure("frozen qualifier did not return an object")
    validate_semantic_receipt(
        receipt,
        expected_ontology_hash=str(freeze["ontology_hash"]),
        expected_self_sha256=str(
            freeze["expected_development_receipt_self_sha256"]
        ),
    )
    raw = _canonical_bytes(receipt)
    output = sys.stdout.buffer if stdout_buffer is None else stdout_buffer
    output.write(raw)
    output.flush()
    return 0


def run_preflight(
    freeze_path: Path,
    *,
    expected_formal_root: Path,
    expected_python: Path,
    enforce_invocation_path: bool = True,
) -> Mapping[str, Any]:
    """Validate all frozen bindings without consuming the formal attempt."""

    freeze = load_and_validate_freeze(
        freeze_path,
        expected_formal_root=expected_formal_root,
        expected_python=expected_python,
        enforce_invocation_path=enforce_invocation_path,
        expected_environment=_outer_environment(expected_formal_root),
    )
    project_tree_audit = _audit_exact_frozen_project_tree(
        freeze["_project_root_path"]
    )
    capability_audit = _audit_qualifier_capability_closure(
        freeze["_project_root_path"]
    )
    if enforce_invocation_path:
        sandbox = {
            **_attest_formal_service_sandbox(),
            **_attest_installed_service_binding(
                project_root=freeze["_project_root_path"]
            ),
        }
        service_attested = True
    else:
        sandbox = {
            "formal_service_cgroup_sha256": None,
            "AF_UNIX_socket_creation_denied": False,
            "AF_UNIX_socket_denial_errno": None,
            "AF_INET_socket_creation_denied": False,
            "AF_INET_socket_denial_errno": None,
            "AF_INET6_socket_creation_denied": False,
            "AF_INET6_socket_denial_errno": None,
            "installed_formal_service_binding_attested": False,
            "installed_formal_service_source_sha256": None,
        }
        service_attested = False
    _assert_pristine_work_root(freeze["_work_root_path"])
    return _self_hashed(
        {
            "schema": f"{VERSION}_preflight_receipt",
            "version": "v2",
            "study_id": STUDY_ID,
            "status": "PASS_UAO_V2_SOURCE_FREE_PREFLIGHT",
            "formal_attempt_created": False,
            "freeze_file_sha256": freeze["_freeze_file_sha256"],
            "freeze_self_sha256": freeze["self_sha256"],
            "qualifier_capability_closure_audit_self_sha256": (
                capability_audit["audit_self_sha256"]
            ),
            "exact_frozen_project_tree_audit_self_sha256": (
                project_tree_audit["audit_self_sha256"]
            ),
            "unregistered_frozen_project_entry_count": 0,
            "qualifier_import_closure_external_channel_found": False,
            "formal_service_unit_attested": service_attested,
            "installed_formal_service_binding_attested": sandbox[
                "installed_formal_service_binding_attested"
            ],
            "installed_formal_service_source_sha256": sandbox[
                "installed_formal_service_source_sha256"
            ],
            "formal_service_cgroup_sha256": sandbox[
                "formal_service_cgroup_sha256"
            ],
            "AF_UNIX_socket_creation_denied": sandbox[
                "AF_UNIX_socket_creation_denied"
            ],
            "AF_UNIX_socket_denial_errno": sandbox[
                "AF_UNIX_socket_denial_errno"
            ],
            "AF_INET_socket_creation_denied": sandbox[
                "AF_INET_socket_creation_denied"
            ],
            "AF_INET_socket_denial_errno": sandbox[
                "AF_INET_socket_denial_errno"
            ],
            "AF_INET6_socket_creation_denied": sandbox[
                "AF_INET6_socket_creation_denied"
            ],
            "AF_INET6_socket_denial_errno": sandbox[
                "AF_INET6_socket_denial_errno"
            ],
            "formal_source_access_count": 0,
            "source_payload_access_count": 0,
            "network_call_count": 0,
            "model_asset_access_count": 0,
            "api_call_count": 0,
            "online_evaluator_call_count": 0,
            "validation_access_count": 0,
            "test_access_count": 0,
            "online_or_API_evaluation_count": 0,
            "retry_replay_resample_or_repair_count": 0,
        },
        "preflight_self_sha256",
    )


def _write_bootstrap_stop(formal_root: Path) -> Mapping[str, Any]:
    """Write one durable STOP if validation fails before attempt creation."""

    work_root = formal_root / "work"
    _prepare_pristine_work_root(work_root)
    result = _self_hashed(
        {
            "schema": f"{VERSION}_bootstrap_safe_result",
            "version": "v2",
            "study_id": STUDY_ID,
            "status": STOP_STATUS,
            "formal_result": True,
            "efficacy_evidence": False,
            "qualification_passed": False,
            "attempt_created": False,
            "failure_code": "bootstrap_validation_failed_closed",
            "retry_replay_resample_or_repair_count": 0,
            "formal_source_access_count": 0,
            "source_payload_access_count": 0,
            "network_call_count": 0,
            "model_asset_access_count": 0,
            "api_call_count": 0,
            "online_evaluator_call_count": 0,
            "validation_access_count": 0,
            "test_access_count": 0,
            "online_or_API_evaluation_count": 0,
            "next_action": "close_UAO_v2_compiler_without_real_source",
        },
        "result_self_sha256",
    )
    result_file_sha256 = _exclusive_write_json(
        work_root / "result.safe.json", result
    )
    terminal = _self_hashed(
        {
            "schema": f"{VERSION}_bootstrap_formal_terminal",
            "version": "v2",
            "study_id": STUDY_ID,
            "status": STOP_STATUS,
            "formal_complete": True,
            "formal_result": True,
            "efficacy_evidence": False,
            "qualification_passed": False,
            "attempt_created": False,
            "result_safe_file_sha256": result_file_sha256,
            "result_safe_self_sha256": result["result_self_sha256"],
            "retry_replay_resample_or_repair_count": 0,
            "formal_source_access_count": 0,
            "source_payload_access_count": 0,
            "network_call_count": 0,
            "online_or_API_evaluation_count": 0,
            "next_action": "close_UAO_v2_compiler_without_real_source",
        },
        "terminal_self_sha256",
    )
    _exclusive_write_json(work_root / "formal_terminal.json", terminal)
    return terminal


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen one-shot UAO v2 source-free qualification"
    )
    parser.add_argument("--freeze", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    parser.add_argument("--worker-ordinal", type=int, choices=(1, 2))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    freeze_path = Path(arguments.freeze)
    try:
        if arguments.worker:
            if arguments.worker_ordinal is None:
                raise QualificationControllerError(
                    "worker ordinal is required"
                )
            return worker_main(
                freeze_path, ordinal=arguments.worker_ordinal
            )
        if arguments.worker_ordinal is not None:
            raise QualificationControllerError(
                "worker ordinal is forbidden outside worker mode"
            )
        expected_freeze = (
            FORMAL_ROOT / "reconstruction_v2" / "manifests" / FREEZE_FILENAME
        )
        if freeze_path != expected_freeze:
            raise QualificationControllerError(
                "formal freeze path drifted"
            )
        if arguments.preflight:
            receipt = run_preflight(
                freeze_path,
                expected_formal_root=FORMAL_ROOT,
                expected_python=FROZEN_PYTHON,
                enforce_invocation_path=True,
            )
            sys.stdout.buffer.write(_canonical_bytes(receipt))
            sys.stdout.buffer.flush()
            return 0
        terminal = run_controller(
            freeze_path,
            expected_formal_root=FORMAL_ROOT,
            expected_python=FROZEN_PYTHON,
            enforce_invocation_path=True,
        )
        return 0 if terminal["status"] == PASS_STATUS else 1
    except Exception as exc:
        if not arguments.worker and not arguments.preflight:
            attempt_path = FORMAL_ROOT / "work" / "attempt.json"
            if not attempt_path.exists():
                try:
                    _write_bootstrap_stop(FORMAL_ROOT)
                except Exception:
                    pass
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
