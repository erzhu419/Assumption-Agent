#!/usr/bin/env python3
"""One-shot formal, source-free qualification controller for RJMC-V1.

The controller consumes one future self-hashed implementation freeze, verifies
every frozen implementation byte, and then starts exactly two *sequential*
worker processes with the same frozen Python interpreter.  Each worker imports
the frozen development qualifier and emits only its canonical semantic JSON
receipt on stdout.  The controller requires the two receipts to be byte
identical before recording a pass.

No QuAC payload path, network endpoint, API credential, retry, replay, repair,
or alternative candidate is accepted by this interface.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


VERSION = "quac_rjmc_source_free_qualification_controller_v1"
STUDY_ID = "QUAC_P1_RJMC_DIALOGUE_EVIDENCE_L5_V1"
FREEZE_SCHEMA = "quac_rjmc_source_free_qualification_freeze_v1"
FREEZE_FILENAME = "quac_rjmc_source_free_qualification_freeze_v1.json"
FORMAL_ROOT = Path(
    "/home/erzhu419/quac_rjmc_20260728/source_free_qualification_v1"
)
FROZEN_PYTHON = Path(
    "/home/erzhu419/p19_runtime_assets_20260723/typed_venv/bin/python"
)
FROZEN_RUNTIME_IDENTITY = {
    "python_executable_sha256": (
        "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
    ),
    "python_executable_size_bytes": 5917224,
    "python_executable_mode": "0755",
    "python_version": "3.10.12",
    "torch_version": "2.8.0+cu128",
    "numpy_version": "2.2.6",
}
ARCHITECTURE_DECISION_SELF_SHA256 = (
    "9efb416359c1efc315846523a67382b0b942a8a827976cece72175085fe79462"
)
SOURCE_CUSTODY_SELF_SHA256 = (
    "d098b6e7a14e0e7d77f6b59869a4e913a210e4d30bf8bb72f97addd89bba3c30"
)
PASS_STATUS = "PASS_RJMC_SOURCE_FREE_QUALIFICATION"
STOP_STATUS = "STOP_RJMC_ARCHITECTURE_BEFORE_QUAC_SOURCE_DOWNLOAD"
WORKER_TIMEOUT_SECONDS = 3600
MAX_JSON_BYTES = 2 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

CORE_RELATIVE = (
    "assumption_agent/benchmarks/quac_rjmc_evaluator_v1.py"
)
QUALIFIER_RELATIVE = "scripts/qualify_quac_rjmc_source_free_v1.py"
CORE_TEST_RELATIVE = "tests/test_quac_rjmc_evaluator_v1.py"
CONTROLLER_RELATIVE = (
    "scripts/run_quac_rjmc_source_free_qualification_v1.py"
)
CONTROLLER_TEST_RELATIVE = (
    "tests/test_run_quac_rjmc_source_free_qualification_v1.py"
)
SERVICE_RELATIVE = (
    "manifests/quac_rjmc_source_free_qualification_v1.service"
)
ARCHITECTURE_RELATIVE = (
    "manifests/red_queen_poststop_rjmc_architecture_decision_v1.json"
)
SOURCE_CUSTODY_RELATIVE = "manifests/quac_p1_source_custody_v1.json"
REQUIRED_RELATIVE_FILES = (
    CORE_RELATIVE,
    QUALIFIER_RELATIVE,
    CORE_TEST_RELATIVE,
    CONTROLLER_RELATIVE,
    CONTROLLER_TEST_RELATIVE,
    SERVICE_RELATIVE,
    ARCHITECTURE_RELATIVE,
    SOURCE_CUSTODY_RELATIVE,
)

FREEZE_KEYS = {
    "architecture_decision_self_sha256",
    "formal_attempt_limit",
    "formal_root",
    "implementation_commit",
    "numpy_version",
    "online_or_API_evaluation_count_before_qualification",
    "project_root",
    "python_executable",
    "python_executable_mode",
    "python_executable_sha256",
    "python_executable_size_bytes",
    "python_version",
    "qualification_worker_count",
    "required_file_sha256s",
    "schema",
    "self_sha256",
    "source_custody_self_sha256",
    "source_payload_access_count_before_qualification",
    "study_id",
    "torch_version",
    "version",
    "work_root",
    "worker_launch_policy",
    "worker_timeout_seconds",
}
RECEIPT_KEYS = {
    "A_hold",
    "M_search",
    "QuAC_source_payload_access_count",
    "RAW_structural_zero",
    "antisymmetric",
    "architecture_decision_self_sha256",
    "behavior_sha256",
    "complete_state_count_for_three_candidates",
    "component_jackknife_head_count",
    "evaluator_version",
    "fixture_provenance",
    "fixture_topologies",
    "formal_result",
    "online_or_API_evaluation_count",
    "parameter_sha256",
    "permutation_invariant",
    "prior_private_source_access_count",
    "qualification_weights_disposition",
    "receipt_self_sha256",
    "same_process_repeat_exact",
    "schema",
    "status",
    "version",
}
BLOCK_REQUIRED_KEYS = {
    "E0_total_utility",
    "E1_minus_E0",
    "E1_total_utility",
    "item_count",
    "topology_delta",
    "topology_raw_harm",
    "topology_required_complete",
}
TOPOLOGIES = (
    "pair_complement",
    "redundancy_trap",
    "retention_trap",
    "null_shift",
)
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


class QualificationControllerError(RuntimeError):
    """The frozen one-shot source-free qualification contract drifted."""


class OneShotRefusal(QualificationControllerError):
    """The requested formal output root is not pristine."""


class WorkerFailure(QualificationControllerError):
    """One frozen qualification worker failed closed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise QualificationControllerError(
            "value is not canonical JSON"
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
        or metadata.st_size > MAX_JSON_BYTES * 8
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


def _validate_canonical_absolute_path(
    value: object, *, field: str, require_realpath_identity: bool = True
) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or not Path(value).is_absolute()
        or str(Path(value)) != value
        or (
            require_realpath_identity
            and os.path.realpath(value) != value
        )
    ):
        raise QualificationControllerError(
            f"{field} is not a canonical absolute path"
        )
    return Path(value)


def _validate_self_hashed_manifest(
    path: Path,
    *,
    schema: str,
    expected_self_sha256: str,
) -> None:
    value = _decode_json(
        _regular_file_bytes(path, description=schema),
        description=schema,
    )
    body = dict(value)
    declared = body.pop("self_sha256", None)
    if (
        value.get("schema") != schema
        or value.get("version") != "v1"
        or declared != expected_self_sha256
        or declared != stable_hash(body)
    ):
        raise QualificationControllerError(
            f"{schema} self binding drifted"
        )


def _outer_environment(formal_root: Path) -> Mapping[str, str]:
    return {
        **COMMON_OFFLINE_ENVIRONMENT,
        "HOME": str(formal_root),
        "TEMP": "/tmp",
        "TMP": "/tmp",
        "TMPDIR": "/tmp",
    }


def _expected_worker_environment(
    *,
    project_root: Path,
    work_root: Path,
    ordinal: int,
) -> Mapping[str, str]:
    if ordinal not in (1, 2):
        raise QualificationControllerError(
            "worker ordinal is outside the frozen registry"
        )
    worker_root = work_root / "sandbox" / f"worker_{ordinal}"
    return {
        **COMMON_OFFLINE_ENVIRONMENT,
        "HOME": str(worker_root / "home"),
        "HF_HOME": str(worker_root / "cache"),
        "PYTHONPATH": str(project_root),
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


def _observe_runtime_versions() -> Mapping[str, str]:
    try:
        import numpy  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
    except Exception as exc:
        raise QualificationControllerError(
            "frozen numeric runtime cannot be imported"
        ) from exc
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(numpy.__version__),
    }


def load_and_validate_freeze(
    freeze_path: Path,
    *,
    expected_formal_root: Path | None,
    expected_python: Path | None,
    enforce_invocation_path: bool,
    expected_runtime_identity: Mapping[str, object] | None = None,
    expected_environment: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    """Load a future freeze and verify every semantic and byte binding."""

    raw = _regular_file_bytes(
        freeze_path, description="source-free qualification freeze"
    )
    value = _decode_json(
        raw, description="source-free qualification freeze"
    )
    if set(value) != FREEZE_KEYS:
        raise QualificationControllerError("freeze keyset drifted")
    body = dict(value)
    declared_self = body.pop("self_sha256", None)
    if (
        value.get("schema") != FREEZE_SCHEMA
        or value.get("version") != "v1"
        or value.get("study_id") != STUDY_ID
        or not isinstance(declared_self, str)
        or _SHA256.fullmatch(declared_self) is None
        or declared_self != stable_hash(body)
    ):
        raise QualificationControllerError("freeze self binding drifted")

    formal_root = _validate_canonical_absolute_path(
        value.get("formal_root"), field="formal root"
    )
    project_root = _validate_canonical_absolute_path(
        value.get("project_root"), field="project root"
    )
    work_root = _validate_canonical_absolute_path(
        value.get("work_root"), field="work root"
    )
    python_executable = _validate_canonical_absolute_path(
        value.get("python_executable"),
        field="Python executable",
        require_realpath_identity=False,
    )
    runtime_identity = dict(
        FROZEN_RUNTIME_IDENTITY
        if expected_runtime_identity is None
        else expected_runtime_identity
    )
    if set(runtime_identity) != set(FROZEN_RUNTIME_IDENTITY):
        raise QualificationControllerError(
            "expected runtime identity registry drifted"
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
            "freeze implementation commit metadata drifted"
        )
    for field, expected_value in runtime_identity.items():
        if value.get(field) != expected_value:
            raise QualificationControllerError(
                f"freeze runtime identity drifted: {field}"
            )
    executable_raw = _regular_file_bytes(
        python_executable, description="frozen Python executable"
    )
    executable_mode = stat.S_IMODE(python_executable.stat().st_mode)
    if (
        hashlib.sha256(executable_raw).hexdigest()
        != runtime_identity["python_executable_sha256"]
        or len(executable_raw)
        != runtime_identity["python_executable_size_bytes"]
        or f"{executable_mode:04o}"
        != runtime_identity["python_executable_mode"]
    ):
        raise QualificationControllerError(
            "observed Python executable identity drifted"
        )
    observed_versions = _observe_runtime_versions()
    if any(
        observed_versions[field] != runtime_identity[field]
        for field in (
            "python_version",
            "torch_version",
            "numpy_version",
        )
    ):
        raise QualificationControllerError(
            "observed numeric runtime version drifted"
        )
    if (
        value.get("architecture_decision_self_sha256")
        != ARCHITECTURE_DECISION_SELF_SHA256
        or value.get("source_custody_self_sha256")
        != SOURCE_CUSTODY_SELF_SHA256
        or value.get("source_payload_access_count_before_qualification") != 0
        or value.get(
            "online_or_API_evaluation_count_before_qualification"
        )
        != 0
        or value.get("formal_attempt_limit") != 1
        or value.get("qualification_worker_count") != 2
        or value.get("worker_launch_policy")
        != "same_frozen_interpreter_sequential_distinct_processes"
        or value.get("worker_timeout_seconds") != WORKER_TIMEOUT_SECONDS
    ):
        raise QualificationControllerError(
            "freeze qualification policy drifted"
        )

    hashes = value.get("required_file_sha256s")
    if not isinstance(hashes, dict) or set(hashes) != set(
        REQUIRED_RELATIVE_FILES
    ):
        raise QualificationControllerError(
            "freeze implementation file registry drifted"
        )
    for relative in REQUIRED_RELATIVE_FILES:
        expected_hash = hashes.get(relative)
        if (
            not isinstance(expected_hash, str)
            or _SHA256.fullmatch(expected_hash) is None
        ):
            raise QualificationControllerError(
                "freeze contains an invalid implementation hash"
            )
        path = project_root / relative
        observed = hashlib.sha256(
            _regular_file_bytes(path, description=relative)
        ).hexdigest()
        if observed != expected_hash:
            raise QualificationControllerError(
                f"frozen implementation hash drifted: {relative}"
            )

    _validate_self_hashed_manifest(
        project_root / ARCHITECTURE_RELATIVE,
        schema="red_queen_poststop_rjmc_architecture_decision_v1",
        expected_self_sha256=ARCHITECTURE_DECISION_SELF_SHA256,
    )
    _validate_self_hashed_manifest(
        project_root / SOURCE_CUSTODY_RELATIVE,
        schema="quac_p1_source_custody_v1",
        expected_self_sha256=SOURCE_CUSTODY_SELF_SHA256,
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
                "controller did not use the frozen interpreter"
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
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise QualificationControllerError(
            "formal artifact mode drifted"
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
    if work_root.exists():
        if not work_root.is_dir() or any(work_root.iterdir()):
            raise OneShotRefusal("formal work root is not pristine")


def _prepare_pristine_work_root(work_root: Path) -> None:
    _assert_pristine_work_root(work_root)
    if not work_root.exists():
        work_root.mkdir(mode=0o700)
    os.chmod(work_root, 0o700)
    if any((work_root / name).exists() for name in OUTPUT_FILENAMES):
        raise OneShotRefusal("a formal artifact already exists")


def _validate_measurement_block(
    value: object, *, block_name: str
) -> None:
    expected_keys = BLOCK_REQUIRED_KEYS | (
        {"promotion_passed"}
        if block_name == "A_hold"
        else {"structural_variant"}
    )
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise WorkerFailure("worker measurement block shape drifted")
    if value.get("item_count") != 8:
        raise WorkerFailure("worker measurement item count drifted")
    for field in (
        "E0_total_utility",
        "E1_total_utility",
        "E1_minus_E0",
    ):
        if type(value.get(field)) is not int:
            raise WorkerFailure("worker utility type drifted")
    if value["E1_minus_E0"] <= 0:
        raise WorkerFailure("worker synthetic promotion failed")
    for field in (
        "topology_delta",
        "topology_raw_harm",
        "topology_required_complete",
    ):
        nested = value.get(field)
        if not isinstance(nested, dict) or set(nested) != set(TOPOLOGIES):
            raise WorkerFailure("worker topology registry drifted")
    if any(
        type(value["topology_delta"][name]) is not int
        or value["topology_delta"][name] <= 0
        for name in TOPOLOGIES
    ):
        raise WorkerFailure("worker held topology did not improve")
    if (
        value["topology_raw_harm"]["retention_trap"] < 0
        or value["topology_raw_harm"]["null_shift"] < 0
        or value["topology_required_complete"]["pair_complement"] is not True
    ):
        raise WorkerFailure("worker retention or interaction contract failed")
    if (
        block_name == "A_hold"
        and value.get("promotion_passed") is not True
    ):
        raise WorkerFailure("worker synthetic A_hold did not promote")
    if (
        block_name == "M_search"
        and value.get("structural_variant")
        != "extra_distractor_and_two_new_edges"
    ):
        raise WorkerFailure("worker synthetic M is not structurally fresh")


def validate_semantic_receipt(
    value: Mapping[str, Any]
) -> Mapping[str, Any]:
    if set(value) != RECEIPT_KEYS:
        raise WorkerFailure("worker receipt keyset drifted")
    body = dict(value)
    declared = body.pop("receipt_self_sha256", None)
    if (
        value.get("schema")
        != "qualify_quac_rjmc_source_free_v1_development_receipt"
        or value.get("version") != "qualify_quac_rjmc_source_free_v1"
        or value.get("status")
        != "passed_nonformal_source_free_development_qualification"
        or value.get("formal_result") is not False
        or value.get("architecture_decision_self_sha256")
        != ARCHITECTURE_DECISION_SELF_SHA256
        or value.get("evaluator_version")
        != "quac_rjmc_evaluator_v1"
        or value.get("fixture_provenance")
        != "hand_authored_source_free_synthetic_only"
        or value.get("fixture_topologies") != list(TOPOLOGIES)
        or value.get("complete_state_count_for_three_candidates") != 46
        or value.get("component_jackknife_head_count") != 5
        or value.get("antisymmetric") is not True
        or value.get("permutation_invariant") is not True
        or value.get("RAW_structural_zero") is not True
        or value.get("same_process_repeat_exact") is not True
        or value.get("qualification_weights_disposition")
        != "discarded_at_process_exit"
        or value.get("QuAC_source_payload_access_count") != 0
        or value.get("prior_private_source_access_count") != 0
        or value.get("online_or_API_evaluation_count") != 0
        or not isinstance(value.get("parameter_sha256"), str)
        or _SHA256.fullmatch(str(value.get("parameter_sha256"))) is None
        or not isinstance(value.get("behavior_sha256"), str)
        or _SHA256.fullmatch(str(value.get("behavior_sha256"))) is None
        or not isinstance(declared, str)
        or _SHA256.fullmatch(declared) is None
        or declared != stable_hash(body)
    ):
        raise WorkerFailure("worker semantic receipt binding drifted")
    _validate_measurement_block(value["A_hold"], block_name="A_hold")
    _validate_measurement_block(value["M_search"], block_name="M_search")
    return value


def _worker_environment(
    *, project_root: Path, work_root: Path, ordinal: int
) -> Mapping[str, str]:
    environment = _expected_worker_environment(
        project_root=project_root,
        work_root=work_root,
        ordinal=ordinal,
    )
    sandbox = work_root / "sandbox"
    if sandbox.is_symlink():
        raise WorkerFailure("worker sandbox root is a symlink")
    sandbox.mkdir(mode=0o700, exist_ok=True)
    worker_root = sandbox / f"worker_{ordinal}"
    worker_root.mkdir(mode=0o700, exist_ok=False)
    for name in ("home", "tmp", "cache"):
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
        process.communicate()
        raise WorkerFailure("qualification worker timed out") from exc
    if (
        process.returncode != 0
        or stderr != b""
        or not stdout
        or len(stdout) > MAX_JSON_BYTES
    ):
        raise WorkerFailure("qualification worker process failed")
    value = _decode_json(stdout, description="worker stdout receipt")
    validate_semantic_receipt(value)
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
        freeze=freeze,
        environment=environment,
        ordinal=ordinal,
    )


def _safe_failure_result(
    *,
    freeze: Mapping[str, Any],
    attempt_self_sha256: str,
    attempt_file_sha256: str,
    worker_file_sha256s: Sequence[str],
) -> Mapping[str, Any]:
    return _self_hashed(
        {
            "schema": f"{VERSION}_safe_result",
            "version": "v1",
            "study_id": STUDY_ID,
            "status": STOP_STATUS,
            "formal_result": True,
            "qualification_passed": False,
            "failure_code": "frozen_source_free_qualification_failed_closed",
            "freeze_file_sha256": freeze["_freeze_file_sha256"],
            "freeze_self_sha256": freeze["self_sha256"],
            "attempt_file_sha256": attempt_file_sha256,
            "attempt_self_sha256": attempt_self_sha256,
            "completed_worker_receipt_count": len(worker_file_sha256s),
            "worker_receipt_file_sha256s": list(worker_file_sha256s),
            "same_host_two_process_exact": False,
            "same_host_two_process_receipt_byte_exact": False,
            "retry_replay_resample_or_repair_count": 0,
            "QuAC_source_payload_access_count": 0,
            "online_or_API_evaluation_count": 0,
            "qualification_weights_disposition": (
                "discarded_with_each_worker_process_exit"
            ),
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
    expected_runtime_identity: Mapping[str, object] | None = None,
) -> Mapping[str, Any]:
    """Consume one formal attempt and return its immutable safe terminal."""

    freeze = load_and_validate_freeze(
        freeze_path,
        expected_formal_root=expected_formal_root,
        expected_python=expected_python,
        enforce_invocation_path=enforce_invocation_path,
        expected_runtime_identity=expected_runtime_identity,
        expected_environment=_outer_environment(expected_formal_root),
    )
    work_root = freeze["_work_root_path"]
    _prepare_pristine_work_root(work_root)

    attempt = _self_hashed(
        {
            "schema": f"{VERSION}_attempt",
            "version": "v1",
            "study_id": STUDY_ID,
            "status": "formal_attempt_consumed_once",
            "attempt_ordinal": 1,
            "controller_pid": os.getpid(),
            "freeze_path": str(freeze_path),
            "freeze_file_sha256": freeze["_freeze_file_sha256"],
            "freeze_self_sha256": freeze["self_sha256"],
            "qualification_worker_count": 2,
            "worker_launch_policy": (
                "same_frozen_interpreter_sequential_distinct_processes"
            ),
            "retry_replay_resample_or_repair_count": 0,
            "QuAC_source_payload_access_count": 0,
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
    try:
        for ordinal in (1, 2):
            environment = _worker_environment(
                project_root=freeze["_project_root_path"],
                work_root=work_root,
                ordinal=ordinal,
            )
            pid, raw = launcher(freeze, environment, ordinal)
            if (
                type(pid) is not int
                or pid <= 0
                or pid in worker_pids
            ):
                raise WorkerFailure(
                    "qualification workers were not distinct processes"
                )
            value = _decode_json(
                raw, description=f"worker {ordinal} receipt"
            )
            validate_semantic_receipt(value)
            if raw != _canonical_bytes(value):
                raise WorkerFailure(
                    "qualification worker receipt is not canonical"
                )
            worker_pids.append(pid)
            worker_receipts.append(raw)
            worker_file_sha256s.append(
                _exclusive_write_bytes(
                    work_root / f"worker_{ordinal}.receipt.json",
                    raw,
                )
            )
        if worker_receipts[0] != worker_receipts[1]:
            raise WorkerFailure(
                "two-process semantic receipts are not byte identical"
            )
        semantic = _decode_json(
            worker_receipts[0],
            description="common worker semantic receipt",
        )
        result = _self_hashed(
            {
                "schema": f"{VERSION}_safe_result",
                "version": "v1",
                "study_id": STUDY_ID,
                "status": PASS_STATUS,
                "formal_result": True,
                "qualification_passed": True,
                "freeze_file_sha256": freeze["_freeze_file_sha256"],
                "freeze_self_sha256": freeze["self_sha256"],
                "attempt_file_sha256": attempt_file_sha256,
                "attempt_self_sha256": attempt["attempt_self_sha256"],
                "worker_process_count": 2,
                "worker_pids_distinct": True,
                "worker_receipt_file_sha256s": worker_file_sha256s,
                "worker_semantic_receipt_self_sha256": (
                    semantic["receipt_self_sha256"]
                ),
                "same_host_two_process_exact": True,
                "same_host_two_process_receipt_byte_exact": True,
                "retry_replay_resample_or_repair_count": 0,
                "QuAC_source_payload_access_count": 0,
                "online_or_API_evaluation_count": 0,
                "qualification_weights_disposition": (
                    "discarded_with_each_worker_process_exit"
                ),
            },
            "result_self_sha256",
        )
    except Exception:
        result = _safe_failure_result(
            freeze=freeze,
            attempt_self_sha256=attempt["attempt_self_sha256"],
            attempt_file_sha256=attempt_file_sha256,
            worker_file_sha256s=worker_file_sha256s,
        )

    result_file_sha256 = _exclusive_write_json(
        work_root / "result.safe.json", result
    )
    passed = result["status"] == PASS_STATUS
    terminal = _self_hashed(
        {
            "schema": f"{VERSION}_formal_terminal",
            "version": "v1",
            "study_id": STUDY_ID,
            "status": PASS_STATUS if passed else STOP_STATUS,
            "formal_complete": True,
            "qualification_passed": passed,
            "freeze_file_sha256": freeze["_freeze_file_sha256"],
            "freeze_self_sha256": freeze["self_sha256"],
            "attempt_file_sha256": attempt_file_sha256,
            "attempt_self_sha256": attempt["attempt_self_sha256"],
            "result_safe_file_sha256": result_file_sha256,
            "result_safe_self_sha256": result["result_self_sha256"],
            "worker_receipt_file_sha256s": worker_file_sha256s,
            "same_host_two_process_exact": passed,
            "retry_replay_resample_or_repair_count": 0,
            "QuAC_source_payload_access_count": 0,
            "online_or_API_evaluation_count": 0,
            "next_action": (
                "source_download_may_be_separately_authorized"
                if passed
                else "close_RJMC_without_downloading_QuAC"
            ),
        },
        "terminal_self_sha256",
    )
    _exclusive_write_json(work_root / "formal_terminal.json", terminal)
    return terminal


def _load_qualifier(project_root: Path) -> Any:
    path = project_root / QUALIFIER_RELATIVE
    specification = importlib.util.spec_from_file_location(
        "_quac_rjmc_frozen_source_free_qualifier_v1", path
    )
    if specification is None or specification.loader is None:
        raise WorkerFailure("frozen qualifier cannot be imported")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    if not callable(getattr(module, "qualify", None)):
        raise WorkerFailure("frozen qualifier entry point drifted")
    return module


def worker_main(
    freeze_path: Path,
    *,
    ordinal: int,
    stdout_buffer: Any | None = None,
) -> int:
    """Emit exactly one canonical semantic receipt and no other stdout."""

    freeze = load_and_validate_freeze(
        freeze_path,
        expected_formal_root=None,
        expected_python=Path(sys.executable),
        enforce_invocation_path=True,
        expected_runtime_identity=None,
        expected_environment=_expected_worker_environment(
            project_root=freeze_path.parents[1],
            work_root=freeze_path.parents[2] / "work",
            ordinal=ordinal,
        ),
    )
    qualifier = _load_qualifier(freeze["_project_root_path"])
    receipt = qualifier.qualify()
    if not isinstance(receipt, dict):
        raise WorkerFailure("frozen qualifier did not return an object")
    validate_semantic_receipt(receipt)
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
    expected_runtime_identity: Mapping[str, object] | None = None,
) -> Mapping[str, Any]:
    """Validate the full bootstrap without consuming or creating an attempt."""

    freeze = load_and_validate_freeze(
        freeze_path,
        expected_formal_root=expected_formal_root,
        expected_python=expected_python,
        enforce_invocation_path=enforce_invocation_path,
        expected_runtime_identity=expected_runtime_identity,
        expected_environment=_outer_environment(expected_formal_root),
    )
    _assert_pristine_work_root(freeze["_work_root_path"])
    body = {
        "schema": f"{VERSION}_preflight_receipt",
        "version": "v1",
        "study_id": STUDY_ID,
        "status": "PASS_RJMC_SOURCE_FREE_PREFLIGHT",
        "formal_attempt_created": False,
        "freeze_file_sha256": freeze["_freeze_file_sha256"],
        "freeze_self_sha256": freeze["self_sha256"],
        "QuAC_source_payload_access_count": 0,
        "online_or_API_evaluation_count": 0,
        "retry_replay_resample_or_repair_count": 0,
    }
    return _self_hashed(body, "preflight_self_sha256")


def _write_bootstrap_stop(formal_root: Path) -> Mapping[str, Any]:
    """Write one safe STOP when formal bootstrap fails before an attempt."""

    work_root = formal_root / "work"
    _prepare_pristine_work_root(work_root)
    result = _self_hashed(
        {
            "schema": f"{VERSION}_bootstrap_safe_result",
            "version": "v1",
            "study_id": STUDY_ID,
            "status": STOP_STATUS,
            "formal_result": True,
            "qualification_passed": False,
            "attempt_created": False,
            "failure_code": "bootstrap_validation_failed_closed",
            "QuAC_source_payload_access_count": 0,
            "online_or_API_evaluation_count": 0,
            "retry_replay_resample_or_repair_count": 0,
            "next_action": "close_RJMC_without_downloading_QuAC",
        },
        "result_self_sha256",
    )
    result_file_sha256 = _exclusive_write_json(
        work_root / "result.safe.json", result
    )
    terminal = _self_hashed(
        {
            "schema": f"{VERSION}_bootstrap_formal_terminal",
            "version": "v1",
            "study_id": STUDY_ID,
            "status": STOP_STATUS,
            "formal_complete": True,
            "qualification_passed": False,
            "attempt_created": False,
            "result_safe_file_sha256": result_file_sha256,
            "result_safe_self_sha256": result["result_self_sha256"],
            "QuAC_source_payload_access_count": 0,
            "online_or_API_evaluation_count": 0,
            "retry_replay_resample_or_repair_count": 0,
            "next_action": "close_RJMC_without_downloading_QuAC",
        },
        "terminal_self_sha256",
    )
    _exclusive_write_json(work_root / "formal_terminal.json", terminal)
    return terminal


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen one-shot RJMC source-free qualification"
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
    except QualificationControllerError as exc:
        if not arguments.worker and not arguments.preflight:
            attempt_path = FORMAL_ROOT / "work" / "attempt.json"
            if not attempt_path.exists():
                try:
                    _write_bootstrap_stop(FORMAL_ROOT)
                except OneShotRefusal:
                    pass
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
