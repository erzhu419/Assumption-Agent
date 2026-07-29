"""Stable, iterative, source-free WikiSQL UAO runtime qualification.

This is an implementation-development harness, not a formal study and not a
scoring canary.  It admits work on a shared host only when configured resource
headroom exists, aggregates independent static failures, then launches Agent,
RAW, and official HippoRAG before waiting for any lane.  Every attempt is
immutable under one stable root and one stable systemd unit.  Busy shared
resources are a retryable deferral, never an implementation failure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence

from assumption_agent.benchmarks import (
    wikisql_uao_action_runtime_v1 as action_runtime,
)
from replication_runtime.wikisql_uao_formal_v1 import runner as formal
from replication_runtime.wikisql_uao_official_v1 import (
    contract as official_contract,
)
from replication_runtime.wikisql_uao_official_v1 import (
    worker as official_worker,
)
from replication_runtime.wikisql_uao_runtime_qualification import (
    alias_runtime,
    contract,
    resource_admission,
)


MODULE = "replication_runtime.wikisql_uao_runtime_qualification.runner"
RUNTIME_TIMEOUT_SECONDS = 5 * 60 * 60
COMMON_ORDER = contract.COMMON_ORDER
OFFICIAL_ORDER = contract.OFFICIAL_ORDER
EXPECTED_BABEL_VERSION = contract.EXPECTED_BABEL_VERSION
LANES = ("Agent", "RAW", "HippoRAG")
TERMINAL_STATUSES = frozenset(
    {
        "DEFERRED_SHARED_RESOURCE",
        "FAILED_INFRASTRUCTURE",
        "PASSED_FULL_STACK",
    }
)
_REQUIRED_SERVICE_LINES = frozenset(
    {
        "Type=oneshot",
        "UMask=0077",
        "CPUQuota=400%",
        "CPUWeight=25",
        "Nice=10",
        "IOWeight=25",
        "IOSchedulingClass=idle",
        "MemoryHigh=25769803776",
        "MemoryMax=34359738368",
        "MemorySwapMax=0",
        "TasksMax=96",
        "CPUAccounting=yes",
        "IOAccounting=yes",
        "MemoryAccounting=yes",
        "TasksAccounting=yes",
        "KillMode=control-group",
        "Restart=no",
        "SuccessExitStatus=75",
        "RestrictAddressFamilies=AF_UNIX",
        "IPAddressDeny=any",
        "NoNewPrivileges=yes",
        "PrivateTmp=yes",
        "TimeoutStartSec=6h",
    }
)


class QualificationRuntimeError(RuntimeError):
    """A runtime-qualification invariant failed."""


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: str
    detail: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        return {
            "detail": dict(self.detail),
            "name": self.name,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class LaunchResult:
    lane: str
    launched: bool
    returncode: int | None
    timed_out: bool
    launch_ordinal: int | None
    launch_error_class: str | None

    def payload(self) -> dict[str, object]:
        return {
            "lane": self.lane,
            "launch_error_class": self.launch_error_class,
            "launch_ordinal": self.launch_ordinal,
            "launched": self.launched,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
        }


def _write(path: Path, value: Mapping[str, object]) -> str:
    try:
        return formal._write_once(path, value, mode=0o600)
    except formal.WikiSQLUAOFormalError as exc:
        raise QualificationRuntimeError(
            "immutable qualification artifact cannot be published"
        ) from exc


def _load(path: Path, field: str) -> dict[str, object]:
    try:
        return formal._load_canonical_json(path, mode=0o600, field=field)
    except formal.WikiSQLUAOFormalError as exc:
        raise QualificationRuntimeError(
            f"{field} is unavailable or malformed"
        ) from exc


def _error_detail(exc: BaseException) -> dict[str, object]:
    message = " ".join(str(exc).split())
    return {
        "error_class": type(exc).__name__,
        "message": message[:500],
        "message_sha256": hashlib.sha256(
            message.encode("utf-8", errors="replace")
        ).hexdigest(),
    }


def _run_check(
    name: str, probe: Callable[[], Mapping[str, object] | None]
) -> CheckResult:
    try:
        detail = probe()
        if detail is None:
            detail = {}
        if type(detail) is not dict:
            raise QualificationRuntimeError(
                "check detail must be a mapping"
            )
        contract.canonical_json_bytes(detail)
        return CheckResult(name, "passed", detail)
    except BaseException as exc:
        return CheckResult(name, "failed", _error_detail(exc))


def _roots(
    config: contract.QualificationConfig, order: Sequence[str]
) -> tuple[Path, ...]:
    return tuple(config.tree(name).path for name in order)


def _environment(
    root: Path,
    *,
    cuda: str,
    module_roots: Sequence[Path],
    pythonhome: Path,
) -> dict[str, str]:
    return {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_MODULE_LOADING": "LAZY",
        "CUDA_VISIBLE_DEVICES": cuda,
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "HOME": str(root / "home"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        **contract.THREAD_ENVIRONMENT,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONHOME": str(pythonhome),
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join(map(str, module_roots)),
        "TEMP": str(root / "tmp"),
        "TMP": str(root / "tmp"),
        "TMPDIR": str(root / "tmp"),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }


def _dependency_arguments(
    config: contract.QualificationConfig,
    *,
    official: bool,
    output: Path,
) -> tuple[str, ...]:
    executable = config.file(
        "official_python_executable"
        if official
        else "python_executable"
    )
    order = OFFICIAL_ORDER if official else COMMON_ORDER
    result = [
        "--expected-python",
        str(executable.path),
        "--expected-python-sha256",
        executable.sha256,
        "--config-self-sha256",
        config.self_sha256,
        "--babel-root",
        str(config.tree("babel_dependency_tree").path),
        "--lane-receipt-output",
        str(output),
    ]
    for root in _roots(config, order):
        result += ["--pythonpath-root", str(root)]
    return tuple(result)


def synthetic_input() -> dict[str, object]:
    return official_contract.input_payload(
        items=[
            {
                "headers": ["City", "Score"],
                "item_id": hashlib.sha256(
                    b"WIKISQL-UAO-runtime-qualification-synthetic"
                ).hexdigest(),
                "question": "Which synthetic city has score seven?",
                "rows": [
                    [f"City-{index}", index] for index in range(11)
                ],
                "types": ["text", "real"],
            }
        ]
    )


def _all_gpu_device_paths() -> tuple[Path, ...]:
    return tuple(
        dict.fromkeys(
            (
                *formal._gpu_device_paths("0"),
                *formal._gpu_device_paths("1"),
            )
        )
    )


def build_commands(
    config: contract.QualificationConfig,
    paths: contract.AttemptPaths,
) -> Mapping[str, formal.CommandSpec]:
    common = _roots(config, COMMON_ORDER)
    official = _roots(config, OFFICIAL_ORDER)
    common_read = (
        *formal._existing_system_read_paths(),
        config.file("python_executable").path,
        config.tree("python_runtime_tree").path,
        *common,
    )
    official_read = (
        *formal._existing_system_read_paths(),
        config.file("official_python_executable").path,
        config.tree("official_python_runtime_tree").path,
        *official,
    )
    devices = _all_gpu_device_paths()

    def prefix(official_lane: bool) -> tuple[str, ...]:
        name = (
            "official_python_executable"
            if official_lane
            else "python_executable"
        )
        return (
            str(config.file(name).path),
            "-S",
            "-B",
            "-s",
            "-m",
            MODULE,
        )

    commands = {
        "Agent": formal.CommandSpec(
            name="Agent",
            argv=(
                *prefix(False),
                "agent",
                *_dependency_arguments(
                    config,
                    official=False,
                    output=paths.agent / "lane.safe.json",
                ),
                "--model",
                str(config.tree("encoder_model_tree").path),
                "--model-semantic-sha256",
                config.encoder_model_semantic_sha256,
            ),
            cwd=paths.agent,
            environment=_environment(
                paths.agent,
                cuda="1",
                module_roots=common,
                pythonhome=config.tree("python_runtime_tree").path,
            ),
            read_paths=(
                *common_read,
                config.tree("encoder_model_tree").path,
            ),
            write_paths=(
                paths.agent,
                Path("/proc/self/task"),
            ),
            device_paths=devices,
        ),
        "RAW": formal.CommandSpec(
            name="RAW",
            argv=(
                *prefix(False),
                "raw",
                *_dependency_arguments(
                    config,
                    official=False,
                    output=paths.raw / "lane.safe.json",
                ),
                "--input",
                str(paths.input),
            ),
            cwd=paths.raw,
            environment=_environment(
                paths.raw,
                cuda="",
                module_roots=common,
                pythonhome=config.tree("python_runtime_tree").path,
            ),
            read_paths=(*common_read, paths.input),
            write_paths=(paths.raw,),
        ),
        "HippoRAG": formal.CommandSpec(
            name="HippoRAG",
            argv=(
                *prefix(True),
                "hippo",
                *_dependency_arguments(
                    config,
                    official=True,
                    output=paths.hippo / "lane.safe.json",
                ),
                "--input",
                str(paths.input),
                "--action-output",
                str(paths.hippo / "action.private.json"),
                "--official-receipt-output",
                str(paths.hippo / "official.safe.json"),
                "--alias-receipt-output",
                str(paths.hippo / "model_alias.safe.json"),
                "--index-parent",
                str(paths.hippo / "indexes"),
                "--llm-model",
                str(config.tree("hippo_llm_model_tree").path),
                "--embedding-model",
                str(config.tree("encoder_model_tree").path),
            ),
            cwd=paths.hippo,
            environment=_environment(
                paths.hippo,
                cuda="0",
                module_roots=official,
                pythonhome=config.tree(
                    "official_python_runtime_tree"
                ).path,
            ),
            read_paths=(
                *official_read,
                config.tree("encoder_model_tree").path,
                config.tree("hippo_llm_model_tree").path,
                paths.input,
            ),
            write_paths=(
                paths.hippo,
                Path("/proc/self/task"),
            ),
            device_paths=devices,
        ),
    }
    return commands


def _dependency(arguments: argparse.Namespace) -> dict[str, object]:
    roots = tuple(
        Path(value).resolve() for value in arguments.pythonpath_root
    )
    if (
        Path(sys.executable).resolve()
        != arguments.expected_python.resolve()
        or sys.flags.no_site != 1
        or sys.flags.no_user_site != 1
        or os.environ.get("PYTHONPATH")
        != os.pathsep.join(map(str, roots))
        or tuple(
            Path(value).resolve()
            for value in sys.path[1 : 1 + len(roots)]
        )
        != roots
        or any(
            os.environ.get(key) != value
            for key, value in contract.THREAD_ENVIRONMENT.items()
        )
    ):
        raise QualificationRuntimeError(
            "fixed dependency order or thread environment drifted"
        )
    python_sha, _ = formal._file_sha256(arguments.expected_python)
    if python_sha != arguments.expected_python_sha256:
        raise QualificationRuntimeError(
            "interpreter binding drifted"
        )
    import babel

    origin = Path(babel.__file__).resolve()
    if (
        babel.__version__ != EXPECTED_BABEL_VERSION
        or origin
        != arguments.babel_root.resolve() / "babel/__init__.py"
    ):
        raise QualificationRuntimeError(
            "Babel 2.10.3 origin drifted"
        )
    babel_sha, _ = formal._file_sha256(origin)
    return {
        "babel_origin_file_sha256": babel_sha,
        "babel_version": babel.__version__,
        "config_self_sha256": arguments.config_self_sha256,
        "interpreter_file_sha256": python_sha,
        "pythonpath_order_sha256": contract.semantic_sha256(
            list(map(str, roots))
        ),
        "thread_environment": dict(contract.THREAD_ENVIRONMENT),
    }


def _lane_receipt(
    arguments: argparse.Namespace,
    lane: str,
    fields: Mapping[str, object],
) -> None:
    _write(
        arguments.lane_receipt_output,
        contract.addressed(
            {
                **_dependency(arguments),
                **fields,
                "API_call_count": 0,
                "effect_study_attempt_count": 0,
                "formal_source_access_count": 0,
                "lane": lane,
                "network_call_count": 0,
                "online_evaluator_call_count": 0,
                "replay_count": 0,
                "retry_count": 0,
                "schema": contract.lane_schema(lane),
                "status": "passed",
            }
        ),
    )


def _read_input(path: Path) -> dict[str, object]:
    value = _load(path, "synthetic input")
    items = official_contract.validate_input(value)
    if len(items) != 1 or len(items[0].rows) != 11:
        raise QualificationRuntimeError(
            "synthetic qualification item shape drifted"
        )
    return value


def _agent(arguments: argparse.Namespace) -> None:
    encoder = action_runtime.LocalSentenceTransformerEncoder(
        model_root=arguments.model,
        expected_model_sha256=arguments.model_semantic_sha256,
        device="cuda:0",
    )
    matrix = tuple(
        tuple(map(float, row))
        for row in encoder.encode(
            (
                "fixed synthetic question",
                "fixed synthetic evidence",
            ),
            batch_size=action_runtime.ENCODER_BATCH_SIZE,
        )
    )
    import torch

    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "1"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or len(matrix) != 2
        or not matrix[0]
        or len(matrix[0]) != len(matrix[1])
        or any(
            not math.isfinite(value)
            for row in matrix
            for value in row
        )
    ):
        raise QualificationRuntimeError(
            "real GPU1 MiniLM probe drifted"
        )
    _lane_receipt(
        arguments,
        "Agent",
        {
            "cuda_logical_device_count": 1,
            "embedding_dimension": len(matrix[0]),
            "embedding_matrix_sha256": action_runtime.canonical_sha256(
                matrix
            ),
            "model_semantic_sha256": (
                arguments.model_semantic_sha256
            ),
            "request_count": 2,
        },
    )


def _raw(arguments: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise QualificationRuntimeError(
            "RAW unexpectedly sees a GPU"
        )
    view = _read_input(arguments.input)
    action = action_runtime.run_raw(view_pack=view)
    action_runtime.decode_action_pack(
        action,
        expected_block="A_hold",
        expected_arm="RAW",
        expected_action_view_pack_sha256=str(view["self_sha256"]),
    )
    _lane_receipt(
        arguments,
        "RAW",
        {
            "action_pack_sha256": action["self_sha256"],
            "cpu_only": True,
            "input_pack_sha256": view["self_sha256"],
            "item_count": 1,
            "row_count": 11,
        },
    )


def _hippo(arguments: argparse.Namespace) -> None:
    dependency = _dependency(arguments)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise QualificationRuntimeError(
            "HippoRAG GPU0 assignment drifted"
        )
    view = _read_input(arguments.input)
    alias_receipt = alias_runtime.bind_and_verify_short_model_aliases(
        writable_root=arguments.index_parent.parent,
        llm_model_root=arguments.llm_model,
        embedding_model_root=arguments.embedding_model,
        identity_fn=formal.tree_identity,
    )
    _write(arguments.alias_receipt_output, alias_receipt)
    alias_root = (
        arguments.index_parent.parent
        / alias_runtime.ALIAS_DIRECTORY
    )
    previous_cwd = Path.cwd()
    try:
        os.chdir(alias_root)
        official_worker._require_offline_single_gpu_environment()
        official_worker._prepare_official_runtime()

        def production_core_factory(
            *,
            index_root: Path,
            item: official_contract.WikiSQLItem,
            item_ordinal: int,
            row_count: int,
        ) -> object:
            del item, item_ordinal
            return official_worker.build_official_core(
                index_root=index_root,
                llm_model=Path(alias_runtime.LLM_ALIAS),
                embedding_model=Path(
                    alias_runtime.EMBEDDING_ALIAS
                ),
                row_count=row_count,
            )

        artifacts = official_worker.run_once(
            private_input=view,
            action_output_path=arguments.action_output,
            safe_receipt_output_path=(
                arguments.official_receipt_output
            ),
            index_parent=arguments.index_parent,
            core_factory=production_core_factory,
        )
    finally:
        os.chdir(previous_cwd)
    action = _load(arguments.action_output, "official action")
    receipt = _load(
        arguments.official_receipt_output, "official receipt"
    )
    runtime = receipt.get("runtime")
    receipt_body = {
        key: child
        for key, child in receipt.items()
        if key != "self_sha256"
    }
    if (
        artifacts.action_pack.get("self_sha256")
        != action.get("self_sha256")
        or receipt.get("schema")
        != official_contract.SAFE_RECEIPT_SCHEMA
        or receipt.get("official_hipporag_commit")
        != official_contract.OFFICIAL_HIPPORAG_COMMIT
        or formal.semantic_sha256(receipt_body)
        != receipt.get("self_sha256")
        or receipt.get("item_count") != 1
        or not isinstance(runtime, dict)
        or runtime.get("index_call_count") != 1
        or runtime.get("retrieve_call_count") != 1
        or runtime.get("network_call_count") != 0
        or runtime.get("evaluator_call_count") != 0
        or runtime.get("retry_count") != 0
        or runtime.get("replay_count") != 0
    ):
        raise QualificationRuntimeError(
            "official construct/index/retrieve probe drifted"
        )
    action_file, _ = formal._file_sha256(arguments.action_output)
    official_file, _ = formal._file_sha256(
        arguments.official_receipt_output
    )
    alias_file, _ = formal._file_sha256(
        arguments.alias_receipt_output
    )
    _write(
        arguments.lane_receipt_output,
        contract.addressed(
            {
                **dependency,
                "API_call_count": 0,
                "action_file_sha256": action_file,
                "action_pack_sha256": action["self_sha256"],
                "alias_receipt_file_sha256": alias_file,
                "alias_receipt_self_sha256": alias_receipt[
                    "self_sha256"
                ],
                "cuda_logical_device_count": 1,
                "derived_hipporag_component_utf8_bytes": 40,
                "effect_study_attempt_count": 0,
                "formal_source_access_count": 0,
                "index_call_count": 1,
                "item_count": 1,
                "lane": "HippoRAG",
                "network_call_count": 0,
                "official_hipporag_commit": (
                    official_contract.OFFICIAL_HIPPORAG_COMMIT
                ),
                "official_receipt_file_sha256": official_file,
                "official_receipt_self_sha256": receipt[
                    "self_sha256"
                ],
                "online_evaluator_call_count": 0,
                "replay_count": 0,
                "retrieve_call_count": 1,
                "retry_count": 0,
                "row_count": 11,
                "schema": contract.lane_schema("HippoRAG"),
                "status": "passed",
            }
        ),
    )


def _service_source_check(
    config: contract.QualificationConfig,
) -> Mapping[str, object]:
    raw = config.file("service_unit").path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QualificationRuntimeError(
            "qualification service is not UTF-8"
        ) from exc
    lines = {
        line.strip()
        for line in text.splitlines()
        if line.strip()
    }
    required = {
        *_REQUIRED_SERVICE_LINES,
        (
            "WorkingDirectory="
            f"{contract.QUALIFICATION_ROOT}/reconstruction_v2"
        ),
    }
    if (
        not required <= lines
        or any(
            line.startswith(prefix)
            for line in lines
            for prefix in formal._FORBIDDEN_SERVICE_PREFIXES
        )
    ):
        raise QualificationRuntimeError(
            "shared-node service profile drifted"
        )
    exec_lines = [
        line for line in lines if line.startswith("ExecStart=")
    ]
    if (
        len(exec_lines) != 1
        or f"-m {MODULE} controller" not in exec_lines[0]
        or str(config.path) not in exec_lines[0]
        or str(config.file("python_executable").path)
        not in exec_lines[0]
        or "VECLIB_MAXIMUM_THREADS=1" not in exec_lines[0]
        or f"PYTHONHOME={contract.PYTHONHOME_ROOT}"
        not in exec_lines[0]
        or str(contract.BABEL_ROOT) not in exec_lines[0]
    ):
        raise QualificationRuntimeError(
            "qualification service ExecStart drifted"
        )
    return {
        "cpu_quota_percent": 400,
        "cpu_weight": 25,
        "io_weight": 25,
        "memory_high_bytes": 25769803776,
        "memory_max_bytes": 34359738368,
        "shared_node_profile": True,
        "tasks_max": 96,
    }


def _systemctl_properties(
    config: contract.QualificationConfig,
) -> Mapping[str, object]:
    runtime_root = Path(f"/run/user/{os.getuid()}")
    properties = (
        "ActiveState",
        "CPUQuotaPerSecUSec",
        "CPUWeight",
        "ControlGroup",
        "DropInPaths",
        "ExecStart",
        "FragmentPath",
        "IOWeight",
        "IOSchedulingClass",
        "IPAddressDeny",
        "InvocationID",
        "KillMode",
        "MemoryHigh",
        "MemoryMax",
        "MemorySwapMax",
        "NRestarts",
        "Nice",
        "NoNewPrivileges",
        "PrivateTmp",
        "Restart",
        "RestrictAddressFamilies",
        "SubState",
        "SuccessExitStatus",
        "TasksMax",
        "TimeoutStartUSec",
        "Type",
        "UMask",
    )
    argv = [
        str(config.file("systemctl_executable").path),
        "--user",
        "show",
        contract.UNIT_NAME,
        "--no-pager",
        *(f"--property={name}" for name in properties),
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
        raise QualificationRuntimeError(
            "live user-service query failed"
        ) from exc
    rows: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator != "=" or key in rows:
            raise QualificationRuntimeError(
                "live user-service output drifted"
            )
        rows[key] = value
    if set(rows) != set(properties):
        raise QualificationRuntimeError(
            "live user-service property set drifted"
        )
    expected_invocation = contract.invocation_id_from_environment(
        os.environ.get("INVOCATION_ID")
    )
    expected_fragment = contract.INSTALLED_UNIT_PATH
    try:
        fragment_raw = expected_fragment.read_bytes()
    except OSError as exc:
        raise QualificationRuntimeError(
            "installed user-service fragment is unavailable"
        ) from exc
    if (
        rows["NRestarts"] != "0"
        or rows["InvocationID"] != expected_invocation
        or rows["ActiveState"] not in {"activating", "active"}
        or rows["SubState"] not in {"start", "running"}
        or Path(rows["FragmentPath"]) != expected_fragment
        or rows["DropInPaths"] != ""
        or rows["Restart"] != "no"
        or rows["Type"] != "oneshot"
        or rows["TimeoutStartUSec"] != "6h"
        or rows["CPUQuotaPerSecUSec"] != "4s"
        or rows["CPUWeight"] != "25"
        or rows["IOWeight"] != "25"
        or rows["IOSchedulingClass"] != "3"
        or rows["MemoryHigh"] != "25769803776"
        or rows["MemoryMax"] != "34359738368"
        or rows["MemorySwapMax"] != "0"
        or rows["TasksMax"] != "96"
        or rows["Nice"] != "10"
        or rows["SuccessExitStatus"] != "75"
        or set(rows["IPAddressDeny"].split())
        != {"::/0", "0.0.0.0/0"}
        or rows["UMask"] != "0077"
        or rows["PrivateTmp"] != "yes"
        or rows["NoNewPrivileges"] != "yes"
        or rows["RestrictAddressFamilies"] != "AF_UNIX"
        or rows["KillMode"] != "control-group"
        or f"-m {MODULE} controller" not in rows["ExecStart"]
        or str(config.path) not in rows["ExecStart"]
        or hashlib.sha256(fragment_raw).hexdigest()
        != config.file("service_unit").sha256
    ):
        raise QualificationRuntimeError(
            "effective shared-node user-service profile drifted"
        )
    return {
        "active_state": rows["ActiveState"],
        "control_group_sha256": hashlib.sha256(
            rows["ControlGroup"].encode("utf-8")
        ).hexdigest(),
        "invocation_id_sha256": hashlib.sha256(
            expected_invocation.encode("ascii")
        ).hexdigest(),
        "nrestarts": 0,
        "shared_resource_caps_effective": True,
    }


def _dev_null_check() -> Mapping[str, object]:
    descriptor = -1
    try:
        descriptor = os.open(
            "/dev/null", os.O_RDWR | os.O_CLOEXEC
        )
        written = os.write(descriptor, b"x")
    except OSError as exc:
        raise QualificationRuntimeError(
            "/dev/null O_RDWR is unavailable"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if written != 1:
        raise QualificationRuntimeError(
            "/dev/null O_RDWR short write"
        )
    return {"dev_null_o_rdwr": True}


def _alias_path_check(
    config: contract.QualificationConfig,
    paths: contract.AttemptPaths,
) -> Mapping[str, object]:
    for name in ("hippo_llm_model_tree", "encoder_model_tree"):
        target = config.tree(name).path
        metadata = target.lstat()
        if (
            target.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
            or target.resolve(strict=True) != target
        ):
            raise QualificationRuntimeError(
                "short alias target is not a direct directory"
            )
    name_max = os.pathconf(paths.root, "PC_NAME_MAX")
    path_max = os.pathconf(paths.root, "PC_PATH_MAX")
    component_bytes = len(
        alias_runtime.DERIVED_HIPPORAG_COMPONENT.encode("utf-8")
    )
    longest_path_bytes = max(
        len(str(path).encode("utf-8"))
        for path in (
            paths.root,
            paths.hippo / alias_runtime.ALIAS_DIRECTORY,
            paths.hippo / "indexes",
        )
    )
    if (
        component_bytes != 40
        or component_bytes > name_max
        or longest_path_bytes >= path_max
    ):
        raise QualificationRuntimeError(
            "qualification alias or path length exceeds filesystem limit"
        )
    return {
        "derived_component_utf8_bytes": component_bytes,
        "longest_fixed_path_utf8_bytes": longest_path_bytes,
        "name_max_bytes": name_max,
        "path_max_bytes": path_max,
    }


def _command_check(
    commands: Mapping[str, formal.CommandSpec],
) -> Mapping[str, object]:
    if set(commands) != set(LANES):
        raise QualificationRuntimeError(
            "three-lane command registry drifted"
        )
    devices = set(_all_gpu_device_paths())
    if not {"nvidia0", "nvidia1"} <= {
        path.name for path in devices
    }:
        raise QualificationRuntimeError(
            "both physical GPU device nodes are unavailable"
        )
    for lane, command in commands.items():
        if (
            "-S" not in command.argv
            or "-B" not in command.argv
            or "-s" not in command.argv
            or command.argv.count(MODULE) != 1
            or any(
                command.environment.get(key) != value
                for key, value in contract.THREAD_ENVIRONMENT.items()
            )
        ):
            raise QualificationRuntimeError(
                f"{lane} command or thread environment drifted"
            )
    if (
        commands["Agent"].environment["CUDA_VISIBLE_DEVICES"] != "1"
        or commands["RAW"].environment["CUDA_VISIBLE_DEVICES"] != ""
        or commands["HippoRAG"].environment[
            "CUDA_VISIBLE_DEVICES"
        ]
        != "0"
        or set(commands["Agent"].device_paths) != devices
        or set(commands["HippoRAG"].device_paths) != devices
        or commands["RAW"].device_paths
        or Path("/proc/self/task")
        not in commands["Agent"].write_paths
        or Path("/proc/self/task")
        not in commands["HippoRAG"].write_paths
    ):
        raise QualificationRuntimeError(
            "CUDA routing or nested Landlock command drifted"
        )
    return {
        "all_native_thread_limits_one": True,
        "child_gpu_device_enumeration_count": len(devices),
        "lane_count": 3,
        "logical_gpu_assignment": {
            "Agent": "1",
            "HippoRAG": "0",
            "RAW": "",
        },
        "submission_contract": (
            "launch_all_three_before_waiting_for_any"
        ),
    }


def _capability_check(
    config: contract.QualificationConfig,
) -> Mapping[str, object]:
    if any(
        component.casefold()
        in {
            "dataset",
            "datasets",
            "label",
            "labels",
            "qrel",
            "qrels",
            "score",
            "scores",
            "source",
        }
        for binding in (*config.files.values(), *config.trees.values())
        for component in binding.path.parts
    ):
        raise QualificationRuntimeError(
            "formal or scoring capability entered qualification config"
        )
    return dict(contract.CAPABILITY_BOUNDARY)


def static_checks(
    config: contract.QualificationConfig,
    paths: contract.AttemptPaths,
    commands: Mapping[str, formal.CommandSpec],
) -> tuple[CheckResult, ...]:
    checks: list[
        tuple[str, Callable[[], Mapping[str, object] | None]]
    ] = []
    for name in sorted(config.files):
        checks.append(
            (
                f"file_binding.{name}",
                lambda name=name: (
                    config.file(name).verify(f"file binding {name}")
                    or {"verified": True}
                ),
            )
        )
    for name in sorted(config.trees):
        checks.append(
            (
                f"tree_binding.{name}",
                lambda name=name: (
                    config.tree(name).verify(f"tree binding {name}")
                    or {"verified": True}
                ),
            )
        )
    checks.extend(
        (
            (
                "encoder_model_semantic_identity",
                lambda: (
                    _verify_encoder_semantic(config)
                    or {"verified": True}
                ),
            ),
            (
                "service_source_profile",
                lambda: _service_source_check(config),
            ),
            (
                "systemd_effective_profile",
                lambda: _systemctl_properties(config),
            ),
            (
                "landlock_abi",
                lambda: _landlock_check(),
            ),
            ("dev_null_o_rdwr", _dev_null_check),
            (
                "short_alias_and_path_limits",
                lambda: _alias_path_check(config, paths),
            ),
            (
                "three_lane_command_and_thread_contract",
                lambda: _command_check(commands),
            ),
            (
                "non_scoring_capability_boundary",
                lambda: _capability_check(config),
            ),
        )
    )
    return tuple(_run_check(name, probe) for name, probe in checks)


def _verify_encoder_semantic(
    config: contract.QualificationConfig,
) -> None:
    if (
        action_runtime.directory_tree_sha256(
            config.tree("encoder_model_tree").path
        )
        != config.encoder_model_semantic_sha256
    ):
        raise QualificationRuntimeError(
            "encoder semantic tree binding drifted"
        )


def _landlock_check() -> Mapping[str, object]:
    abi = formal.landlock_abi_version()
    if type(abi) is not int or abi < 3:
        raise QualificationRuntimeError(
            "Landlock ABI is below the required minimum"
        )
    return {"abi": abi, "minimum_abi": 3}


def _make_attempt_directories(
    paths: contract.AttemptPaths,
) -> None:
    attempts = paths.root.parent
    attempts.mkdir(mode=0o700, parents=True, exist_ok=True)
    paths.root.mkdir(mode=0o700)
    for lane in (paths.agent, paths.raw, paths.hippo):
        lane.mkdir(mode=0o700)
        (lane / "home").mkdir(mode=0o700)
        (lane / "tmp").mkdir(mode=0o700)
    for path in (
        contract.QUALIFICATION_ROOT,
        contract.QUALIFICATION_ROOT / "control",
        attempts,
        paths.root,
        paths.agent,
        paths.raw,
        paths.hippo,
    ):
        metadata = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise QualificationRuntimeError(
                "qualification attempt directory metadata drifted"
            )


def _open_log(path: Path) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        return descriptor
    except OSError as exc:
        raise QualificationRuntimeError(
            "lane log cannot be created"
        ) from exc


def _launch_one(
    command: formal.CommandSpec,
    *,
    child_landlock: Callable[..., None],
) -> subprocess.Popen[bytes]:
    stdout = _open_log(command.cwd / "stdout.log")
    stderr = _open_log(command.cwd / "stderr.log")

    def isolate() -> None:
        child_landlock(
            read_paths=command.read_paths,
            write_paths=command.write_paths,
            device_paths=command.device_paths,
        )

    try:
        return subprocess.Popen(
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
        raise QualificationRuntimeError(
            f"{command.name} child could not be launched"
        ) from exc
    finally:
        os.close(stdout)
        os.close(stderr)


def launch_all_and_collect(
    commands: Mapping[str, formal.CommandSpec],
    *,
    child_landlock: Callable[..., None] = formal.apply_landlock,
    timeout_seconds: float = RUNTIME_TIMEOUT_SECONDS,
) -> tuple[LaunchResult, ...]:
    """Attempt every launch, then wait; one lane never suppresses another."""

    if set(commands) != set(LANES):
        raise QualificationRuntimeError(
            "three-lane command registry drifted"
        )
    processes: dict[str, subprocess.Popen[bytes]] = {}
    results: dict[str, LaunchResult] = {}
    launch_ordinal = 0
    deadline = time.monotonic() + timeout_seconds
    for lane in LANES:
        try:
            process = _launch_one(
                commands[lane], child_landlock=child_landlock
            )
            launch_ordinal += 1
            processes[lane] = process
            results[lane] = LaunchResult(
                lane=lane,
                launched=True,
                returncode=None,
                timed_out=False,
                launch_ordinal=launch_ordinal,
                launch_error_class=None,
            )
        except BaseException as exc:
            results[lane] = LaunchResult(
                lane=lane,
                launched=False,
                returncode=None,
                timed_out=False,
                launch_ordinal=None,
                launch_error_class=type(exc).__name__,
            )
    for lane in LANES:
        process = processes.get(lane)
        if process is None:
            continue
        remaining = max(0.0, deadline - time.monotonic())
        timed_out = False
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                returncode = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait(timeout=10)
        previous = results[lane]
        results[lane] = LaunchResult(
            lane=lane,
            launched=previous.launched,
            returncode=returncode,
            timed_out=timed_out,
            launch_ordinal=previous.launch_ordinal,
            launch_error_class=previous.launch_error_class,
        )
    return tuple(results[lane] for lane in LANES)


def _verify_lane(
    path: Path,
    lane: str,
    config: contract.QualificationConfig,
) -> dict[str, object]:
    value = _load(path, f"{lane} lane receipt")
    body = {
        key: child
        for key, child in value.items()
        if key != "self_sha256"
    }
    order = OFFICIAL_ORDER if lane == "HippoRAG" else COMMON_ORDER
    executable = config.file(
        "official_python_executable"
        if lane == "HippoRAG"
        else "python_executable"
    )
    babel_sha, _ = formal._file_sha256(
        config.tree("babel_dependency_tree").path
        / "babel/__init__.py"
    )
    if (
        value.get("schema") != contract.lane_schema(lane)
        or contract.semantic_sha256(body)
        != value.get("self_sha256")
        or value.get("status") != "passed"
        or value.get("lane") != lane
        or value.get("config_self_sha256") != config.self_sha256
        or value.get("interpreter_file_sha256")
        != executable.sha256
        or value.get("pythonpath_order_sha256")
        != contract.semantic_sha256(
            [
                str(config.tree(name).path)
                for name in order
            ]
        )
        or value.get("babel_version") != EXPECTED_BABEL_VERSION
        or value.get("babel_origin_file_sha256") != babel_sha
        or value.get("effect_study_attempt_count") != 0
        or value.get("formal_source_access_count") != 0
        or any(
            value.get(key) != 0
            for key in (
                "API_call_count",
                "network_call_count",
                "online_evaluator_call_count",
                "retry_count",
                "replay_count",
            )
        )
        or value.get("thread_environment")
        != dict(contract.THREAD_ENVIRONMENT)
    ):
        raise QualificationRuntimeError(
            f"{lane} safe receipt drifted"
        )
    return value


def _outer_landlock(
    config: contract.QualificationConfig,
    paths: contract.AttemptPaths,
) -> None:
    formal.apply_landlock(
        read_paths=(
            *formal._existing_system_read_paths(),
            config.path,
            *(binding.path for binding in config.files.values()),
            *(binding.path for binding in config.trees.values()),
        ),
        write_paths=(
            paths.root,
            contract.QUALIFICATION_ROOT / "ledger",
            Path("/tmp"),
            Path("/dev/null"),
            Path("/proc"),
        ),
        device_paths=_all_gpu_device_paths(),
    )


def _checks_receipt(
    results: Sequence[CheckResult],
) -> dict[str, object]:
    failed = sum(result.status == "failed" for result in results)
    return contract.addressed(
        {
            "check_count": len(results),
            "checks": [result.payload() for result in results],
            "failed_check_count": failed,
            "schema": contract.CHECK_SCHEMA,
            "status": (
                "all_static_checks_passed"
                if failed == 0
                else "static_checks_failed"
            ),
        }
    )


def _terminal(
    *,
    status: str,
    config_self_sha256: str | None,
    attempt_id: str | None,
    fields: Mapping[str, object],
) -> dict[str, object]:
    if status not in TERMINAL_STATUSES:
        raise QualificationRuntimeError(
            "unknown qualification terminal status"
        )
    return contract.addressed(
        {
            **fields,
            "API_or_online_evaluation_count": 0,
            "attempt_id": attempt_id,
            "config_self_sha256": config_self_sha256,
            "effect_study_attempt_count": 0,
            "formal_source_access_count": 0,
            "qualification_id": contract.QUALIFICATION_ID,
            "retryable": status == "DEFERRED_SHARED_RESOURCE",
            "schema": contract.TERMINAL_SCHEMA,
            "status": status,
        }
    )


def _ledger_directory() -> Path:
    path = contract.QUALIFICATION_ROOT / "ledger"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _append_ledger(
    *,
    invocation_id: str,
    kind: str,
    payload: Mapping[str, object],
) -> str:
    directory = _ledger_directory()
    existing = sorted(directory.glob("*.json"))
    previous_sha: str | None = None
    if existing:
        previous_raw = existing[-1].read_bytes()
        previous_sha = hashlib.sha256(previous_raw).hexdigest()
    sequence = len(existing) + 1
    event = contract.addressed(
        {
            "event": dict(payload),
            "invocation_id_sha256": hashlib.sha256(
                invocation_id.encode("ascii")
            ).hexdigest(),
            "kind": kind,
            "previous_event_file_sha256": previous_sha,
            "qualification_id": contract.QUALIFICATION_ID,
            "sequence": sequence,
        }
    )
    path = directory / (
        f"{sequence:06d}.{invocation_id}.{kind}.json"
    )
    return _write(path, event)


def _write_unlocked_deferral(
    invocation_id: str,
    reason: str,
) -> dict[str, object]:
    directory = contract.QUALIFICATION_ROOT / "deferred"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    terminal = _terminal(
        status="DEFERRED_SHARED_RESOURCE",
        config_self_sha256=None,
        attempt_id=None,
        fields={
            "deferral_reason": reason,
            "qualification_lock_acquired": False,
            "runtime_lane_launch_count": 0,
        },
    )
    _write(directory / f"{invocation_id}.safe.json", terminal)
    return terminal


def _resource_payload(decision: object) -> Mapping[str, object]:
    payload = getattr(decision, "receipt", None)
    if callable(payload):
        payload = payload()
    if type(payload) is not dict:
        payload = getattr(decision, "payload", None)
        if callable(payload):
            payload = payload()
    if type(payload) is not dict:
        raise QualificationRuntimeError(
            "resource admission receipt shape drifted"
        )
    contract.canonical_json_bytes(payload)
    return payload


def _resource_status(decision: object) -> str:
    status = getattr(decision, "status", None)
    if hasattr(status, "value"):
        status = status.value
    if not isinstance(status, str):
        raise QualificationRuntimeError(
            "resource admission status drifted"
        )
    return status


def run_controller(
    config_path: Path,
    *,
    invocation_id: str | None = None,
    lock_factory: Callable[..., object] | None = None,
    admission_probe: Callable[..., object] | None = None,
    static_probe: Callable[
        [
            contract.QualificationConfig,
            contract.AttemptPaths,
            Mapping[str, formal.CommandSpec],
        ],
        Sequence[CheckResult],
    ] = static_checks,
    outer_landlock: Callable[
        [contract.QualificationConfig, contract.AttemptPaths], None
    ] = _outer_landlock,
    launcher: Callable[..., Sequence[LaunchResult]] = (
        launch_all_and_collect
    ),
) -> Mapping[str, object]:
    """Run one iterative qualification attempt without scoring capability."""

    invocation = contract.invocation_id_from_environment(
        invocation_id
        if invocation_id is not None
        else os.environ.get("INVOCATION_ID")
    )
    if lock_factory is None:
        lock_factory = resource_admission.qualification_lock
    lock_path = Path(
        f"/run/user/{os.getuid()}/"
        "wikisql-uao-runtime-qualification.lock"
    )
    lock = lock_factory(lock_path)
    try:
        acquired = lock.__enter__()
    except resource_admission.ResourceBusyError:
        return _write_unlocked_deferral(
            invocation, "qualification_already_running"
        )
    except BaseException as exc:
        terminal = _terminal(
            status="FAILED_INFRASTRUCTURE",
            config_self_sha256=None,
            attempt_id=None,
            fields={
                "controller_error": _error_detail(exc),
                "failure_stage": "qualification_lock",
                "runtime_lane_launch_count": 0,
            },
        )
        directory = contract.QUALIFICATION_ROOT / "controller_failures"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write(
            directory / f"{invocation}.lock.safe.json",
            terminal,
        )
        return terminal
    if acquired is not True:
        lock.__exit__(None, None, None)
        return _write_unlocked_deferral(
            invocation, "qualification_already_running"
        )
    try:
        try:
            config = contract.load_config(config_path)
        except BaseException as exc:
            terminal = _terminal(
                status="FAILED_INFRASTRUCTURE",
                config_self_sha256=None,
                attempt_id=None,
                fields={
                    "controller_error": _error_detail(exc),
                    "failure_stage": "config_load",
                    "runtime_lane_launch_count": 0,
                },
            )
            _append_ledger(
                invocation_id=invocation,
                kind="config_failed",
                payload=terminal,
            )
            return terminal
        identifier = contract.attempt_id(
            invocation, config.self_sha256
        )
        paths = contract.AttemptPaths.for_attempt(identifier)
        if admission_probe is None:
            admission_probe = resource_admission.sample_and_decide
        try:
            decision = admission_probe(
                policy=config.resource_policy,
                expected_gpu_uuids=config.gpu_uuids,
                nvidia_smi_path=(
                    config.file("nvidia_smi_executable").path
                ),
            )
            admission_status = _resource_status(decision)
            admission_payload = _resource_payload(decision)
        except BaseException as exc:
            terminal = _terminal(
                status="FAILED_INFRASTRUCTURE",
                config_self_sha256=config.self_sha256,
                attempt_id=None,
                fields={
                    "controller_error": _error_detail(exc),
                    "failure_stage": "resource_telemetry",
                    "runtime_lane_launch_count": 0,
                },
            )
            _append_ledger(
                invocation_id=invocation,
                kind="telemetry_failed",
                payload=terminal,
            )
            return terminal
        if admission_status == "DEFERRED_SHARED_RESOURCE":
            terminal = _terminal(
                status="DEFERRED_SHARED_RESOURCE",
                config_self_sha256=config.self_sha256,
                attempt_id=None,
                fields={
                    "admission": admission_payload,
                    "failure_stage": None,
                    "qualification_lock_acquired": True,
                    "runtime_lane_launch_count": 0,
                },
            )
            _append_ledger(
                invocation_id=invocation,
                kind="deferred",
                payload=terminal,
            )
            return terminal
        if admission_status != "ADMITTED":
            terminal = _terminal(
                status="FAILED_INFRASTRUCTURE",
                config_self_sha256=config.self_sha256,
                attempt_id=None,
                fields={
                    "admission": admission_payload,
                    "failure_stage": "resource_telemetry",
                    "runtime_lane_launch_count": 0,
                },
            )
            _append_ledger(
                invocation_id=invocation,
                kind="telemetry_failed",
                payload=terminal,
            )
            return terminal
        try:
            _make_attempt_directories(paths)
            _write(
                paths.root / "attempt.started.safe.json",
                contract.addressed(
                    {
                        "admission": admission_payload,
                        "attempt_id": identifier,
                        "config_self_sha256": config.self_sha256,
                        "effect_study_attempt_count": 0,
                        "formal_source_access_count": 0,
                        "qualification_id": contract.QUALIFICATION_ID,
                        "status": "qualification_attempt_started",
                    }
                ),
            )
            _append_ledger(
                invocation_id=invocation,
                kind="started",
                payload={
                    "attempt_id": identifier,
                    "config_self_sha256": config.self_sha256,
                },
            )
        except BaseException as exc:
            terminal = _terminal(
                status="FAILED_INFRASTRUCTURE",
                config_self_sha256=config.self_sha256,
                attempt_id=identifier,
                fields={
                    "controller_error": _error_detail(exc),
                    "failure_stage": "attempt_creation",
                    "runtime_lane_launch_count": 0,
                },
            )
            _append_ledger(
                invocation_id=invocation,
                kind="attempt_creation_failed",
                payload=terminal,
            )
            return terminal
        command_build_error: CheckResult | None = None
        try:
            commands = build_commands(config, paths)
        except BaseException as exc:
            commands = {}
            command_build_error = CheckResult(
                "three_lane_command_build",
                "failed",
                _error_detail(exc),
            )
        try:
            checks = tuple(static_probe(config, paths, commands))
        except BaseException as exc:
            checks = (
                CheckResult(
                    "static_check_controller",
                    "failed",
                    _error_detail(exc),
                ),
            )
        if command_build_error is not None:
            checks = (command_build_error, *checks)
        check_receipt = _checks_receipt(checks)
        _write(paths.checks, check_receipt)
        failed_checks = [
            result.name
            for result in checks
            if result.status == "failed"
        ]
        if failed_checks:
            terminal = _terminal(
                status="FAILED_INFRASTRUCTURE",
                config_self_sha256=config.self_sha256,
                attempt_id=identifier,
                fields={
                    "admission": admission_payload,
                    "check_receipt_self_sha256": check_receipt[
                        "self_sha256"
                    ],
                    "failed_check_names": failed_checks,
                    "failure_stage": "aggregated_static_checks",
                    "runtime_lane_launch_count": 0,
                },
            )
            _write(paths.terminal, terminal)
            _append_ledger(
                invocation_id=invocation,
                kind="failed",
                payload=terminal,
            )
            return terminal
        try:
            outer_landlock(config, paths)
            _write(paths.input, synthetic_input())
            launch_results = tuple(launcher(commands))
            if {result.lane for result in launch_results} != set(
                LANES
            ):
                raise QualificationRuntimeError(
                    "launcher did not report every lane"
                )
            runtime_checks: list[CheckResult] = []
            for result in launch_results:
                runtime_checks.append(
                    CheckResult(
                        f"runtime_lane.{result.lane}.process",
                        (
                            "passed"
                            if result.launched
                            and result.returncode == 0
                            and not result.timed_out
                            else "failed"
                        ),
                        result.payload(),
                    )
                )
            for lane, receipt_path in (
                ("Agent", paths.agent / "lane.safe.json"),
                ("RAW", paths.raw / "lane.safe.json"),
                (
                    "HippoRAG",
                    paths.hippo / "lane.safe.json",
                ),
            ):
                runtime_checks.append(
                    _run_check(
                        f"runtime_lane.{lane}.receipt",
                        lambda lane=lane, receipt_path=receipt_path: {
                            "receipt_self_sha256": _verify_lane(
                                receipt_path, lane, config
                            )["self_sha256"]
                        },
                    )
                )
            all_checks = (*checks, *runtime_checks)
            complete_receipt = _checks_receipt(all_checks)
            _write(
                paths.root / "checks.complete.safe.json",
                complete_receipt,
            )
            runtime_failed = [
                result.name
                for result in runtime_checks
                if result.status == "failed"
            ]
            if runtime_failed:
                terminal = _terminal(
                    status="FAILED_INFRASTRUCTURE",
                    config_self_sha256=config.self_sha256,
                    attempt_id=identifier,
                    fields={
                        "admission": admission_payload,
                        "check_receipt_self_sha256": (
                            complete_receipt["self_sha256"]
                        ),
                        "failed_check_names": runtime_failed,
                        "failure_stage": "aggregated_runtime_checks",
                        "runtime_lane_launch_count": sum(
                            result.launched
                            for result in launch_results
                        ),
                    },
                )
            else:
                terminal = _terminal(
                    status="PASSED_FULL_STACK",
                    config_self_sha256=config.self_sha256,
                    attempt_id=identifier,
                    fields={
                        "admission": admission_payload,
                        "all_three_submitted_before_wait": True,
                        "check_receipt_self_sha256": (
                            complete_receipt["self_sha256"]
                        ),
                        "failure_stage": None,
                        "hipporag_construct_index_retrieve_count": 1,
                        "model_alias_component_utf8_bytes": 40,
                        "raw_cpu_action_count": 1,
                        "real_gpu1_minilm_encode_count": 1,
                        "runtime_lane_launch_count": 3,
                    },
                )
        except BaseException as exc:
            terminal = _terminal(
                status="FAILED_INFRASTRUCTURE",
                config_self_sha256=config.self_sha256,
                attempt_id=identifier,
                fields={
                    "controller_error": _error_detail(exc),
                    "failure_stage": "outer_or_runtime_controller",
                    "runtime_lane_launch_count": 0,
                },
            )
        _write(paths.terminal, terminal)
        _append_ledger(
            invocation_id=invocation,
            kind=(
                "passed"
                if terminal["status"] == "PASSED_FULL_STACK"
                else "failed"
            ),
            payload=terminal,
        )
        return terminal
    finally:
        lock.__exit__(None, None, None)


def _dependency_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--expected-python", required=True, type=Path
    )
    parser.add_argument("--expected-python-sha256", required=True)
    parser.add_argument("--config-self-sha256", required=True)
    parser.add_argument(
        "--pythonpath-root", action="append", required=True
    )
    parser.add_argument("--babel-root", required=True, type=Path)
    parser.add_argument(
        "--lane-receipt-output", required=True, type=Path
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    controller_parser = sub.add_parser("controller")
    controller_parser.add_argument(
        "--config", required=True, type=Path
    )
    agent_parser = sub.add_parser("agent")
    _dependency_parser(agent_parser)
    agent_parser.add_argument("--model", required=True, type=Path)
    agent_parser.add_argument(
        "--model-semantic-sha256", required=True
    )
    raw_parser = sub.add_parser("raw")
    _dependency_parser(raw_parser)
    raw_parser.add_argument("--input", required=True, type=Path)
    hippo_parser = sub.add_parser("hippo")
    _dependency_parser(hippo_parser)
    hippo_parser.add_argument("--input", required=True, type=Path)
    hippo_parser.add_argument(
        "--action-output", required=True, type=Path
    )
    hippo_parser.add_argument(
        "--official-receipt-output", required=True, type=Path
    )
    hippo_parser.add_argument(
        "--alias-receipt-output", required=True, type=Path
    )
    hippo_parser.add_argument(
        "--index-parent", required=True, type=Path
    )
    hippo_parser.add_argument(
        "--llm-model", required=True, type=Path
    )
    hippo_parser.add_argument(
        "--embedding-model", required=True, type=Path
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.mode == "controller":
        terminal = run_controller(arguments.config)
        print(
            json.dumps(
                terminal,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return (
            75
            if terminal.get("status")
            == "DEFERRED_SHARED_RESOURCE"
            else 0
        )
    {"agent": _agent, "raw": _raw, "hippo": _hippo}[
        arguments.mode
    ](arguments)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
