"""Shared-node WikiSQL UAO formal implementation after full qualification.

The efficacy contract remains byte/semantically identical to P4 v1.  This
wrapper incorporates only mechanisms already exercised by the stable,
non-scoring runtime qualification:

* verified short cwd-local model aliases for official HippoRAG;
* CUDA-compatible nested Landlock and the complete native-thread envelope;
* shared-node resource admission with cooperative deferral before an effect
  attempt or formal-source byte is opened; and
* bounded systemd CPU, memory, task, and I/O resources.

Resource shortage is not a formal failure.  It produces an invocation-scoped
``DEFERRED_SHARED_RESOURCE`` receipt and exit status 75 without creating the
formal attempt.  Once admitted, the original no-retry stopping rule applies.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Mapping, Sequence

from replication_runtime.wikisql_uao_runtime_qualification import (
    alias_runtime,
    prepare as qualification_prepare,
    resource_admission,
)


_SOURCE = (
    Path(__file__).parents[1] / "wikisql_uao_formal_v1/runner.py"
)
_ISOLATED_NAME = (
    "replication_runtime.wikisql_uao_formal_v5._isolated_formal_v1"
)
_SPEC = importlib.util.spec_from_file_location(_ISOLATED_NAME, _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("frozen formal v1 controller cannot be isolated")
_base = importlib.util.module_from_spec(_SPEC)
sys.modules[_ISOLATED_NAME] = _base
_SPEC.loader.exec_module(_base)

FORMAL_ROOT = Path("/home/erzhu419/wikisql_uao_p4_20260729/formal_v5")
UNIT_NAME = "wikisql-uao-p4-formal-v5.service"
INSTALLED_UNIT_PATH = (
    Path("/home/erzhu419/.config/systemd/user") / UNIT_NAME
)
SERVICE_RELATIVE_PATH = Path(
    "manifests/wikisql-uao-p4-formal-v5.service"
)
PYTHONHOME_ROOT = FORMAL_ROOT / "runtime_assets/python310_clean"
MODULE = "replication_runtime.wikisql_uao_formal_v5.runner"
SHARED_LOCK_PATH = Path(
    f"/run/user/{os.getuid()}/wikisql-uao-runtime-qualification.lock"
)
ADMISSION_PATH = FORMAL_ROOT / "control/resource_admission.safe.json"
ADMISSION_FAILURE_PATH = (
    FORMAL_ROOT / "control/resource_admission_failure.safe.json"
)
DEFERRAL_ROOT = FORMAL_ROOT / "control/resource_deferrals"

_base.FORMAL_ROOT = FORMAL_ROOT
_base.UNIT_NAME = UNIT_NAME
_base.INSTALLED_UNIT_PATH = INSTALLED_UNIT_PATH
_base.SERVICE_RELATIVE_PATH = SERVICE_RELATIVE_PATH

_original_load_config = _base.load_config
_original_lane_environment = _base._lane_environment
_original_verify_service_profile = _base._verify_service_profile
_original_verify_effective_service_profile = (
    _base._verify_effective_service_profile
)
_original_action_commands = _base._production_action_commands
RESOURCE_POLICY_SHA256 = _base.semantic_sha256(
    qualification_prepare.RESOURCE_POLICY
)


def load_config(path: Path):
    config = _original_load_config(path)
    if config.tree("python_runtime_tree").path != PYTHONHOME_ROOT:
        raise _base.WikiSQLUAOFormalError(
            "private Python home binding drifted"
        )
    return config


def _lane_environment(
    config,
    root: Path,
    *,
    cuda_visible_devices: str,
) -> dict[str, str]:
    environment = _original_lane_environment(
        config,
        root,
        cuda_visible_devices=cuda_visible_devices,
    )
    environment["PYTHONHOME"] = str(PYTHONHOME_ROOT)
    environment["VECLIB_MAXIMUM_THREADS"] = "1"
    return environment


def _verify_service_profile(raw: bytes, config) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _base.WikiSQLUAOFormalError(
            "formal v5 service is not UTF-8"
        ) from exc
    lines = {line.strip() for line in text.splitlines() if line.strip()}
    shared_lines = {
        "CPUQuota=400%",
        "CPUWeight=25",
        "IOWeight=25",
        "IOSchedulingClass=idle",
        "Nice=10",
        "MemoryHigh=25769803776",
        "MemoryMax=34359738368",
        "MemorySwapMax=0",
        "TasksMax=96",
        "CPUAccounting=yes",
        "IOAccounting=yes",
        "MemoryAccounting=yes",
        "TasksAccounting=yes",
        "SuccessExitStatus=75",
        "TimeoutStartSec=6h",
    }
    required_exec_tokens = {
        f"PYTHONHOME={PYTHONHOME_ROOT}",
        "MKL_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "VECLIB_MAXIMUM_THREADS=1",
        f"-m {MODULE}",
    }
    exec_lines = [
        line for line in lines if line.startswith("ExecStart=")
    ]
    retired_modules = tuple(
        f"replication_runtime.wikisql_uao_formal_v{revision}.runner"
        for revision in (1, 2, 3, 4)
    )
    if (
        not shared_lines.issubset(lines)
        or len(exec_lines) != 1
        or any(token not in exec_lines[0] for token in required_exec_tokens)
        or any(retired in exec_lines[0] for retired in retired_modules)
    ):
        raise _base.WikiSQLUAOFormalError(
            "shared-node formal v5 service profile drifted"
        )
    rewritten = raw.replace(
        MODULE.encode("ascii"),
        retired_modules[0].encode("ascii"),
    )
    for current, legacy in (
        (b"CPUQuota=400%", b"CPUQuota=700%"),
        (b"MemoryMax=34359738368", b"MemoryMax=42949672960"),
        (b"TasksMax=96", b"TasksMax=128"),
        (b"TimeoutStartSec=6h", b"TimeoutStartSec=infinity"),
    ):
        rewritten = rewritten.replace(current, legacy)
    _original_verify_service_profile(rewritten, config)


def _systemctl_rows(
    config,
    properties: Sequence[str],
) -> Mapping[str, str]:
    runtime_root = Path(f"/run/user/{os.getuid()}")
    command = [
        str(config.file("systemctl_executable").path),
        "--user",
        "show",
        UNIT_NAME,
        "--no-pager",
        *(f"--property={name}" for name in properties),
    ]
    try:
        completed = subprocess.run(
            command,
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
        raise _base.WikiSQLUAOFormalError(
            "shared-node systemd attestation failed"
        ) from exc
    rows: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator != "=" or key in rows:
            raise _base.WikiSQLUAOFormalError(
                "shared-node systemd attestation shape drifted"
            )
        rows[key] = value
    if set(rows) != set(properties):
        raise _base.WikiSQLUAOFormalError(
            "shared-node systemd property registry drifted"
        )
    return rows


def _service_probe(config):
    service = _base._systemctl_attestation(
        config,
        unit_name=UNIT_NAME,
        installed_unit_path=INSTALLED_UNIT_PATH,
    )
    properties = (
        "CPUWeight",
        "CPUAccounting",
        "IOAccounting",
        "IOWeight",
        "IOSchedulingClass",
        "InvocationID",
        "MemoryAccounting",
        "MemoryHigh",
        "MemorySwapMax",
        "Nice",
        "SuccessExitStatus",
        "TasksAccounting",
    )
    rows = _systemctl_rows(config, properties)
    if (
        rows["CPUAccounting"] != "yes"
        or rows["CPUWeight"] != "25"
        or rows["IOAccounting"] != "yes"
        or rows["IOWeight"] != "25"
        or rows["IOSchedulingClass"] != "3"
        or rows["InvocationID"] != service.invocation_id
        or rows["MemoryHigh"] != "25769803776"
        or rows["MemoryAccounting"] != "yes"
        or rows["MemorySwapMax"] != "0"
        or rows["Nice"] != "10"
        or rows["SuccessExitStatus"] != "75"
        or rows["TasksAccounting"] != "yes"
    ):
        raise _base.WikiSQLUAOFormalError(
            "effective shared-node service controls drifted"
        )
    return service


def _verify_effective_service_profile(service, unit_raw: bytes) -> None:
    if (
        service.timeout_start_usec != "6h"
        or service.cpu_quota_per_sec_usec != "4s"
        or service.memory_max != "34359738368"
        or service.tasks_max != "96"
    ):
        raise _base.WikiSQLUAOFormalError(
            "effective shared-node service limits drifted"
        )
    legacy_projection = replace(
        service,
        timeout_start_usec="infinity",
        cpu_quota_per_sec_usec="7s",
        memory_max="42949672960",
        tasks_max="128",
    )
    _original_verify_effective_service_profile(
        legacy_projection, unit_raw
    )


def _all_gpu_device_paths() -> tuple[Path, ...]:
    return tuple(
        dict.fromkeys(
            (
                *_base._gpu_device_paths("0"),
                *_base._gpu_device_paths("1"),
            )
        )
    )


def _outer_landlock(config, paths) -> None:
    _base.apply_landlock(
        read_paths=(
            *_base._existing_system_read_paths(),
            *(binding.path for binding in config.files.values()),
            *(binding.path for binding in config.trees.values()),
        ),
        write_paths=(
            paths.root,
            Path("/tmp"),
            Path("/dev/null"),
            Path("/proc"),
        ),
        device_paths=_all_gpu_device_paths(),
    )


def _replace_option(
    argv: Sequence[str],
    option: str,
    value: str,
) -> tuple[str, ...]:
    matches = [index for index, token in enumerate(argv) if token == option]
    if len(matches) != 1 or matches[0] + 1 >= len(argv):
        raise _base.WikiSQLUAOFormalError(
            "formal HippoRAG model option drifted"
        )
    result = list(argv)
    result[matches[0] + 1] = value
    return tuple(result)


def _action_commands(config, paths, source):
    commands = dict(_original_action_commands(config, paths, source))
    devices = _all_gpu_device_paths()
    for lane in ("Agent", "HippoRAG"):
        command = commands[lane]
        commands[lane] = replace(
            command,
            write_paths=(
                *command.write_paths,
                Path("/proc/self/task"),
            ),
            device_paths=devices,
        )

    alias_receipt = alias_runtime.bind_and_verify_short_model_aliases(
        writable_root=paths.hippo_root,
        llm_model_root=config.tree("hippo_llm_model_tree").path,
        embedding_model_root=config.tree("encoder_model_tree").path,
        identity_fn=_base.tree_identity,
    )
    _base._write_once(
        paths.hippo_root / "model_alias.safe.json",
        alias_receipt,
        mode=0o600,
    )
    alias_root = paths.hippo_root / alias_runtime.ALIAS_DIRECTORY
    hippo = commands["HippoRAG"]
    hippo_argv = _replace_option(
        hippo.argv, "--llm-model", alias_runtime.LLM_ALIAS
    )
    hippo_argv = _replace_option(
        hippo_argv,
        "--embedding-model",
        alias_runtime.EMBEDDING_ALIAS,
    )
    commands["HippoRAG"] = replace(
        hippo,
        argv=hippo_argv,
        cwd=alias_root,
    )
    _base._validate_action_command_isolation(commands, paths)
    return commands


def _gpu_probe_shared(config):
    observed = _base._gpu_attestation(config)
    return replace(observed, compute_process_count=0)


def _private_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        metadata = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise _base.WikiSQLUAOFormalError(
                "resource receipt directory drifted"
            )
        return
    try:
        path.mkdir(mode=0o700)
    except OSError as exc:
        raise _base.WikiSQLUAOFormalError(
            "resource receipt directory cannot be created"
        ) from exc
    _base._fsync_directory(path.parent)


def _resource_receipt(
    *,
    service,
    decision,
    status: str,
) -> dict[str, object]:
    return _base._self_hashed(
        {
            "API_or_online_evaluation_count": 0,
            "effect_study_attempt_count": 0,
            "external_process_identity_is_not_an_exclusion_predicate": True,
            "formal_source_access_count": 0,
            "invocation_id_sha256": hashlib.sha256(
                service.invocation_id.encode("ascii")
            ).hexdigest(),
            "resource_admission": decision.to_dict(),
            "resource_policy_sha256": RESOURCE_POLICY_SHA256,
            "schema": "wikisql_uao_formal_v5_resource_admission_v1",
            "shared_node_must_be_empty": False,
            "status": status,
            "study_id": _base.STUDY_ID,
        }
    )


def _write_deferral(service, decision) -> Mapping[str, object]:
    _private_directory(DEFERRAL_ROOT)
    receipt = _resource_receipt(
        service=service,
        decision=decision,
        status="DEFERRED_SHARED_RESOURCE",
    )
    _base._write_once(
        DEFERRAL_ROOT / f"{service.invocation_id}.safe.json",
        receipt,
        mode=0o600,
    )
    return receipt


def _resource_policy():
    return resource_admission.ResourcePolicy.parse(
        qualification_prepare.RESOURCE_POLICY
    )


def run_formal_production(config_path: Path) -> Mapping[str, object]:
    paths = _base.FormalPaths.for_root(FORMAL_ROOT)
    if (
        paths.attempt.exists()
        or paths.attempt.is_symlink()
        or paths.terminal.exists()
        or paths.terminal.is_symlink()
        or ADMISSION_PATH.exists()
        or ADMISSION_PATH.is_symlink()
        or ADMISSION_FAILURE_PATH.exists()
        or ADMISSION_FAILURE_PATH.is_symlink()
    ):
        raise _base.WikiSQLUAOFormalError(
            "formal v5 already admitted, failed, or attempted"
        )
    config = load_config(config_path)
    service = _service_probe(config)
    lock = resource_admission.QualificationFlock(SHARED_LOCK_PATH)
    try:
        try:
            acquired = lock.acquire_nonblocking()
        except resource_admission.ResourceAdmissionInfrastructureError as exc:
            decision = resource_admission.AdmissionDecision(
                status=resource_admission.FAILED_INFRASTRUCTURE,
                reason_codes=(exc.reason_code,),
            )
        else:
            if not acquired:
                decision = resource_admission.AdmissionDecision(
                    status=resource_admission.DEFERRED_SHARED_RESOURCE,
                    reason_codes=("SHARED_WIKISQL_FLOCK_OCCUPIED",),
                )
            else:
                decision = resource_admission.sample_and_decide(
                    _resource_policy(),
                    config.gpu_uuids,
                    config.file("nvidia_smi_executable").path,
                )
        if decision.status == resource_admission.DEFERRED_SHARED_RESOURCE:
            return _write_deferral(service, decision)
        if decision.status != resource_admission.ADMITTED:
            receipt = _resource_receipt(
                service=service,
                decision=decision,
                status="FAILED_INFRASTRUCTURE_PRE_ATTEMPT",
            )
            _base._write_once(
                ADMISSION_FAILURE_PATH, receipt, mode=0o600
            )
            return receipt
        receipt = _resource_receipt(
            service=service,
            decision=decision,
            status="ADMITTED_SHARED_RESOURCE",
        )
        _base._write_once(ADMISSION_PATH, receipt, mode=0o600)
        return _base._run_with_dependencies(
            config_path, _base.PRODUCTION_DEPENDENCIES
        )
    finally:
        lock.release()


_base.load_config = load_config
_base._lane_environment = _lane_environment
_base._verify_service_profile = _verify_service_profile
_base._verify_effective_service_profile = (
    _verify_effective_service_profile
)
_base.PRODUCTION_DEPENDENCIES = replace(
    _base.PRODUCTION_DEPENDENCIES,
    service_probe=_service_probe,
    gpu_probe=_gpu_probe_shared,
    outer_landlock=_outer_landlock,
    action_commands=_action_commands,
)

CONFIG_SCHEMA = _base.CONFIG_SCHEMA
STUDY_ID = _base.STUDY_ID
WikiSQLUAOFormalError = _base.WikiSQLUAOFormalError


def _parser():
    return _base._parser()


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
    status = terminal.get("status")
    if status == "completed_protocol_valid":
        return 0
    if status == "DEFERRED_SHARED_RESOURCE":
        return resource_admission.EX_TEMPFAIL
    if status == "FAILED_INFRASTRUCTURE_PRE_ATTEMPT":
        return resource_admission.EX_SOFTWARE
    return 1


def __getattr__(name: str):
    return getattr(_base, name)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
