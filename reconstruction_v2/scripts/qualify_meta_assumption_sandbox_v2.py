#!/usr/bin/env python3
"""One-shot source-free sandbox canary for the UAO v2 deployment.

This canary exercises the exact user-service controls needed by the formal
controller without opening a reality source or consuming formal capability.
It has no retry, model, API, evaluator, validation, or test channel.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping


CANARY_ROOT = Path(
    "/home/erzhu419/uao_p2_20260729/source_free_sandbox_canary_v2"
)
PROJECT_ROOT = CANARY_ROOT / "reconstruction_v2"
WORK_ROOT = CANARY_ROOT / "work"
CONTROLLER_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "run_meta_assumption_source_free_qualification_v2.py"
)
CANARY_SERVICE_UNIT = (
    "meta-assumption-source-free-sandbox-canary-v2.service"
)
CANARY_SERVICE_RELATIVE = (
    "manifests/meta-assumption-source-free-sandbox-canary-v2.service"
)
INSTALLED_CANARY_SERVICE_PATH = (
    Path("/home/erzhu419/.config/systemd/user") / CANARY_SERVICE_UNIT
)
ATTEMPT_PATH = WORK_ROOT / "canary_attempt.json"
RESULT_PATH = WORK_ROOT / "sandbox_canary.safe.json"


class CanaryError(RuntimeError):
    """The one allowed infrastructure canary failed closed."""


def _load_controller() -> Any:
    spec = importlib.util.spec_from_file_location(
        "uao_v2_controller_for_sandbox_canary",
        CONTROLLER_PATH,
    )
    if spec is None or spec.loader is None:
        raise CanaryError("v2 controller import specification is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _audit_exact_canary_tree(controller: Any) -> Mapping[str, Any]:
    expected_files = set(controller.REQUIRED_RELATIVE_FILES)
    expected_directories = {""}
    for relative in expected_files:
        parent = Path(relative).parent
        while str(parent) != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent

    observed_files: set[str] = set()
    observed_directories = {""}
    hashes: dict[str, str] = {}
    for path in sorted(
        PROJECT_ROOT.rglob("*"),
        key=lambda item: item.relative_to(PROJECT_ROOT).as_posix(),
    ):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise CanaryError(
                f"canary project contains a symlink: {relative}"
            )
        if stat.S_ISDIR(metadata.st_mode):
            observed_directories.add(relative)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise CanaryError(
                f"canary project contains a special file: {relative}"
            )
        observed_files.add(relative)
        raw = controller._regular_file_bytes(
            path, description=f"canary project file {relative}"
        )
        hashes[relative] = hashlib.sha256(raw).hexdigest()

    if (
        observed_files != expected_files
        or observed_directories != expected_directories
    ):
        raise CanaryError("canary project exact allowlist drifted")
    body = {
        "expected_relative_files": sorted(expected_files),
        "file_sha256s": hashes,
    }
    return {
        "canary_project_file_count": len(expected_files),
        "canary_project_tree_self_sha256": controller.stable_hash(body),
    }


def _attest_installed_canary_service(
    controller: Any,
) -> Mapping[str, Any]:
    expected_source = PROJECT_ROOT / CANARY_SERVICE_RELATIVE
    metadata = INSTALLED_CANARY_SERVICE_PATH.lstat()
    if not stat.S_ISLNK(metadata.st_mode):
        raise CanaryError("installed canary service is not an exact symlink")
    target = os.readlink(INSTALLED_CANARY_SERVICE_PATH)
    if target != str(expected_source):
        raise CanaryError("installed canary service target drifted")
    raw = controller._regular_file_bytes(
        expected_source, description="frozen canary service source"
    )
    return {
        "installed_canary_service_binding_attested": True,
        "installed_canary_service_path": str(
            INSTALLED_CANARY_SERVICE_PATH
        ),
        "installed_canary_service_source_sha256": hashlib.sha256(
            raw
        ).hexdigest(),
        "installed_canary_service_target": target,
    }


def _attest_service_network_denial(controller: Any) -> Mapping[str, Any]:
    original_unit = controller.FORMAL_SERVICE_UNIT
    try:
        controller.FORMAL_SERVICE_UNIT = CANARY_SERVICE_UNIT
        return controller._attest_formal_service_sandbox()
    finally:
        controller.FORMAL_SERVICE_UNIT = original_unit


def _run_frozen_child(
    controller: Any,
    *,
    expected_controller_sha256: str,
) -> Mapping[str, Any]:
    environment = controller._worker_environment(
        work_root=WORK_ROOT,
        ordinal=1,
    )
    output = WORK_ROOT / "sandbox" / "worker_1" / "child_probe.txt"
    program = (
        "import hashlib\n"
        "from pathlib import Path\n"
        "import sys\n"
        "raw = Path(sys.argv[1]).read_bytes()\n"
        "Path(sys.argv[2]).write_text("
        "hashlib.sha256(raw).hexdigest() + '\\n', encoding='ascii')\n"
    )
    completed = subprocess.run(
        (
            str(controller.FROZEN_PYTHON),
            "-I",
            "-B",
            "-c",
            program,
            str(CONTROLLER_PATH),
            str(output),
        ),
        cwd=PROJECT_ROOT,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if (
        completed.returncode != 0
        or completed.stdout
        or completed.stderr
    ):
        raise CanaryError("frozen child execution was not exact and silent")
    observed = output.read_text(encoding="ascii")
    if observed != expected_controller_sha256 + "\n":
        raise CanaryError("frozen child project read or work write drifted")
    return {
        "frozen_child_exit_code": completed.returncode,
        "frozen_child_project_read_attested": True,
        "frozen_child_work_write_attested": True,
        "frozen_child_stdout_bytes": 0,
        "frozen_child_stderr_bytes": 0,
    }


def run() -> Mapping[str, Any]:
    os.umask(0o077)
    controller = _load_controller()
    expected_script = (
        PROJECT_ROOT / "scripts" / "qualify_meta_assumption_sandbox_v2.py"
    )
    if Path(__file__).resolve() != expected_script:
        raise CanaryError("canary invocation path drifted")
    if Path(sys.executable) != controller.FROZEN_PYTHON:
        raise CanaryError("canary Python executable drifted")
    controller._validate_isolated_launch(
        controller._outer_environment(CANARY_ROOT)
    )
    controller._validate_architecture_manifest(
        PROJECT_ROOT / controller.ARCHITECTURE_RELATIVE
    )
    tree = _audit_exact_canary_tree(controller)
    service = _attest_installed_canary_service(controller)

    controller._assert_pristine_work_root(WORK_ROOT)
    WORK_ROOT.mkdir(mode=0o700, exist_ok=True)
    os.chmod(WORK_ROOT, 0o700)
    attempt = controller._self_hashed(
        {
            "schema": "meta_assumption_source_free_sandbox_canary_attempt_v2",
            "version": "v2",
            "study_id": controller.STUDY_ID,
            "status": "source_free_sandbox_canary_attempt_consumed_once",
            "attempt_ordinal": 1,
            "formal_capability_consumed": False,
            "formal_attempt_created": False,
            "retry_replay_resample_or_repair_count": 0,
            "formal_source_access_count": 0,
            "source_payload_access_count": 0,
            "network_call_count": 0,
            "model_asset_access_count": 0,
            "api_call_count": 0,
            "online_evaluator_call_count": 0,
            "validation_access_count": 0,
            "test_access_count": 0,
        },
        "attempt_self_sha256",
    )
    attempt_file_sha256 = controller._exclusive_write_json(
        ATTEMPT_PATH, attempt
    )

    network = _attest_service_network_denial(controller)
    landlock = controller._apply_landlock_filesystem_sandbox(
        python_executable=controller.FROZEN_PYTHON,
        project_root=PROJECT_ROOT,
        work_root=WORK_ROOT,
    )
    parent_probe = WORK_ROOT / "parent_write_probe.txt"
    controller._exclusive_write_bytes(parent_probe, b"UAO_V2_CANARY\n")
    controller_raw = controller._regular_file_bytes(
        CONTROLLER_PATH, description="v2 controller after Landlock"
    )
    controller_sha256 = hashlib.sha256(controller_raw).hexdigest()
    child = _run_frozen_child(
        controller,
        expected_controller_sha256=controller_sha256,
    )

    body = {
        "schema": "meta_assumption_source_free_sandbox_canary_safe_result_v2",
        "version": "v2",
        "study_id": controller.STUDY_ID,
        "status": "PASS_UAO_V2_SOURCE_FREE_SANDBOX_CANARY",
        "architecture_decision_self_sha256": (
            controller.ARCHITECTURE_DECISION_SELF_SHA256
        ),
        "development_receipt_reused_without_reselection": (
            controller.EXPECTED_DEVELOPMENT_RECEIPT_SELF_SHA256
        ),
        "attempt_ordinal": 1,
        "attempt_file_sha256": attempt_file_sha256,
        "attempt_self_sha256": attempt["attempt_self_sha256"],
        **tree,
        **service,
        **network,
        **landlock,
        **child,
        "parent_project_read_attested": True,
        "parent_work_write_attested": True,
        "formal_capability_consumed": False,
        "formal_attempt_created": False,
        "formal_result": False,
        "efficacy_evidence": False,
        "retry_replay_resample_or_repair_count": 0,
        "formal_source_access_count": 0,
        "source_payload_access_count": 0,
        "network_call_count": 0,
        "model_asset_access_count": 0,
        "api_call_count": 0,
        "online_evaluator_call_count": 0,
        "validation_access_count": 0,
        "test_access_count": 0,
        "next_action": (
            "authorize_exactly_one_fresh_UAO_v2_source_free_formal_attempt"
        ),
    }
    result = controller._self_hashed(body, "canary_self_sha256")
    controller._exclusive_write_json(RESULT_PATH, result)
    return result


def main() -> int:
    try:
        run()
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
