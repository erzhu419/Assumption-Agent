from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Mapping

import pytest

from assumption_agent.benchmarks import (
    meta_assumption_synthetic_worlds_v1 as qualification,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "run_meta_assumption_source_free_qualification_v2.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_meta_assumption_source_free_qualification_v2_for_test",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
controller = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = controller
SPEC.loader.exec_module(controller)
TEST_PYTHON = Path("/usr/bin/python3.10")
TEST_RUNTIME_IDENTITY = controller.FROZEN_RUNTIME_IDENTITIES[
    str(TEST_PYTHON)
]


@pytest.fixture
def posix_tmp_path() -> Path:
    path = Path(
        tempfile.mkdtemp(prefix="meta_assumption_controller_", dir="/tmp")
    )
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _copy_frozen_tree(formal_root: Path) -> Path:
    project = formal_root / "reconstruction_v2"
    for relative in controller.REQUIRED_RELATIVE_FILES:
        source = PROJECT_ROOT / relative
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return project


def _freeze(formal_root: Path) -> Path:
    project = _copy_frozen_tree(formal_root)
    semantic = qualification.qualify()
    required = {
        relative: hashlib.sha256(
            (project / relative).read_bytes()
        ).hexdigest()
        for relative in controller.REQUIRED_RELATIVE_FILES
    }
    body = {
        "schema": controller.FREEZE_SCHEMA,
        "version": "v2",
        "study_id": controller.STUDY_ID,
        "formal_root": str(formal_root),
        "project_root": str(project),
        "work_root": str(formal_root / "work"),
        "python_executable": str(TEST_PYTHON),
        "python_executable_sha256": (
            TEST_RUNTIME_IDENTITY[
                "python_executable_sha256"
            ]
        ),
        "python_executable_size_bytes": (
            TEST_RUNTIME_IDENTITY[
                "python_executable_size_bytes"
            ]
        ),
        "python_executable_mode": (
            TEST_RUNTIME_IDENTITY[
                "python_executable_mode"
            ]
        ),
        "python_version": TEST_RUNTIME_IDENTITY["python_version"],
        "implementation_commit": "1" * 40,
        "architecture_decision_self_sha256": (
            controller.ARCHITECTURE_DECISION_SELF_SHA256
        ),
        "expected_development_receipt_self_sha256": (
            controller.EXPECTED_DEVELOPMENT_RECEIPT_SELF_SHA256
        ),
        "ontology_hash": semantic["ontology_hash"],
        "formal_source_access_count_before_qualification": 0,
        "source_payload_access_count_before_qualification": 0,
        "network_call_count_before_qualification": 0,
        "online_or_API_evaluation_count_before_qualification": 0,
        "formal_attempt_limit": 1,
        "qualification_worker_count": 2,
        "worker_launch_policy": (
            "same_frozen_python_sequential_distinct_processes"
        ),
        "worker_timeout_seconds": controller.WORKER_TIMEOUT_SECONDS,
        "retry_replay_resample_or_repair_count": 0,
        "required_file_sha256s": required,
    }
    value = {**body, "self_sha256": controller.stable_hash(body)}
    path = project / "manifests" / controller.FREEZE_FILENAME
    path.write_bytes(controller._canonical_bytes(value))
    return path


def _load(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _canonical_receipt() -> bytes:
    receipt = qualification.qualify()
    assert (
        receipt["self_sha256"]
        == controller.EXPECTED_DEVELOPMENT_RECEIPT_SELF_SHA256
    )
    return controller._canonical_bytes(receipt)


def test_freeze_binds_python_architecture_and_complete_file_closure(
    posix_tmp_path: Path,
) -> None:
    formal_root = posix_tmp_path / "formal"
    freeze_path = _freeze(formal_root)
    loaded = controller.load_and_validate_freeze(
        freeze_path,
        expected_formal_root=formal_root,
        expected_python=TEST_PYTHON,
        enforce_invocation_path=False,
    )

    assert loaded["_python_path"] == TEST_PYTHON
    assert loaded["architecture_decision_self_sha256"] == (
        controller.ARCHITECTURE_DECISION_SELF_SHA256
    )
    assert set(loaded["required_file_sha256s"]) == set(
        controller.REQUIRED_RELATIVE_FILES
    )
    assert (
        controller.ARCHITECTURE_RELATIVE
        in controller.REQUIRED_RELATIVE_FILES
    )

    catalog = (
        formal_root
        / "reconstruction_v2"
        / "assumption_agent"
        / "universal_assumption_ontology_v1.py"
    )
    catalog.write_bytes(catalog.read_bytes() + b"\n")
    with pytest.raises(
        controller.QualificationControllerError,
        match="implementation hash drifted",
    ):
        controller.load_and_validate_freeze(
            freeze_path,
            expected_formal_root=formal_root,
            expected_python=TEST_PYTHON,
            enforce_invocation_path=False,
        )


def test_qualifier_capability_closure_rejects_import_and_file_read(
    posix_tmp_path: Path,
) -> None:
    formal_root = posix_tmp_path / "formal"
    project = _copy_frozen_tree(formal_root)
    audit = controller._audit_qualifier_capability_closure(project)
    assert (
        audit[
            "external_source_network_model_API_or_process_channel_found"
        ]
        is False
    )
    assert set(audit["audited_relative_files"]) == set(
        controller.QUALIFIER_IMPORT_CLOSURE_RELATIVE_FILES
    )

    subject = (
        project
        / "assumption_agent"
        / "benchmarks"
        / "meta_assumption_synthetic_worlds_v1.py"
    )
    original = subject.read_bytes()
    subject.write_bytes(original + b"\nimport socket\n")
    with pytest.raises(
        controller.QualificationControllerError,
        match="capability import is forbidden",
    ):
        controller._audit_qualifier_capability_closure(project)

    subject.write_bytes(
        original
        + b"\ndef forbidden_read(path):\n"
        + b"    return path.read_text(encoding='utf-8')\n"
    )
    with pytest.raises(
        controller.QualificationControllerError,
        match="file method is forbidden",
    ):
        controller._audit_qualifier_capability_closure(project)


def test_exact_frozen_project_tree_rejects_extra_import_asset(
    posix_tmp_path: Path,
) -> None:
    formal_root = posix_tmp_path / "formal"
    freeze_path = _freeze(formal_root)
    project = freeze_path.parents[1]
    audit = controller._audit_exact_frozen_project_tree(project)
    assert audit["unregistered_file_count"] == 0
    assert audit["symlink_or_special_file_count"] == 0

    extra = project / "assumption_agent" / "unregistered_module.py"
    extra.write_text("VALUE = 1\n", encoding="ascii")
    with pytest.raises(
        controller.QualificationControllerError,
        match="exact allowlist drifted",
    ):
        controller._audit_exact_frozen_project_tree(project)


def test_formal_service_attestation_binds_unit_and_denied_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePath:
        def __init__(self, _value: str) -> None:
            pass

        def read_bytes(self) -> bytes:
            return (
                b"0::/user.slice/user-1000.slice/user@1000.service/"
                + controller.FORMAL_SERVICE_UNIT.encode("ascii")
                + b"\n"
            )

    def denied_socket(*_args: object, **_kwargs: object) -> object:
        raise OSError(controller.errno.EAFNOSUPPORT, "denied")

    monkeypatch.setattr(controller, "Path", FakePath)
    monkeypatch.setattr(controller.socket, "socket", denied_socket)
    attestation = controller._attest_formal_service_sandbox()
    assert attestation["formal_service_unit"] == (
        controller.FORMAL_SERVICE_UNIT
    )
    assert attestation["AF_UNIX_socket_creation_denied"] is True
    assert attestation["AF_UNIX_socket_denial_errno"] == (
        controller.errno.EAFNOSUPPORT
    )
    assert attestation["AF_INET_socket_creation_denied"] is True
    assert attestation["AF_INET_socket_denial_errno"] == (
        controller.errno.EAFNOSUPPORT
    )
    assert attestation["AF_INET6_socket_creation_denied"] is True
    assert attestation["AF_INET6_socket_denial_errno"] == (
        controller.errno.EAFNOSUPPORT
    )


def test_installed_service_must_be_exact_symlink_to_frozen_source(
    posix_tmp_path: Path,
) -> None:
    project = posix_tmp_path / "reconstruction_v2"
    source = project / controller.SERVICE_RELATIVE
    source.parent.mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / controller.SERVICE_RELATIVE, source)
    installed = posix_tmp_path / controller.FORMAL_SERVICE_UNIT
    installed.symlink_to(source)
    attestation = controller._attest_installed_service_binding(
        project_root=project,
        installed_service_path=installed,
    )
    assert attestation["installed_formal_service_binding_attested"] is True
    assert attestation["installed_formal_service_target"] == str(source)
    installed.unlink()
    installed.write_bytes(source.read_bytes())
    with pytest.raises(
        controller.QualificationControllerError,
        match="not an exact symlink",
    ):
        controller._attest_installed_service_binding(
            project_root=project,
            installed_service_path=installed,
        )


def test_landlock_allows_frozen_tree_and_work_but_denies_home_and_etc(
    posix_tmp_path: Path,
) -> None:
    formal_root = posix_tmp_path / "formal"
    project = _copy_frozen_tree(formal_root)
    work = formal_root / "work"
    work.mkdir(mode=0o700)
    probe = """
import importlib.util
import json
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("landlock_subject", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
result = module._apply_landlock_filesystem_sandbox(
    python_executable=Path(sys.argv[4]),
    project_root=Path(sys.argv[2]),
    work_root=Path(sys.argv[3]),
)
print(json.dumps(result, sort_keys=True))
"""
    completed = subprocess.run(
        (
            str(TEST_PYTHON),
            "-I",
            "-B",
            "-c",
            probe,
            str(SCRIPT),
            str(project),
            str(work),
            str(TEST_PYTHON),
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", errors="replace"
    )
    attestation = json.loads(completed.stdout.decode("ascii"))
    assert attestation["landlock_filesystem_restriction_attested"] is True
    assert attestation["landlock_abi"] >= controller.LANDLOCK_MINIMUM_ABI
    assert attestation["landlock_home_directory_denial_errno"] in {
        controller.errno.EACCES,
        controller.errno.EPERM,
    }
    assert attestation["landlock_outside_direct_file_denial_errno"] in {
        controller.errno.EACCES,
        controller.errno.EPERM,
    }


def test_strict_receipt_requires_10_of_10_40_of_40_tamper_and_real_noop() -> None:
    receipt = qualification.qualify()
    validated = controller.validate_semantic_receipt(
        receipt,
        expected_ontology_hash=str(receipt["ontology_hash"]),
        expected_self_sha256=str(receipt["self_sha256"]),
    )
    assert validated["correct_identification_count"] == 10
    assert validated["wrong_claims_with_counterevidence_count"] == 40
    assert validated["tamper_rejected_count"] == 19

    tampered = dict(receipt)
    tampered["tamper_rejected_count"] = 18
    body = dict(tampered)
    body.pop("self_sha256")
    tampered["self_sha256"] = controller.stable_hash(body)
    with pytest.raises(controller.WorkerFailure, match="binding drifted"):
        controller.validate_semantic_receipt(
            tampered,
            expected_ontology_hash=str(receipt["ontology_hash"]),
            expected_self_sha256=str(tampered["self_sha256"]),
        )

    rows = validated["world_compilations"]
    assert isinstance(rows, list)
    assert sum(
        row["compiled_operator"] != "PRESERVE_BASELINE" for row in rows
    ) == 8
    assert sum(
        row["compiled_operator"] == "PRESERVE_BASELINE" for row in rows
    ) == 2


def test_controller_runs_two_ordered_workers_and_writes_durable_pass(
    posix_tmp_path: Path,
) -> None:
    formal_root = posix_tmp_path / "formal"
    freeze_path = _freeze(formal_root)
    raw = _canonical_receipt()
    calls: list[int] = []
    environments: list[Mapping[str, str]] = []

    def launcher(
        freeze: Mapping[str, object],
        environment: Mapping[str, str],
        ordinal: int,
    ) -> tuple[int, bytes]:
        assert freeze["_python_path"] == TEST_PYTHON
        assert ordinal == len(calls) + 1
        assert "PATH" not in environment
        assert "HTTP_PROXY" not in environment
        assert environment["HF_HUB_OFFLINE"] == "1"
        assert environment["HOME"].endswith(
            f"/worker_{ordinal}/home"
        )
        environments.append(dict(environment))
        calls.append(ordinal)
        return 5000 + ordinal, raw

    terminal = controller.run_controller(
        freeze_path,
        expected_formal_root=formal_root,
        expected_python=TEST_PYTHON,
        enforce_invocation_path=False,
        launcher=launcher,
    )
    assert calls == [1, 2]
    assert environments[0]["HOME"] != environments[1]["HOME"]
    assert terminal["status"] == controller.NONFORMAL_TEST_PASS_STATUS
    assert terminal["formal_result"] is False
    assert terminal["formal_complete"] is False
    assert terminal["qualification_passed"] is False
    assert terminal["nonformal_controller_test_passed"] is True
    assert terminal["efficacy_evidence"] is False
    assert terminal["next_action"] == (
        "nonformal_test_complete_no_reality_authorization"
    )

    work = formal_root / "work"
    assert (work / "worker_1.receipt.json").read_bytes() == raw
    assert (work / "worker_2.receipt.json").read_bytes() == raw
    for filename in controller.OUTPUT_FILENAMES:
        path = work / filename
        assert path.is_file()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    result = _load(work / "result.safe.json")
    assert result["status"] == controller.NONFORMAL_TEST_PASS_STATUS
    assert result["formal_result"] is False
    assert result["qualification_passed"] is False
    assert result["nonformal_controller_test_passed"] is True
    assert result["known_mechanism_identification"] == "10/10"
    assert result["wrong_claim_counterevidence"] == "40/40"
    assert result["active_program_compile_count"] == 8
    assert result["preserve_baseline_compile_count"] == 2
    assert result["tamper_rejection"] == "19/19"
    assert result["probe_evidence_bundle_count"] == 50
    assert result["probe_receipts_trusted_recomputed"] is True
    assert result["minimum_commitment_two_stage"] is True
    assert result["active_runtime_differential"] == "8/8"
    assert result["noop_runtime_equivalence"] == "2/2"
    assert result["wrong_operator_harm"] == "32/32_across_10/10_worlds"
    assert result["claim_order_invariance"] == "10/10"
    assert result["probe_rule_order_invariance"] == "10/10"
    assert result["world_id_invariance"] == "10/10"
    assert result["expected_label_invariance"] == "10/10"
    assert result["safe_recomputed_counts"] == (
        qualification.qualify()["safe_recomputed_counts"]
    )
    assert result["same_host_two_process_receipt_byte_exact"] is True
    assert result["formal_source_access_count"] == 0
    assert result["network_call_count"] == 0
    assert result["model_asset_access_count"] == 0
    assert result["api_call_count"] == 0
    assert result["online_evaluator_call_count"] == 0
    assert result["validation_access_count"] == 0
    assert result["test_access_count"] == 0
    assert result["formal_service_unit_attested"] is False
    assert result["installed_formal_service_binding_attested"] is False
    assert result["AF_UNIX_socket_creation_denied"] is False
    assert result["AF_INET_socket_creation_denied"] is False
    assert result["AF_INET6_socket_creation_denied"] is False
    result_body = dict(result)
    result_self = result_body.pop("result_self_sha256")
    assert result_self == controller.stable_hash(result_body)
    terminal_body = dict(terminal)
    terminal_self = terminal_body.pop("terminal_self_sha256")
    assert terminal_self == controller.stable_hash(terminal_body)
    assert terminal["result_safe_file_sha256"] == hashlib.sha256(
        (work / "result.safe.json").read_bytes()
    ).hexdigest()
    assert terminal["result_safe_self_sha256"] == result_self

    with pytest.raises(controller.OneShotRefusal, match="not pristine"):
        controller.run_controller(
            freeze_path,
            expected_formal_root=formal_root,
            expected_python=TEST_PYTHON,
            enforce_invocation_path=False,
            launcher=launcher,
        )
    assert calls == [1, 2]


def test_worker_failure_stops_once_without_retry_or_second_terminal(
    posix_tmp_path: Path,
) -> None:
    formal_root = posix_tmp_path / "formal"
    freeze_path = _freeze(formal_root)
    raw = _canonical_receipt()
    calls: list[int] = []

    def launcher(
        _freeze: Mapping[str, object],
        _environment: Mapping[str, str],
        ordinal: int,
    ) -> tuple[int, bytes]:
        calls.append(ordinal)
        return (
            6000 + ordinal,
            raw if ordinal == 1 else raw + b" ",
        )

    terminal = controller.run_controller(
        freeze_path,
        expected_formal_root=formal_root,
        expected_python=TEST_PYTHON,
        enforce_invocation_path=False,
        launcher=launcher,
    )
    assert calls == [1, 2]
    assert terminal["status"] == controller.NONFORMAL_TEST_STOP_STATUS
    assert terminal["formal_result"] is False
    assert terminal["qualification_passed"] is False
    assert terminal["nonformal_controller_test_passed"] is False
    assert terminal["next_action"] == (
        "nonformal_test_complete_no_reality_authorization"
    )
    result = _load(formal_root / "work" / "result.safe.json")
    assert result["status"] == controller.NONFORMAL_TEST_STOP_STATUS
    assert result["formal_result"] is False
    assert result["completed_worker_receipt_count"] == 1
    assert result["retry_replay_resample_or_repair_count"] == 0
    assert result["same_host_two_process_receipt_byte_exact"] is False
    assert result["failure_stage"] == "worker_2_launch_and_validation"
    assert result["failure_exception_type"] == "WorkerFailure"
    assert result["failure_issue_id"] == (
        "worker_receipt_is_not_canonical_bytes"
    )
    assert result["worker_exit_code"] is None
    assert result["worker_stderr_sha256"] is None
    assert (formal_root / "work" / "formal_terminal.json").is_file()
    result_body = dict(result)
    assert result_body.pop("result_self_sha256") == controller.stable_hash(
        result_body
    )
    terminal_body = dict(terminal)
    assert terminal_body.pop("terminal_self_sha256") == (
        controller.stable_hash(terminal_body)
    )


def test_preflight_validates_without_consuming_or_writing_attempt(
    posix_tmp_path: Path,
) -> None:
    formal_root = posix_tmp_path / "formal"
    freeze_path = _freeze(formal_root)
    receipt = controller.run_preflight(
        freeze_path,
        expected_formal_root=formal_root,
        expected_python=TEST_PYTHON,
        enforce_invocation_path=False,
    )
    assert receipt["status"] == "PASS_UAO_V2_SOURCE_FREE_PREFLIGHT"
    assert receipt["formal_attempt_created"] is False
    assert not (formal_root / "work").exists()


def test_actual_isolated_worker_has_empty_stderr_and_canonical_stdout(
    posix_tmp_path: Path,
) -> None:
    formal_root = posix_tmp_path / "formal"
    freeze_path = _freeze(formal_root)
    work = formal_root / "work"
    work.mkdir(mode=0o700)
    environment = controller._worker_environment(
        work_root=work,
        ordinal=1,
    )
    completed = subprocess.run(
        (
            str(TEST_PYTHON),
            "-I",
            "-B",
            str(
                formal_root
                / "reconstruction_v2"
                / controller.CONTROLLER_RELATIVE
            ),
            "--worker",
            "--worker-ordinal",
            "1",
            "--freeze",
            str(freeze_path),
        ),
        cwd=formal_root / "reconstruction_v2",
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=controller.WORKER_TIMEOUT_SECONDS,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", errors="replace"
    )
    assert completed.stderr == b""
    value = controller._decode_json(
        completed.stdout, description="actual worker receipt"
    )
    controller.validate_semantic_receipt(
        value,
        expected_ontology_hash=str(value["ontology_hash"]),
        expected_self_sha256=(
            controller.EXPECTED_DEVELOPMENT_RECEIPT_SELF_SHA256
        ),
    )
    assert completed.stdout == controller._canonical_bytes(value)
    assert not (work / "attempt.json").exists()
    assert not (work / "result.safe.json").exists()
    assert not (work / "formal_terminal.json").exists()


def test_default_controller_launches_two_actual_sequential_workers(
    posix_tmp_path: Path,
) -> None:
    formal_root = posix_tmp_path / "formal"
    freeze_path = _freeze(formal_root)

    terminal = controller.run_controller(
        freeze_path,
        expected_formal_root=formal_root,
        expected_python=TEST_PYTHON,
        enforce_invocation_path=False,
    )
    assert terminal["status"] == controller.NONFORMAL_TEST_PASS_STATUS
    assert terminal["formal_result"] is False
    assert terminal["qualification_passed"] is False
    assert terminal["nonformal_controller_test_passed"] is True
    assert terminal["same_host_two_process_receipt_byte_exact"] is True
    work = formal_root / "work"
    first = (work / "worker_1.receipt.json").read_bytes()
    second = (work / "worker_2.receipt.json").read_bytes()
    assert first == second == _canonical_receipt()
    result = _load(work / "result.safe.json")
    assert result["worker_process_count"] == 2
    assert result["worker_pids_distinct"] is True
    assert result["same_host_two_process_receipt_byte_exact"] is True


def test_bootstrap_failure_writes_one_safe_stop_before_attempt(
    monkeypatch: pytest.MonkeyPatch,
    posix_tmp_path: Path,
) -> None:
    formal_root = posix_tmp_path / "formal"
    freeze_path = (
        formal_root
        / "reconstruction_v2"
        / "manifests"
        / controller.FREEZE_FILENAME
    )
    freeze_path.parent.mkdir(parents=True)
    freeze_path.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(controller, "FORMAL_ROOT", formal_root)

    assert controller.main(["--freeze", str(freeze_path)]) == 2
    work = formal_root / "work"
    assert not (work / "attempt.json").exists()
    first_result = (work / "result.safe.json").read_bytes()
    first_terminal = (work / "formal_terminal.json").read_bytes()
    result = _load(work / "result.safe.json")
    terminal = _load(work / "formal_terminal.json")
    assert result["status"] == controller.STOP_STATUS
    assert result["attempt_created"] is False
    assert result["retry_replay_resample_or_repair_count"] == 0
    assert terminal["status"] == controller.STOP_STATUS
    assert terminal["attempt_created"] is False
    assert stat.S_IMODE(
        (work / "result.safe.json").stat().st_mode
    ) == 0o600
    assert stat.S_IMODE(
        (work / "formal_terminal.json").stat().st_mode
    ) == 0o600

    assert controller.main(["--freeze", str(freeze_path)]) == 2
    assert (work / "result.safe.json").read_bytes() == first_result
    assert (work / "formal_terminal.json").read_bytes() == first_terminal


def test_exclusive_writer_rejects_symlink_without_touching_target(
    posix_tmp_path: Path,
) -> None:
    target = posix_tmp_path / "target"
    target.write_bytes(b"preserve")
    link = posix_tmp_path / "artifact"
    link.symlink_to(target)

    with pytest.raises(FileExistsError):
        controller._exclusive_write_bytes(link, b"replace")
    assert target.read_bytes() == b"preserve"


def test_service_is_fixed_offline_one_shot_and_freeze_only() -> None:
    service = (
        PROJECT_ROOT
        / "manifests"
        / "meta-assumption-source-free-qualification-v2.service"
    ).read_text(encoding="ascii")

    assert (
        "WorkingDirectory=/home/erzhu419/uao_p2_20260729/"
        "source_free_qualification_v2/reconstruction_v2"
    ) in service
    assert f"{controller.FROZEN_PYTHON} -I -B" in service
    assert service.count("--freeze ") == 1
    assert "--work-root" not in service
    assert "CPUQuota=200%" in service
    assert "MemoryMax=1073741824" in service
    assert "TasksMax=16" in service
    assert "Restart=no" in service
    assert "RestrictAddressFamilies=none" in service
    assert "SystemCallArchitectures=native" in service
    assert "IPAddressDeny=any" in service
    assert "PrivateTmp=yes" in service
    assert "PrivateDevices=" not in service
    assert "ProtectSystem=strict" in service
    assert "ProtectHome=read-only" in service
    assert "InaccessiblePaths=/home/erzhu419/mine_code" in service
    assert (
        "ReadWritePaths=/home/erzhu419/uao_p2_20260729/"
        "source_free_qualification_v2/work"
    ) in service
    assert "ProtectControlGroups=yes" in service
    assert "ProtectKernelModules=yes" in service
    assert "ProtectKernelTunables=yes" in service
    assert "RestrictRealtime=yes" in service
    assert "RestrictSUIDSGID=yes" in service
    assert "LockPersonality=yes" in service
    assert "StandardInput=null" in service
    assert "StandardOutput=null" in service
    assert "StandardError=journal" in service
    assert (
        controller.LANDLOCK_WORK_ACCESS
        & (
            controller.LANDLOCK_ACCESS_FS_EXECUTE
            | controller.LANDLOCK_ACCESS_FS_MAKE_SOCK
            | controller.LANDLOCK_ACCESS_FS_MAKE_SYM
            | controller.LANDLOCK_ACCESS_FS_MAKE_FIFO
            | controller.LANDLOCK_ACCESS_FS_MAKE_BLOCK
            | controller.LANDLOCK_ACCESS_FS_MAKE_CHAR
        )
        == 0
    )
    exec_line = next(
        line for line in service.splitlines()
        if line.startswith("ExecStart=")
    )
    tokens = shlex.split(exec_line.removeprefix("ExecStart="))
    assert tokens[:2] == ["/usr/bin/env", "-i"]
    python_index = tokens.index(str(controller.FROZEN_PYTHON))
    observed_environment = dict(
        token.split("=", 1) for token in tokens[2:python_index]
    )
    assert observed_environment == controller._outer_environment(
        controller.FORMAL_ROOT
    )
    assert tokens[python_index + 1 : python_index + 3] == ["-I", "-B"]


def test_canary_service_is_production_isomorphic_without_formal_capability() -> None:
    formal = (
        PROJECT_ROOT
        / "manifests"
        / "meta-assumption-source-free-qualification-v2.service"
    ).read_text(encoding="ascii")
    canary = (
        PROJECT_ROOT
        / "manifests"
        / "meta-assumption-source-free-sandbox-canary-v2.service"
    ).read_text(encoding="ascii")

    security_properties = (
        "CPUQuota=200%",
        "MemoryMax=1073741824",
        "TasksMax=16",
        "KillMode=control-group",
        "Restart=no",
        "RestrictAddressFamilies=none",
        "SystemCallArchitectures=native",
        "IPAddressDeny=any",
        "NoNewPrivileges=yes",
        "PrivateTmp=yes",
        "ProtectSystem=strict",
        "ProtectHome=read-only",
        "InaccessiblePaths=/home/erzhu419/mine_code",
        "ProtectControlGroups=yes",
        "ProtectKernelModules=yes",
        "ProtectKernelTunables=yes",
        "RestrictRealtime=yes",
        "RestrictSUIDSGID=yes",
        "LockPersonality=yes",
        "StandardInput=null",
        "StandardOutput=null",
        "StandardError=journal",
    )
    for property_line in security_properties:
        assert formal.count(property_line) == 1
        assert canary.count(property_line) == 1
    assert "PrivateDevices=" not in formal
    assert "PrivateDevices=" not in canary
    assert "qualify_meta_assumption_sandbox_v2.py" in canary
    assert "--freeze" not in canary
    assert "source_free_qualification_v2" not in canary
    assert "source_free_sandbox_canary_v2" in canary
    assert controller.CANARY_RELATIVE in controller.REQUIRED_RELATIVE_FILES
    assert (
        controller.CANARY_SERVICE_RELATIVE
        in controller.REQUIRED_RELATIVE_FILES
    )
