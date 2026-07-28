from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile

import pytest

from assumption_agent.benchmarks import quac_p1_source_qualification_v1 as q
from test_quac_p1_source_qualification_v1 import (
    TEST_QUOTAS,
    _fixture,
    _raw,
)


LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    LOCAL_PROJECT_ROOT / "scripts/run_quac_p1_source_qualification_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_quac_p1_source_qualification_v1_for_test",
    RUNNER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


@pytest.fixture
def linux_root() -> Path:
    # The project-wide pytest temp is on DrvFS in this workspace, which cannot
    # prove 0600/0700.  Formal execution and this fixture both use Linux fs.
    path = Path(tempfile.mkdtemp(prefix="quac-p1-runner-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture_sources(
    source_root: Path,
) -> tuple[Path, Path, q.QualificationContract, dict[str, dict[str, object]]]:
    train, dev = _fixture()
    train_raw = _raw(train)
    dev_raw = _raw(dev)
    source_root.mkdir(mode=0o700, parents=True)
    train_path = source_root / "train_v0.2.json"
    dev_path = source_root / "val_v0.2.json"
    train_path.write_bytes(train_raw)
    dev_path.write_bytes(dev_raw)
    os.chmod(train_path, 0o600)
    os.chmod(dev_path, 0o600)
    contract = q.QualificationContract(
        train=q.SourceFileContract(
            len(train_raw), hashlib.sha256(train_raw).hexdigest()
        ),
        dev=q.SourceFileContract(
            len(dev_raw), hashlib.sha256(dev_raw).hexdigest()
        ),
        quotas=TEST_QUOTAS,
    )
    source_contract = {
        "train": {
            "path": str(train_path),
            "size_bytes": len(train_raw),
            "sha256": hashlib.sha256(train_raw).hexdigest(),
            "mode_octal": "0600",
        },
        "dev": {
            "path": str(dev_path),
            "size_bytes": len(dev_raw),
            "sha256": hashlib.sha256(dev_raw).hexdigest(),
            "mode_octal": "0600",
        },
    }
    return train_path, dev_path, contract, source_contract


def _deploy_required_files(project_root: Path) -> dict[str, str]:
    for relative in runner.REQUIRED_FILE_RELATIVE_PATHS:
        source = LOCAL_PROJECT_ROOT / relative
        target = project_root / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return {
        relative: _sha256(project_root / relative)
        for relative in runner.REQUIRED_FILE_RELATIVE_PATHS
    }


def _formal_fixture(root: Path) -> dict[str, object]:
    formal_root = root / "source_qualification_v1"
    project_root = formal_root / "reconstruction_v2"
    source_root = root / "official_source_v1"
    required_hashes = _deploy_required_files(project_root)
    (
        train_path,
        dev_path,
        contract,
        source_contract,
    ) = _write_fixture_sources(source_root)

    installed_unit = root / "user-systemd" / runner.UNIT_NAME
    installed_unit.parent.mkdir(mode=0o700, parents=True)
    installed_unit.symlink_to(project_root / runner.UNIT_RELATIVE_PATH)

    environment = dict(os.environ)
    python_path = Path(sys.executable)
    freeze_body = {
        "schema": runner.FREEZE_SCHEMA,
        "version": "v1",
        "study_id": runner.STUDY_ID,
        "formal_attempt_limit": 1,
        "formal_root": str(formal_root),
        "project_root": str(project_root),
        "source_root": str(source_root),
        "work_root": str(formal_root / "work"),
        "implementation_commit": "0" * 40,
        "architecture_decision_self_sha256": (
            runner.ARCHITECTURE_DECISION_SELF_SHA256
        ),
        "source_custody_self_sha256": (
            runner.SOURCE_CUSTODY_SELF_SHA256
        ),
        "source_free_qualification_result_self_sha256": (
            runner.SOURCE_FREE_RESULT_SELF_SHA256
        ),
        "source_free_qualification_result_file_sha256": (
            runner.SOURCE_FREE_RESULT_FILE_SHA256
        ),
        "source_free_qualification_freeze_self_sha256": (
            runner.SOURCE_FREE_FREEZE_SELF_SHA256
        ),
        "python_identity": runner._actual_python_identity(python_path),
        "environment": environment,
        "source_contract": source_contract,
        "unit_name": runner.UNIT_NAME,
        "unit_source_path": str(
            project_root / runner.UNIT_RELATIVE_PATH
        ),
        "unit_installed_path": str(installed_unit),
        "required_file_sha256s": required_hashes,
        "source_payload_access_count_before_qualification": 0,
        "online_or_API_evaluation_count_before_qualification": 0,
        "retry_replay_resample_or_repair_count_before_qualification": 0,
    }
    freeze = {
        **freeze_body,
        "self_sha256": runner._stable_hash(freeze_body),
    }
    freeze_path = project_root / "manifests" / runner.FREEZE_FILENAME
    freeze_path.write_bytes(runner._canonical_bytes(freeze))
    os.chmod(freeze_path, 0o600)
    return {
        "formal_root": formal_root,
        "project_root": project_root,
        "source_root": source_root,
        "train_path": train_path,
        "dev_path": dev_path,
        "contract": contract,
        "source_contract": source_contract,
        "installed_unit": installed_unit,
        "environment": environment,
        "python_path": python_path,
        "freeze_path": freeze_path,
    }


def _execution_kwargs(fixture: dict[str, object]) -> dict[str, object]:
    return {
        "expected_formal_root": fixture["formal_root"],
        "expected_python": fixture["python_path"],
        "expected_environment": fixture["environment"],
        "expected_source_contract": fixture["source_contract"],
        "expected_installed_unit_path": fixture["installed_unit"],
        "enforce_invocation_path": False,
        "contract_override": fixture["contract"],
    }


def test_preflight_is_source_free_then_formal_attempt_is_atomic_and_one_shot(
    linux_root: Path,
) -> None:
    fixture = _formal_fixture(linux_root)
    kwargs = _execution_kwargs(fixture)
    preflight = runner.run_preflight(fixture["freeze_path"], **kwargs)
    assert preflight["status"] == runner.PREFLIGHT_PASS_STATUS
    assert preflight["source_payload_access_count"] == 0
    assert not (fixture["formal_root"] / "work").exists()

    terminal = runner.run_once(fixture["freeze_path"], **kwargs)
    assert terminal["passed"] is True
    work = fixture["formal_root"] / "work"
    attempt_path = work / runner.ATTEMPT_NAME
    result_path = work / runner.RESULT_NAME
    terminal_path = work / runner.TERMINAL_NAME
    assert all(path.is_file() for path in (attempt_path, result_path, terminal_path))
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in (attempt_path, result_path, terminal_path)
    )
    assert not tuple(work.glob(".*.tmp.*"))
    result_text = result_path.read_text(encoding="ascii")
    assert "private_" not in result_text
    assert "assignment_witness_output_count" in result_text

    with pytest.raises(runner.OneShotConsumed):
        runner.run_once(fixture["freeze_path"], **kwargs)


def test_same_size_source_identity_failure_writes_fixed_safe_stop(
    linux_root: Path,
) -> None:
    fixture = _formal_fixture(linux_root)
    train_path = fixture["train_path"]
    raw = bytearray(train_path.read_bytes())
    raw[-2] = ord(" ") if raw[-2] != ord(" ") else ord("\t")
    train_path.write_bytes(raw)
    os.chmod(train_path, 0o600)

    terminal = runner.run_once(
        fixture["freeze_path"],
        **_execution_kwargs(fixture),
    )
    assert terminal["status"] == runner.SOURCE_STOP_STATUS
    assert terminal["passed"] is False
    work = fixture["formal_root"] / "work"
    result = json.loads(
        (work / runner.RESULT_NAME).read_text(encoding="ascii")
    )
    assert result["source_identity_pass"] is False
    assert result["source_aggregates"] == {}
    assert result["capacity_flow"]["assignment_witness_output_count"] == 0
    assert result["activity_counts"] == {
        "selection": 0,
        "model": 0,
        "action": 0,
        "score": 0,
        "online_or_API_evaluation": 0,
    }
    assert (work / runner.ATTEMPT_NAME).is_file()


def test_bootstrap_failure_writes_complete_safe_stop_without_attempt(
    linux_root: Path,
) -> None:
    fixture = _formal_fixture(linux_root)
    core_path = (
        fixture["project_root"] / runner.CORE_RELATIVE_PATH
    )
    core_path.write_bytes(core_path.read_bytes() + b"\n")
    with pytest.raises(runner.QuacP1OneShotError):
        runner.run_once(
            fixture["freeze_path"],
            **_execution_kwargs(fixture),
        )
    terminal = runner._write_bootstrap_stop(
        fixture["formal_root"],
        freeze_path=fixture["freeze_path"],
    )
    work = fixture["formal_root"] / "work"
    assert terminal["status"] == runner.BOOTSTRAP_STOP_STATUS
    assert terminal["formal_complete"] is True
    assert not (work / runner.ATTEMPT_NAME).exists()
    assert (work / runner.RESULT_NAME).stat().st_size > 0
    assert (work / runner.TERMINAL_NAME).stat().st_size > 0
    bootstrap_result = json.loads(
        (work / runner.RESULT_NAME).read_text(encoding="ascii")
    )
    assert bootstrap_result["activity_counts"]["selection"] == 0
    assert bootstrap_result["activity_counts"]["score"] == 0


def test_safe_whitelist_rejects_extra_keys_and_flow_drift() -> None:
    train, dev = _fixture()
    aggregate = q.qualify_decoded_sources(
        train,
        dev,
        quotas=TEST_QUOTAS,
    )
    aggregate["private_item"] = "must not serialize"
    with pytest.raises(
        runner.QuacP1OneShotError,
        match="fields drifted",
    ):
        runner._validate_safe_qualification(
            aggregate,
            quotas=TEST_QUOTAS,
        )

    aggregate.pop("private_item")
    aggregate["capacity_flow"]["slot_slack"]["A_hold"]["FOLLOW"] = 1
    with pytest.raises(
        runner.QuacP1OneShotError,
        match="arithmetic drifted|totals drifted",
    ):
        runner._validate_safe_qualification(
            aggregate,
            quotas=TEST_QUOTAS,
        )


def test_unit_uses_the_exact_freeze_and_formal_mode() -> None:
    unit = (
        LOCAL_PROJECT_ROOT / runner.UNIT_RELATIVE_PATH
    ).read_text(encoding="utf-8")
    assert f"--freeze {runner.FREEZE_PATH}" in unit
    assert "--formal" in unit
    assert "Restart=no" in unit


def test_completed_scientific_stop_is_a_successful_service_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_freeze = Path("/tmp/fixed-quac-freeze-for-exit-test.json")
    monkeypatch.setattr(runner, "FREEZE_PATH", fixed_freeze)
    monkeypatch.setattr(
        runner,
        "run_once",
        lambda _freeze: {"passed": False, "status": runner.CAPACITY_STOP_STATUS},
    )
    assert runner.main(
        ["--freeze", str(fixed_freeze), "--formal"]
    ) == 0
