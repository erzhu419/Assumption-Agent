from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import shlex
import site
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Mapping

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "run_quac_rjmc_source_free_qualification_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_quac_rjmc_source_free_qualification_v1_for_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
controller = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = controller
SPEC.loader.exec_module(controller)


@pytest.fixture
def posix_tmp_path() -> Path:
    path = Path(tempfile.mkdtemp(prefix="quac_rjmc_controller_", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _measurement(block_name: str) -> dict[str, object]:
    value: dict[str, object] = {
        "item_count": 8,
        "E0_total_utility": 0,
        "E1_total_utility": 8,
        "E1_minus_E0": 8,
        "topology_delta": {
            topology: 2 for topology in controller.TOPOLOGIES
        },
        "topology_raw_harm": {
            "pair_complement": 2,
            "redundancy_trap": 1,
            "retention_trap": 0,
            "null_shift": 0,
        },
        "topology_required_complete": {
            topology: True for topology in controller.TOPOLOGIES
        },
    }
    if block_name == "A_hold":
        value["promotion_passed"] = True
    else:
        value["structural_variant"] = (
            "extra_distractor_and_two_new_edges"
        )
    return value


def _receipt(*, behavior: str = "b" * 64) -> dict[str, object]:
    body = {
        "schema": (
            "qualify_quac_rjmc_source_free_v1_development_receipt"
        ),
        "version": "qualify_quac_rjmc_source_free_v1",
        "status": (
            "passed_nonformal_source_free_development_qualification"
        ),
        "formal_result": False,
        "architecture_decision_self_sha256": (
            controller.ARCHITECTURE_DECISION_SELF_SHA256
        ),
        "evaluator_version": "quac_rjmc_evaluator_v1",
        "fixture_provenance": "hand_authored_source_free_synthetic_only",
        "fixture_topologies": list(controller.TOPOLOGIES),
        "complete_state_count_for_three_candidates": 46,
        "component_jackknife_head_count": 5,
        "antisymmetric": True,
        "permutation_invariant": True,
        "RAW_structural_zero": True,
        "same_process_repeat_exact": True,
        "parameter_sha256": "a" * 64,
        "behavior_sha256": behavior,
        "A_hold": _measurement("A_hold"),
        "M_search": _measurement("M_search"),
        "qualification_weights_disposition": "discarded_at_process_exit",
        "QuAC_source_payload_access_count": 0,
        "prior_private_source_access_count": 0,
        "online_or_API_evaluation_count": 0,
    }
    return {
        **body,
        "receipt_self_sha256": controller.stable_hash(body),
    }


def _copy_frozen_tree(formal_root: Path) -> Path:
    project = formal_root / "reconstruction_v2"
    for relative in controller.REQUIRED_RELATIVE_FILES:
        source = PROJECT_ROOT / relative
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return project


def _freeze(
    formal_root: Path, *, python_executable: Path | None = None
) -> Path:
    project = _copy_frozen_tree(formal_root)
    python_path = (
        Path(sys.executable).resolve()
        if python_executable is None
        else python_executable
    )
    required = {
        relative: hashlib.sha256(
            (project / relative).read_bytes()
        ).hexdigest()
        for relative in controller.REQUIRED_RELATIVE_FILES
    }
    body = {
        "schema": controller.FREEZE_SCHEMA,
        "version": "v1",
        "study_id": controller.STUDY_ID,
        "formal_root": str(formal_root),
        "project_root": str(project),
        "work_root": str(formal_root / "work"),
        "python_executable": str(python_path),
        "python_executable_sha256": (
            controller.FROZEN_RUNTIME_IDENTITY[
                "python_executable_sha256"
            ]
        ),
        "python_executable_size_bytes": (
            controller.FROZEN_RUNTIME_IDENTITY[
                "python_executable_size_bytes"
            ]
        ),
        "python_executable_mode": (
            controller.FROZEN_RUNTIME_IDENTITY[
                "python_executable_mode"
            ]
        ),
        "python_version": (
            controller.FROZEN_RUNTIME_IDENTITY["python_version"]
        ),
        "torch_version": (
            controller.FROZEN_RUNTIME_IDENTITY["torch_version"]
        ),
        "numpy_version": (
            controller.FROZEN_RUNTIME_IDENTITY["numpy_version"]
        ),
        "implementation_commit": "1" * 40,
        "architecture_decision_self_sha256": (
            controller.ARCHITECTURE_DECISION_SELF_SHA256
        ),
        "source_custody_self_sha256": (
            controller.SOURCE_CUSTODY_SELF_SHA256
        ),
        "source_payload_access_count_before_qualification": 0,
        "online_or_API_evaluation_count_before_qualification": 0,
        "formal_attempt_limit": 1,
        "qualification_worker_count": 2,
        "worker_launch_policy": (
            "same_frozen_interpreter_sequential_distinct_processes"
        ),
        "worker_timeout_seconds": controller.WORKER_TIMEOUT_SECONDS,
        "required_file_sha256s": required,
    }
    value = {**body, "self_sha256": controller.stable_hash(body)}
    path = project / "manifests" / controller.FREEZE_FILENAME
    path.write_bytes(controller._canonical_bytes(value))
    return path


def _load(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def test_freeze_strictly_binds_roots_self_hash_and_all_required_files(
    posix_tmp_path: Path,
) -> None:
    formal_root = posix_tmp_path / "formal"
    freeze_path = _freeze(formal_root)
    loaded = controller.load_and_validate_freeze(
        freeze_path,
        expected_formal_root=formal_root,
        expected_python=Path(sys.executable).resolve(),
        enforce_invocation_path=False,
        expected_runtime_identity=controller.FROZEN_RUNTIME_IDENTITY,
    )
    assert loaded["_formal_root_path"] == formal_root
    assert set(loaded["required_file_sha256s"]) == set(
        controller.REQUIRED_RELATIVE_FILES
    )

    qualifier = formal_root / "reconstruction_v2" / controller.QUALIFIER_RELATIVE
    qualifier.write_bytes(qualifier.read_bytes() + b"\n")
    with pytest.raises(
        controller.QualificationControllerError,
        match="implementation hash drifted",
    ):
        controller.load_and_validate_freeze(
            freeze_path,
            expected_formal_root=formal_root,
            expected_python=Path(sys.executable).resolve(),
            enforce_invocation_path=False,
            expected_runtime_identity=controller.FROZEN_RUNTIME_IDENTITY,
        )


def test_controller_runs_two_ordered_distinct_workers_and_writes_pass_terminal(
    posix_tmp_path: Path,
) -> None:
    formal_root = posix_tmp_path / "formal"
    freeze_path = _freeze(formal_root)
    raw = controller._canonical_bytes(_receipt())
    calls: list[int] = []
    environments: list[Mapping[str, str]] = []

    def launcher(
        freeze: Mapping[str, object],
        environment: Mapping[str, str],
        ordinal: int,
    ) -> tuple[int, bytes]:
        assert freeze["_python_path"] == Path(sys.executable).resolve()
        assert "PATH" not in environment
        assert "HTTP_PROXY" not in environment
        assert environment["HF_HUB_OFFLINE"] == "1"
        assert environment["HOME"].endswith(f"/worker_{ordinal}/home")
        assert environment["HF_HOME"].endswith(
            f"/worker_{ordinal}/cache"
        )
        assert environment["TMPDIR"].endswith(
            f"/worker_{ordinal}/tmp"
        )
        environments.append(dict(environment))
        calls.append(len(calls) + 1)
        return 5000 + len(calls), raw

    terminal = controller.run_controller(
        freeze_path,
        expected_formal_root=formal_root,
        expected_python=Path(sys.executable).resolve(),
        enforce_invocation_path=False,
        launcher=launcher,
        expected_runtime_identity=controller.FROZEN_RUNTIME_IDENTITY,
    )
    assert calls == [1, 2]
    assert environments[0]["HOME"] != environments[1]["HOME"]
    assert environments[0]["HF_HOME"] != environments[1]["HF_HOME"]
    assert environments[0]["TMPDIR"] != environments[1]["TMPDIR"]
    assert (
        formal_root / "work" / "sandbox" / "worker_1" / "home"
    ).is_dir()
    assert (
        formal_root / "work" / "sandbox" / "worker_2" / "home"
    ).is_dir()
    assert terminal["status"] == controller.PASS_STATUS
    assert terminal["qualification_passed"] is True
    work = formal_root / "work"
    assert (work / "worker_1.receipt.json").read_bytes() == raw
    assert (work / "worker_2.receipt.json").read_bytes() == raw
    for filename in controller.OUTPUT_FILENAMES:
        path = work / filename
        assert path.is_file()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    result = _load(work / "result.safe.json")
    assert result["same_host_two_process_exact"] is True
    assert result["same_host_two_process_receipt_byte_exact"] is True
    assert result["worker_pids_distinct"] is True
    assert result["QuAC_source_payload_access_count"] == 0
    assert result["online_or_API_evaluation_count"] == 0

    with pytest.raises(controller.OneShotRefusal, match="not pristine"):
        controller.run_controller(
            freeze_path,
            expected_formal_root=formal_root,
            expected_python=Path(sys.executable).resolve(),
            enforce_invocation_path=False,
            launcher=launcher,
            expected_runtime_identity=controller.FROZEN_RUNTIME_IDENTITY,
        )
    assert calls == [1, 2]


def test_two_process_semantic_drift_writes_single_stop_without_retry(
    posix_tmp_path: Path,
) -> None:
    formal_root = posix_tmp_path / "formal"
    freeze_path = _freeze(formal_root)
    outputs = (
        controller._canonical_bytes(_receipt(behavior="b" * 64)),
        controller._canonical_bytes(_receipt(behavior="c" * 64)),
    )
    calls = 0

    def launcher(
        _freeze_value: Mapping[str, object],
        _environment: Mapping[str, str],
        _ordinal: int,
    ) -> tuple[int, bytes]:
        nonlocal calls
        raw = outputs[calls]
        calls += 1
        return 6000 + calls, raw

    terminal = controller.run_controller(
        freeze_path,
        expected_formal_root=formal_root,
        expected_python=Path(sys.executable).resolve(),
        enforce_invocation_path=False,
        launcher=launcher,
        expected_runtime_identity=controller.FROZEN_RUNTIME_IDENTITY,
    )
    assert calls == 2
    assert terminal["status"] == controller.STOP_STATUS
    assert terminal["qualification_passed"] is False
    assert (
        terminal["next_action"]
        == "close_RJMC_without_downloading_QuAC"
    )
    result = _load(formal_root / "work" / "result.safe.json")
    assert result["status"] == controller.STOP_STATUS
    assert result["retry_replay_resample_or_repair_count"] == 0
    assert result["completed_worker_receipt_count"] == 2
    assert result["same_host_two_process_exact"] is False
    assert result["same_host_two_process_receipt_byte_exact"] is False
    assert (formal_root / "work" / "formal_terminal.json").is_file()


def test_worker_stdout_is_only_one_canonical_semantic_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    monkeypatch.setattr(
        controller,
        "load_and_validate_freeze",
        lambda *_args, **_kwargs: {
            "_project_root_path": tmp_path,
        },
    )
    monkeypatch.setattr(
        controller,
        "_load_qualifier",
        lambda _root: SimpleNamespace(qualify=lambda: receipt),
    )
    output = io.BytesIO()
    assert (
        controller.worker_main(
            tmp_path / "freeze.json",
            ordinal=1,
            stdout_buffer=output,
        )
        == 0
    )
    assert output.getvalue() == controller._canonical_bytes(receipt)


def test_preflight_validates_without_creating_formal_artifacts(
    posix_tmp_path: Path,
) -> None:
    formal_root = posix_tmp_path / "formal"
    freeze_path = _freeze(formal_root)
    work = formal_root / "work"
    assert not work.exists()
    receipt = controller.run_preflight(
        freeze_path,
        expected_formal_root=formal_root,
        expected_python=Path(sys.executable).resolve(),
        enforce_invocation_path=False,
        expected_runtime_identity=controller.FROZEN_RUNTIME_IDENTITY,
    )
    assert receipt["status"] == "PASS_RJMC_SOURCE_FREE_PREFLIGHT"
    assert receipt["formal_attempt_created"] is False
    assert set(receipt) == {
        "schema",
        "version",
        "study_id",
        "status",
        "formal_attempt_created",
        "freeze_file_sha256",
        "freeze_self_sha256",
        "QuAC_source_payload_access_count",
        "online_or_API_evaluation_count",
        "retry_replay_resample_or_repair_count",
        "preflight_self_sha256",
    }
    assert not work.exists()

    work.mkdir(mode=0o700)
    second = controller.run_preflight(
        freeze_path,
        expected_formal_root=formal_root,
        expected_python=Path(sys.executable).resolve(),
        enforce_invocation_path=False,
        expected_runtime_identity=controller.FROZEN_RUNTIME_IDENTITY,
    )
    assert second == receipt
    assert list(work.iterdir()) == []


def test_formal_bootstrap_failure_writes_one_safe_stop_before_attempt(
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
    monkeypatch.setattr(
        controller, "FROZEN_PYTHON", Path(sys.executable).resolve()
    )

    assert controller.main(["--freeze", str(freeze_path)]) == 2
    work = formal_root / "work"
    assert not (work / "attempt.json").exists()
    result_path = work / "result.safe.json"
    terminal_path = work / "formal_terminal.json"
    first_result = result_path.read_bytes()
    first_terminal = terminal_path.read_bytes()
    result = _load(result_path)
    terminal = _load(terminal_path)
    assert set(result) == {
        "schema",
        "version",
        "study_id",
        "status",
        "formal_result",
        "qualification_passed",
        "attempt_created",
        "failure_code",
        "QuAC_source_payload_access_count",
        "online_or_API_evaluation_count",
        "retry_replay_resample_or_repair_count",
        "next_action",
        "result_self_sha256",
    }
    assert set(terminal) == {
        "schema",
        "version",
        "study_id",
        "status",
        "formal_complete",
        "qualification_passed",
        "attempt_created",
        "result_safe_file_sha256",
        "result_safe_self_sha256",
        "QuAC_source_payload_access_count",
        "online_or_API_evaluation_count",
        "retry_replay_resample_or_repair_count",
        "next_action",
        "terminal_self_sha256",
    }
    assert result["status"] == controller.STOP_STATUS
    assert result["attempt_created"] is False
    assert result["QuAC_source_payload_access_count"] == 0
    assert result["online_or_API_evaluation_count"] == 0
    assert result["retry_replay_resample_or_repair_count"] == 0
    assert (
        result["next_action"]
        == "close_RJMC_without_downloading_QuAC"
    )
    assert terminal["status"] == controller.STOP_STATUS
    assert terminal["attempt_created"] is False
    assert stat.S_IMODE(result_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(terminal_path.stat().st_mode) == 0o600

    assert controller.main(["--freeze", str(freeze_path)]) == 2
    assert result_path.read_bytes() == first_result
    assert terminal_path.read_bytes() == first_terminal


def _isolated_test_python(root: Path) -> Path:
    runtime = root / "runtime"
    executable = runtime / "bin" / "python"
    executable.parent.mkdir(parents=True)
    shutil.copyfile(Path(sys.executable).resolve(), executable)
    executable.chmod(0o755)
    (runtime / "pyvenv.cfg").write_text(
        "home = /usr/bin\n"
        "include-system-site-packages = false\n"
        "version = 3.10.12\n",
        encoding="ascii",
    )
    packages = runtime / "lib" / "python3.10" / "site-packages"
    packages.mkdir(parents=True)
    (packages / "frozen-local-packages.pth").write_text(
        str(Path(site.getusersitepackages()).resolve()) + "\n",
        encoding="ascii",
    )
    return executable


def test_actual_isolated_worker_subprocess_imports_torch_and_emits_receipt(
    posix_tmp_path: Path,
) -> None:
    python_executable = _isolated_test_python(posix_tmp_path)
    formal_root = posix_tmp_path / "formal"
    freeze_path = _freeze(
        formal_root, python_executable=python_executable
    )
    work = formal_root / "work"
    work.mkdir(mode=0o700)
    environment = controller._worker_environment(
        project_root=formal_root / "reconstruction_v2",
        work_root=work,
        ordinal=1,
    )
    completed = subprocess.run(
        (
            str(python_executable),
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
        completed.stdout, description="actual isolated worker receipt"
    )
    controller.validate_semantic_receipt(value)
    assert completed.stdout == controller._canonical_bytes(value)
    assert not (work / "attempt.json").exists()
    assert not (work / "result.safe.json").exists()
    assert not (work / "formal_terminal.json").exists()


def test_service_has_fixed_root_interpreter_offline_limits_and_freeze_only() -> None:
    service = (
        PROJECT_ROOT
        / "manifests"
        / "quac_rjmc_source_free_qualification_v1.service"
    ).read_text(encoding="ascii")
    assert (
        "WorkingDirectory=/home/erzhu419/quac_rjmc_20260728/"
        "source_free_qualification_v1/reconstruction_v2"
    ) in service
    assert (
        "/home/erzhu419/p19_runtime_assets_20260723/"
        "typed_venv/bin/python -I -B"
    ) in service
    assert service.count("--freeze ") == 1
    assert "--work-root" not in service
    assert "CPUQuota=200%" in service
    assert "MemoryMax=4294967296" in service
    assert "Restart=no" in service
    assert "RestrictAddressFamilies=AF_UNIX" in service
    assert "IPAddressDeny=any" in service
    assert "PrivateTmp=yes" in service
    assert "HF_HUB_OFFLINE=1" in service
    assert "TRANSFORMERS_OFFLINE=1" in service
    exec_line = next(
        line for line in service.splitlines() if line.startswith("ExecStart=")
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
