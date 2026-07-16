from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "launch_tmux_detached_controls_once.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location(
    "tmux_detached_controls_once", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def _valid_command(tmp_path: Path, output_root: Path) -> list[str]:
    command = [
        str(Path(sys.executable).resolve()),
        "-B",
        "-m",
        launcher.RUNNER_MODULE,
    ]
    for flag in launcher.REQUIRED_RUNNER_FLAGS:
        command.extend(
            [flag, str(output_root if flag == "--output-root" else tmp_path)]
        )
    return command


def test_command_boundary_accepts_only_frozen_controls_module(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "absent-output"
    command = _valid_command(tmp_path, output_root)
    assert launcher._safe_controls_command(
        command,
        output_root,
        working_directory=tmp_path,
    ) == command

    wrong_module = list(command)
    wrong_module[3] = "replication_runtime.financial_sec13f_contract_v2.runner"
    with pytest.raises(launcher.base.LaunchError, match="only the frozen"):
        launcher._safe_controls_command(
            wrong_module,
            output_root,
            working_directory=tmp_path,
        )

    with pytest.raises(launcher.base.LaunchError, match="retry"):
        launcher._safe_controls_command(
            [*command, "--retry-failed"],
            output_root,
            working_directory=tmp_path,
        )
    with pytest.raises(launcher.base.LaunchError, match="recovery"):
        launcher._safe_controls_command(
            [*command, "--recover-only"],
            output_root,
            working_directory=tmp_path,
        )

    duplicate_output = [*command, "--output-root", str(output_root)]
    with pytest.raises(launcher.base.LaunchError, match="exactly one"):
        launcher._safe_controls_command(
            duplicate_output,
            output_root,
            working_directory=tmp_path,
        )


def test_freeze_identity_binds_both_launcher_sources(
    tmp_path: Path,
) -> None:
    project = SCRIPT.parents[1]
    rows = [
        {
            "role": role,
            "relative_path": source.relative_to(project).as_posix(),
            "file_sha256": launcher.base._sha256_file(source),
            "committed_at_git_commit": "a" * 40,
        }
        for role, source in (
            ("controls_launcher", SCRIPT),
            ("base_launcher", Path(launcher.base.__file__).resolve(strict=True)),
        )
    ]
    body = {
        "launcher_source_closure": {
            "files": rows,
            "file_count": 2,
            "file_set_hash": launcher.base._payload_hash(rows),
        }
    }
    freeze = {**body, "manifest_hash": launcher.base._payload_hash(body)}
    freeze_path = tmp_path / "controls.freeze.json"
    freeze_path.write_text(json.dumps(freeze) + "\n", encoding="utf-8")
    output_root = tmp_path / "absent-output"
    command = _valid_command(tmp_path, output_root)
    index = command.index("--controls-freeze")
    command[index + 1] = str(freeze_path)
    identity = launcher._controls_freeze_identity(
        command,
        working_directory=project,
        launcher_source=SCRIPT,
    )
    assert identity["controls_execution_freeze_hash"] == freeze["manifest_hash"]
    assert identity["controls_launcher_source_sha256"] == rows[0]["file_sha256"]
    assert identity["base_launcher_source_sha256"] == rows[1]["file_sha256"]


def test_sanitized_fake_service_is_one_shot_secret_free_and_controls_aware(
    tmp_path: Path,
) -> None:
    """Exercise the wrapper boundary without tmux, protected data, or a model."""

    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    output_root = tmp_path / "controls-output"
    audit_path = control / "fake-environment.json"
    sentinel_name = "DETACHED_CONTROLS_SECRET_SENTINEL"
    sentinel_value = "fake-secret-that-must-not-cross-the-boundary"
    fake_code = (
        "import json,os,pathlib,sys; "
        "out=pathlib.Path(sys.argv[1]); audit=pathlib.Path(sys.argv[2]); "
        "out.mkdir(); "
        "(out/'controls.report.json').write_text('{}\\n', encoding='utf-8'); "
        "audit.write_text(json.dumps({"
        "'sentinel_present': 'DETACHED_CONTROLS_SECRET_SENTINEL' in os.environ,"
        "'secret_name_present': any("
        "('API_KEY' in k or 'AUTHORIZATION' in k or 'CODEX_THREAD' in k) "
        "for k in os.environ)})+'\\n', encoding='utf-8')"
    )
    command = [
        str(Path(sys.executable).resolve()),
        "-B",
        "-c",
        fake_code,
        str(output_root),
        str(audit_path),
    ]
    command_hash = launcher.base._payload_hash(command)
    run_id = "fake-controls-once"
    unit_name = "tmux:fake-controls:controls-once"
    launcher.base._write_new_json(
        control / "launch.requested.json",
        {
            "receipt_version": launcher.RECEIPT_VERSION,
            "launcher_version": launcher.LAUNCHER_VERSION,
            "requested_at_utc": launcher.base._utc_now(),
            "run_id": run_id,
            "unit_name": unit_name,
            "working_directory": str(tmp_path),
            "output_root": str(output_root),
            "output_root_preexisting": False,
            "command": command,
            "command_hash": command_hash,
            "stdout_stderr_log": str(control / "runner.stdout-stderr.log"),
            "tmux_socket_name": "fake-controls",
            "tmux_session_name": "controls-once",
            "launch_attempt": 1,
            "automatic_restart_authorized": False,
            "retry_authorized": False,
            "recovery_authorized": False,
            "replay_authorized": False,
        },
    )
    service_command = [
        *launcher.base._sanitized_service_environment(),
        str(Path(sys.executable).resolve()),
        str(SCRIPT),
        "_service",
        "--control-directory",
        str(control),
        "--run-id",
        run_id,
        "--unit-name",
        unit_name,
        "--working-directory",
        str(tmp_path),
        "--output-root",
        str(output_root),
        "--command-hash",
        command_hash,
        "--",
        *command,
    ]
    completed = subprocess.run(
        service_command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env={**os.environ, sentinel_name: sentinel_value},
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(audit_path.read_text(encoding="utf-8")) == {
        "sentinel_present": False,
        "secret_name_present": False,
    }
    exited = launcher.base._read_receipt(control / "service.exited.json")
    assert exited["runner_started"] is True
    assert exited["runner_returncode"] == 0
    assert exited["controls_report"]["present"] is True
    assert exited["controls_failure_receipt"]["present"] is False
    assert exited["automatic_restart_authorized"] is False
    assert exited["retry_authorized"] is False
    assert exited["recovery_authorized"] is False
    assert exited["replay_authorized"] is False

    receipt_text = "\n".join(
        path.read_text(encoding="utf-8") for path in control.glob("*.json")
    )
    assert sentinel_value not in receipt_text
    assert not (output_root / "controls.failure.json").exists()
    status = launcher.status(control)
    assert status["phase"] == "completed"
    assert status["controls_report"]["present"] is True
