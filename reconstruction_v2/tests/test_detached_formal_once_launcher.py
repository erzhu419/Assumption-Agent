from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "launch_detached_formal_once.py"
)
SPEC = importlib.util.spec_from_file_location("detached_formal_once", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def _user_systemd_available() -> bool:
    completed = subprocess.run(
        ["systemctl", "--user", "is-system-running"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return completed.stdout.strip() in {"running", "degraded"}


def test_command_boundary_rejects_non_python_executable(tmp_path: Path) -> None:
    output_root = tmp_path / "absent-output"
    command = [
        "/bin/true",
        "-B",
        "-m",
        launcher.RUNNER_MODULE,
    ]
    for flag in launcher.REQUIRED_RUNNER_FLAGS:
        command.extend([flag, str(output_root if flag == "--output-root" else tmp_path)])
    with pytest.raises(launcher.LaunchError):
        launcher._safe_runner_command(
            command,
            output_root,
            working_directory=tmp_path,
        )


@pytest.mark.skipif(
    not _user_systemd_available(), reason="systemd user manager is unavailable"
)
def test_sanitized_detached_wrapper_is_one_shot_and_secret_free(
    tmp_path: Path,
) -> None:
    """Exercise the real wrapper/unit boundary without a model or run root."""

    run_id = f"fake-{uuid.uuid4().hex[:12]}"
    unit = f"assumption-formal-{run_id}.service"
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    output_root = tmp_path / "formal-output-must-remain-absent"
    log = control / "runner.stdout-stderr.log"
    sentinel_name = "DETACHED_FORMAL_SECRET_SENTINEL"
    sentinel_value = "fake-value-that-must-not-be-logged"
    fake_code = (
        "import json,os,time; "
        "print(json.dumps({"
        "'sentinel_present': 'DETACHED_FORMAL_SECRET_SENTINEL' in os.environ,"
        "'secret_name_present': any("
        "('API_KEY' in k or 'AUTHORIZATION' in k or 'CODEX_THREAD' in k) "
        "for k in os.environ)}), flush=True); time.sleep(1)"
    )
    command = [str(Path(os.sys.executable).resolve()), "-B", "-c", fake_code]
    command_hash = launcher._payload_hash(command)
    launcher._write_new_json(
        control / "launch.requested.json",
        {
            "receipt_version": launcher.RECEIPT_VERSION,
            "launcher_version": launcher.LAUNCHER_VERSION,
            "requested_at_utc": launcher._utc_now(),
            "run_id": run_id,
            "unit_name": unit,
            "working_directory": str(tmp_path),
            "output_root": str(output_root),
            "output_root_preexisting": False,
            "command": command,
            "command_hash": command_hash,
            "stdout_stderr_log": str(log),
            "launch_attempt": 1,
            "automatic_restart_authorized": False,
            "retry_authorized": False,
        },
    )
    service = [
        *launcher._sanitized_service_environment(),
        str(Path(os.sys.executable).resolve()),
        str(SCRIPT),
        "_service",
        "--control-directory",
        str(control),
        "--run-id",
        run_id,
        "--unit-name",
        unit,
        "--working-directory",
        str(tmp_path),
        "--output-root",
        str(output_root),
        "--command-hash",
        command_hash,
        "--",
        *command,
    ]
    try:
        dispatched = subprocess.run(
            [
                "systemd-run",
                "--user",
                "--quiet",
                "--unit",
                unit,
                "--collect",
                "--property=Type=exec",
                "--property=Restart=no",
                "--property=KillMode=control-group",
                "--property=UMask=0077",
                f"--property=WorkingDirectory={tmp_path}",
                f"--property=StandardOutput=append:{log}",
                f"--property=StandardError=append:{log}",
                f"--setenv={sentinel_name}={sentinel_value}",
                *service,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        assert dispatched.returncode == 0
        identity = launcher._systemd_identity(unit)
        assert identity["Type"] == "exec"
        assert identity["Restart"] == "no"
        assert identity["KillMode"] == "control-group"
        main_pid = int(identity["MainPID"])
        environment_names = {
            row.split(b"=", 1)[0].decode("ascii")
            for row in Path(f"/proc/{main_pid}/environ").read_bytes().split(b"\0")
            if b"=" in row
        }
        assert sentinel_name not in environment_names
        assert not any("API_KEY" in name for name in environment_names)
        assert "CODEX_THREAD_ID" not in environment_names

        deadline = time.monotonic() + 10
        while not (control / "service.exited.json").is_file():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        exited = launcher._read_receipt(control / "service.exited.json")
        assert exited["runner_started"] is True
        assert exited["runner_returncode"] == 0
        assert exited["retry_authorized"] is False
        assert exited["automatic_restart_authorized"] is False
        assert not os.path.lexists(output_root)
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        assert rows == [
            {"sentinel_present": False, "secret_name_present": False}
        ]
        assert sentinel_value not in log.read_text()
    finally:
        subprocess.run(
            ["systemctl", "--user", "stop", unit],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
