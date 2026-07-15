#!/usr/bin/env python3
from __future__ import annotations

"""Launch one SEC-13F formal runner in an isolated, sanitized tmux server."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Sequence

import launch_detached_formal_once as base


LAUNCHER_VERSION = "tmux_detached_sec13f_contract_formal_once_launcher_v1"
RECEIPT_VERSION = "tmux_detached_sec13f_contract_formal_once_receipt_v1"


def _tmux_identity(socket_name: str, session_name: str) -> dict[str, Any]:
    query = subprocess.run(
        [
            "/usr/bin/tmux",
            "-L",
            socket_name,
            "list-panes",
            "-t",
            session_name,
            "-F",
            "#{pane_pid} #{pane_dead} #{pane_dead_status}",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    row = query.stdout.strip().split()
    return {
        "backend": "isolated_tmux_server_v1",
        "query_returncode": query.returncode,
        "session_exists": query.returncode == 0,
        "pane_pid": int(row[0]) if row and row[0].isdigit() else 0,
        "pane_dead": row[1] == "1" if len(row) > 1 else None,
        "pane_dead_status": (
            int(row[2]) if len(row) > 2 and row[2].isdigit() else None
        ),
        "automatic_restart_authorized": False,
    }


def _create_log(path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)


def launch(args: argparse.Namespace) -> int:
    if not base.RUN_ID_PATTERN.fullmatch(args.run_id):
        raise base.LaunchError("run id is malformed")
    workdir = base._normalized(args.working_directory)
    if workdir.is_symlink() or not workdir.is_dir():
        raise base.LaunchError("working directory is unavailable")
    expected_runner = workdir / Path(*base.RUNNER_MODULE.split(".")).with_suffix(
        ".py"
    )
    if expected_runner.is_symlink() or not expected_runner.is_file():
        raise base.LaunchError("SEC contract runner source is unavailable")
    output_root = base._normalized(args.output_root, relative_to=workdir)
    if os.path.lexists(output_root):
        raise base.LaunchError("formal output root already exists")
    command = base._safe_runner_command(
        args.command, output_root, working_directory=workdir
    )
    if not Path("/usr/bin/tmux").is_file():
        raise base.LaunchError("tmux is unavailable")

    control_base = base._normalized(args.control_base, relative_to=workdir)
    if control_base == output_root or output_root in control_base.parents:
        raise base.LaunchError("control base is inside the formal output root")
    control_base.mkdir(parents=True, exist_ok=True, mode=0o700)
    if control_base.is_symlink() or not control_base.is_dir():
        raise base.LaunchError("control base is not a regular directory")
    os.chmod(control_base, 0o700)
    run_dir = control_base / "runs" / args.run_id
    run_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        run_dir.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise base.LaunchError("run id has already been consumed") from exc

    reservation_key = hashlib.sha256(os.fsencode(str(output_root))).hexdigest()
    reservation = control_base / "output-reservations" / reservation_key
    reservation.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        reservation.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise base.LaunchError("formal output path is already reserved") from exc

    socket_name = "assumption-formal-" + hashlib.sha256(
        args.run_id.encode("ascii")
    ).hexdigest()[:16]
    session_name = "formal-once"
    unit_name = f"tmux:{socket_name}:{session_name}"
    log_path = run_dir / "runner.stdout-stderr.log"
    _create_log(log_path)
    launcher_source = Path(__file__).resolve(strict=True)
    service_source = Path(base.__file__).resolve(strict=True)
    command_hash = base._payload_hash(command)
    common = {
        "receipt_version": RECEIPT_VERSION,
        "launcher_version": LAUNCHER_VERSION,
        "run_id": args.run_id,
        "unit_name": unit_name,
        "detachment_backend": "isolated_tmux_server_v1",
        "tmux_socket_name": socket_name,
        "tmux_session_name": session_name,
        "output_root": str(output_root),
        "output_root_hash": base._payload_hash(str(output_root)),
        "command_hash": command_hash,
        "launch_attempt": 1,
        "automatic_restart_authorized": False,
        "retry_authorized": False,
        "recover_only_authorized": False,
    }
    base._write_new_json(
        reservation / "owner.json",
        {
            **common,
            "reserved_at_utc": base._utc_now(),
            "run_control_directory": str(run_dir),
            "reservation_is_permanent": True,
        },
    )
    base._write_new_json(
        run_dir / "launch.requested.json",
        {
            **common,
            "requested_at_utc": base._utc_now(),
            "working_directory": str(workdir),
            "output_root_preexisting": False,
            "output_reservation": str(reservation),
            "command": command,
            "stdout_stderr_log": str(log_path),
            "launcher_source": str(launcher_source),
            "launcher_source_sha256": base._sha256_file(launcher_source),
            "service_source": str(service_source),
            "service_source_sha256": base._sha256_file(service_source),
            "launcher_pid": os.getpid(),
            "launcher_pid_start_ticks": base._process_start_ticks(os.getpid()),
            "boot_id": base._boot_id(),
        },
    )

    service_command = [
        *base._sanitized_service_environment(),
        str(Path(sys.executable).resolve(strict=True)),
        str(service_source),
        "_service",
        "--control-directory",
        str(run_dir),
        "--run-id",
        args.run_id,
        "--unit-name",
        unit_name,
        "--working-directory",
        str(workdir),
        "--output-root",
        str(output_root),
        "--command-hash",
        command_hash,
        "--",
        *command,
    ]
    shell_command = (
        "exec "
        + shlex.join(service_command)
        + " >> "
        + shlex.quote(str(log_path))
        + " 2>&1"
    )
    dispatch_command = [
        *base._sanitized_service_environment(),
        "/usr/bin/tmux",
        "-L",
        socket_name,
        "new-session",
        "-d",
        "-s",
        session_name,
        "/bin/sh",
        "-c",
        shell_command,
    ]
    dispatch = subprocess.run(
        dispatch_command,
        cwd=workdir,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    identity = _tmux_identity(socket_name, session_name)
    base._write_new_json(
        run_dir / "launch.dispatched.json",
        {
            **common,
            "dispatched_at_utc": base._utc_now(),
            "dispatch_returncode": dispatch.returncode,
            "dispatch_stderr_hash": base._payload_hash(dispatch.stderr),
            "tmux_identity": identity,
            "dispatch_succeeded": dispatch.returncode == 0,
        },
    )
    if dispatch.returncode != 0 or identity["session_exists"] is not True:
        raise base.LaunchError(
            "tmux dispatch failed; reservation is consumed and retry is forbidden"
        )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "control_directory": str(run_dir),
                "output_root": str(output_root),
                "tmux_socket_name": socket_name,
                "tmux_session_name": session_name,
                "pane_pid": identity["pane_pid"],
                "dispatch_succeeded": True,
                "automatic_restart_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


def status(control_directory: Path) -> dict[str, Any]:
    control = control_directory.expanduser().resolve(strict=True)
    requested = base._read_receipt(control / "launch.requested.json")
    socket_name = str(requested["tmux_socket_name"])
    session_name = str(requested["tmux_session_name"])
    identity = _tmux_identity(socket_name, session_name)
    receipts: dict[str, Any] = {}
    for name in (
        "launch.dispatched.json",
        "service.started.json",
        "runner.started.json",
        "service.exited.json",
    ):
        path = control / name
        receipts[name] = base._read_receipt(path) if path.is_file() else None
    exited = receipts["service.exited.json"]
    if isinstance(exited, dict):
        phase = (
            "completed"
            if exited.get("runner_returncode") == 0
            and exited.get("formal_report", {}).get("present") is True
            else "failed"
        )
    elif identity["session_exists"]:
        phase = "running"
    else:
        phase = "orphaned_without_exit_receipt"
    output_root = Path(str(requested["output_root"]))
    return {
        "monitor_version": "tmux_detached_sec13f_formal_once_monitor_v1",
        "observed_at_utc": base._utc_now(),
        "run_id": requested["run_id"],
        "control_directory": str(control),
        "output_root": str(output_root),
        "phase": phase,
        "tmux_identity": identity,
        "receipt_presence": {key: value is not None for key, value in receipts.items()},
        "runner_returncode": (
            exited.get("runner_returncode") if isinstance(exited, dict) else None
        ),
        "formal_report": base._regular_artifact(output_root / "measurement.report.json"),
        "formal_failure_receipt": base._regular_artifact(
            output_root / "measurement.failure.json"
        ),
        "stdout_stderr_log": base._regular_artifact(
            Path(str(requested["stdout_stderr_log"]))
        ),
        "automatic_restart_authorized": False,
        "retry_authorized": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    launch_parser = commands.add_parser("launch")
    launch_parser.add_argument("--run-id", required=True)
    launch_parser.add_argument("--control-base", type=Path, required=True)
    launch_parser.add_argument("--output-root", type=Path, required=True)
    launch_parser.add_argument("--working-directory", type=Path, required=True)
    launch_parser.add_argument("command", nargs=argparse.REMAINDER)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--control-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "launch":
            return launch(args)
        print(json.dumps(status(args.control_directory), sort_keys=True))
        return 0
    except (base.LaunchError, FileExistsError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "launcher_version": LAUNCHER_VERSION,
                    "completed": False,
                    "error_type": type(exc).__name__,
                    "error_message_hash": base._payload_hash(str(exc)),
                    "retry_authorized": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
