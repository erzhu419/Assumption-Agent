#!/usr/bin/env python3
from __future__ import annotations

"""Launch one frozen SEC-13F controls run in an isolated tmux server.

This is a one-shot launcher, not a process supervisor.  It accepts only the
frozen controls module, permanently consumes both the run id and output-path
reservation, strips the inherited environment, and never authorizes restart,
retry, recovery, or replay.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
from typing import Any, Mapping, Sequence

import launch_detached_formal_once as base


LAUNCHER_VERSION = "tmux_detached_sec13f_controls_once_launcher_v1"
RECEIPT_VERSION = "tmux_detached_sec13f_controls_once_receipt_v1"
RUNNER_MODULE = "replication_runtime.financial_sec13f_contract_v2.controls_runner"
CONTROL_BASE_RELATIVE_PATH = (
    "artifacts/financial_sec13f_contract_v2_controls_launch_control"
)
REQUIRED_RUNNER_FLAGS = (
    "--project-root",
    "--controls-preregistration",
    "--controls-freeze",
    "--benchmark-root",
    "--prewarm",
    "--output-root",
    "--env-file",
)


def _controls_freeze_identity(
    command: Sequence[str],
    *,
    working_directory: Path,
    launcher_source: Path,
) -> dict[str, str]:
    """Bind the once-only reservation to the committed freeze and launchers."""

    try:
        index = list(command).index("--controls-freeze")
        raw_path = list(command)[index + 1]
    except (ValueError, IndexError) as exc:
        raise base.LaunchError("controls freeze argument is missing") from exc
    freeze_path = base._normalized(raw_path, relative_to=working_directory)
    if freeze_path.is_symlink() or not freeze_path.is_file():
        raise base.LaunchError("controls freeze is unavailable")
    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise base.LaunchError("controls freeze is not valid JSON") from exc
    if not isinstance(freeze, dict):
        raise base.LaunchError("controls freeze is not one JSON object")
    body = dict(freeze)
    manifest_hash = body.pop("manifest_hash", None)
    closure = freeze.get("launcher_source_closure")
    rows = closure.get("files") if isinstance(closure, dict) else None
    if (
        not isinstance(manifest_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", manifest_hash)
        or base._payload_hash(body) != manifest_hash
        or not isinstance(rows, list)
        or len(rows) != 2
        or closure.get("file_count") != 2
        or closure.get("file_set_hash") != base._payload_hash(rows)
    ):
        raise base.LaunchError("controls freeze identity is malformed")
    expected_sources = {
        "controls_launcher": launcher_source.resolve(strict=True),
        "base_launcher": Path(base.__file__).resolve(strict=True),
    }
    by_role = {
        str(row.get("role")): row for row in rows if isinstance(row, dict)
    }
    if set(by_role) != set(expected_sources):
        raise base.LaunchError("controls launcher closure is incomplete")
    for role, source in expected_sources.items():
        row = by_role[role]
        relative = row.get("relative_path")
        expected_sha = row.get("file_sha256")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(expected_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or (working_directory / relative).resolve(strict=True) != source
            or base._sha256_file(source) != expected_sha
        ):
            raise base.LaunchError("controls launcher source differs from freeze")
    return {
        "controls_execution_freeze_hash": manifest_hash,
        "controls_execution_freeze_file_sha256": base._sha256_file(freeze_path),
        "controls_execution_freeze_path": str(freeze_path),
        "controls_launcher_source_sha256": base._sha256_file(
            expected_sources["controls_launcher"]
        ),
        "base_launcher_source_sha256": base._sha256_file(
            expected_sources["base_launcher"]
        ),
    }


def _safe_controls_command(
    command: Sequence[str],
    output_root: Path,
    *,
    working_directory: Path,
) -> list[str]:
    result = list(command)
    if result and result[0] == "--":
        result = result[1:]
    if len(result) < 4:
        raise base.LaunchError("controls command is incomplete")
    executable = Path(result[0]).expanduser().resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise base.LaunchError("controls Python executable is unavailable")
    if executable != Path(sys.executable).resolve(strict=True):
        raise base.LaunchError(
            "controls must use this launcher's Python executable"
        )
    if result[1:4] != ["-B", "-m", RUNNER_MODULE]:
        raise base.LaunchError("only the frozen financial controls module is accepted")
    for value in result:
        lowered = value.lower()
        if value.startswith("-") and ("recover" in lowered or "retry" in lowered):
            raise base.LaunchError("controls recovery and retry flags are forbidden")
        if lowered.startswith("sk-") or re.search(
            r"(?:api[_-]?key|authorization|bearer)=", lowered
        ):
            raise base.LaunchError("secret-like value must not appear in command argv")
        if "\x00" in value or "\n" in value or "\r" in value:
            raise base.LaunchError(
                "controls argv contains a forbidden control character"
            )
    for flag in REQUIRED_RUNNER_FLAGS:
        if result.count(flag) != 1:
            raise base.LaunchError(f"controls must contain exactly one {flag}")
        index = result.index(flag)
        if index + 1 >= len(result) or result[index + 1].startswith("--"):
            raise base.LaunchError(f"{flag} lacks a value")
    output_index = result.index("--output-root")
    command_output = base._normalized(
        result[output_index + 1], relative_to=working_directory
    )
    if command_output != output_root:
        raise base.LaunchError(
            "controls --output-root differs from launcher reservation"
        )
    result[0] = str(executable)
    return result


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


def _common_receipt(
    *,
    args: argparse.Namespace,
    output_root: Path,
    command_hash: str,
    freeze_identity: Mapping[str, str],
) -> dict[str, Any]:
    socket_name = "assumption-controls-" + hashlib.sha256(
        args.run_id.encode("ascii")
    ).hexdigest()[:16]
    session_name = "controls-once"
    return {
        "receipt_version": RECEIPT_VERSION,
        "launcher_version": LAUNCHER_VERSION,
        "run_id": args.run_id,
        "unit_name": f"tmux:{socket_name}:{session_name}",
        "detachment_backend": "isolated_tmux_server_v1",
        "tmux_socket_name": socket_name,
        "tmux_session_name": session_name,
        "output_root": str(output_root),
        "output_root_hash": base._payload_hash(str(output_root)),
        "command_hash": command_hash,
        **dict(freeze_identity),
        "launch_attempt": 1,
        "automatic_restart_authorized": False,
        "retry_authorized": False,
        "recovery_authorized": False,
        "replay_authorized": False,
    }


def launch(args: argparse.Namespace) -> int:
    if not base.RUN_ID_PATTERN.fullmatch(args.run_id):
        raise base.LaunchError("run id is malformed")
    workdir = base._normalized(args.working_directory)
    if workdir.is_symlink() or not workdir.is_dir():
        raise base.LaunchError("working directory is unavailable")
    expected_runner = workdir / Path(*RUNNER_MODULE.split(".")).with_suffix(".py")
    if expected_runner.is_symlink() or not expected_runner.is_file():
        raise base.LaunchError("SEC-13F controls source is unavailable")
    output_root = base._normalized(args.output_root, relative_to=workdir)
    if os.path.lexists(output_root):
        raise base.LaunchError("controls output root already exists")
    command = _safe_controls_command(
        args.command, output_root, working_directory=workdir
    )
    launcher_source = Path(__file__).resolve(strict=True)
    freeze_identity = _controls_freeze_identity(
        command,
        working_directory=workdir,
        launcher_source=launcher_source,
    )
    if not Path("/usr/bin/tmux").is_file():
        raise base.LaunchError("tmux is unavailable")

    control_base = base._normalized(args.control_base, relative_to=workdir)
    expected_control_base = (workdir / CONTROL_BASE_RELATIVE_PATH).resolve()
    if control_base != expected_control_base:
        raise base.LaunchError("controls must use the fixed one-shot control base")
    if control_base == output_root or output_root in control_base.parents:
        raise base.LaunchError("control base is inside the controls output root")
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
        base._write_new_json(
            run_dir / "launch.rejected.json",
            {
                "receipt_version": RECEIPT_VERSION,
                "launcher_version": LAUNCHER_VERSION,
                "run_id": args.run_id,
                "rejected_at_utc": base._utc_now(),
                "reason": "output_path_already_reserved",
                "output_root_hash": base._payload_hash(str(output_root)),
                "runner_started": False,
                "retry_authorized": False,
                "recovery_authorized": False,
                "replay_authorized": False,
            },
        )
        raise base.LaunchError("controls output path is already reserved") from exc

    freeze_reservation = (
        control_base
        / "freeze-reservations"
        / freeze_identity["controls_execution_freeze_hash"]
    )
    freeze_reservation.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        freeze_reservation.mkdir(mode=0o700)
    except FileExistsError as exc:
        base._write_new_json(
            run_dir / "launch.rejected.json",
            {
                "receipt_version": RECEIPT_VERSION,
                "launcher_version": LAUNCHER_VERSION,
                "run_id": args.run_id,
                "rejected_at_utc": base._utc_now(),
                "reason": "controls_freeze_already_consumed",
                **freeze_identity,
                "runner_started": False,
                "retry_authorized": False,
                "recovery_authorized": False,
                "replay_authorized": False,
            },
        )
        raise base.LaunchError(
            "controls execution freeze has already been consumed"
        ) from exc

    command_hash = base._payload_hash(command)
    common = _common_receipt(
        args=args,
        output_root=output_root,
        command_hash=command_hash,
        freeze_identity=freeze_identity,
    )
    socket_name = str(common["tmux_socket_name"])
    session_name = str(common["tmux_session_name"])
    unit_name = str(common["unit_name"])
    log_path = run_dir / "runner.stdout-stderr.log"
    _create_log(log_path)
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
        freeze_reservation / "owner.json",
        {
            **common,
            "reserved_at_utc": base._utc_now(),
            "run_control_directory": str(run_dir),
            "reservation_is_permanent": True,
            "reservation_scope": "controls_execution_freeze_hash",
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
            "freeze_reservation": str(freeze_reservation),
            "command": command,
            "stdout_stderr_log": str(log_path),
            "launcher_source": str(launcher_source),
            "launcher_source_sha256": base._sha256_file(launcher_source),
            "base_launcher_source": str(Path(base.__file__).resolve(strict=True)),
            "base_launcher_source_sha256": base._sha256_file(
                Path(base.__file__).resolve(strict=True)
            ),
            "launcher_pid": os.getpid(),
            "launcher_pid_start_ticks": base._process_start_ticks(os.getpid()),
            "boot_id": base._boot_id(),
        },
    )

    service_command = [
        *base._sanitized_service_environment(),
        str(Path(sys.executable).resolve(strict=True)),
        str(launcher_source),
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
    dispatch = subprocess.run(
        [
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
        ],
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
                "retry_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _service_exit_receipt(
    *,
    args: argparse.Namespace,
    command: Sequence[str],
    service_pid: int,
    runner_pid: int | None,
    runner_pid_start_ticks: int | None,
    runner_started: bool,
    runner_returncode: int,
    received_signals: Sequence[int],
    started_at: str,
    failure_type: str | None = None,
    failure_message: str | None = None,
) -> dict[str, Any]:
    output_root = Path(args.output_root)
    body: dict[str, Any] = {
        "receipt_version": RECEIPT_VERSION,
        "launcher_version": LAUNCHER_VERSION,
        "run_id": args.run_id,
        "unit_name": args.unit_name,
        "service_pid": service_pid,
        "service_pid_start_ticks": base._process_start_ticks(service_pid),
        "runner_pid": runner_pid,
        "runner_pid_start_ticks": runner_pid_start_ticks,
        "runner_started": runner_started,
        "started_at_utc": started_at,
        "finished_at_utc": base._utc_now(),
        "runner_returncode": runner_returncode,
        "runner_exit_status": runner_returncode if runner_returncode >= 0 else None,
        "runner_term_signal": -runner_returncode if runner_returncode < 0 else None,
        "received_service_signals": list(received_signals),
        "output_root": str(output_root),
        "controls_report": base._regular_artifact(output_root / "controls.report.json"),
        "controls_failure_receipt": base._regular_artifact(
            output_root / "controls.failure.json"
        ),
        "command_hash": base._payload_hash(list(command)),
        "launch_attempt": 1,
        "automatic_restart_authorized": False,
        "retry_authorized": False,
        "recovery_authorized": False,
        "replay_authorized": False,
    }
    if failure_type is not None:
        body["wrapper_failure_type"] = failure_type
        body["wrapper_failure_message_hash"] = base._payload_hash(
            failure_message or ""
        )
    return body


def _service(args: argparse.Namespace) -> int:
    control = Path(args.control_directory).resolve(strict=True)
    requested = base._read_receipt(control / "launch.requested.json")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    output_root = Path(args.output_root).resolve(strict=False)
    if (
        requested.get("run_id") != args.run_id
        or requested.get("unit_name") != args.unit_name
        or requested.get("output_root") != str(output_root)
        or requested.get("command") != command
        or requested.get("command_hash") != args.command_hash
        or base._payload_hash(command) != args.command_hash
    ):
        raise base.LaunchError("service arguments differ from immutable launch receipt")

    started_at = base._utc_now()
    service_pid = os.getpid()
    output_preexisting = os.path.lexists(output_root)
    base._write_new_json(
        control / "service.started.json",
        {
            "receipt_version": RECEIPT_VERSION,
            "launcher_version": LAUNCHER_VERSION,
            "started_at_utc": started_at,
            "run_id": args.run_id,
            "unit_name": args.unit_name,
            "service_pid": service_pid,
            "service_pid_start_ticks": base._process_start_ticks(service_pid),
            "boot_id": base._boot_id(),
            "output_root": str(output_root),
            "output_root_preexisting": output_preexisting,
            "command_hash": args.command_hash,
            "launch_attempt": 1,
            "automatic_restart_authorized": False,
            "retry_authorized": False,
            "recovery_authorized": False,
            "replay_authorized": False,
        },
    )
    if output_preexisting:
        base._write_new_json(
            control / "service.exited.json",
            _service_exit_receipt(
                args=args,
                command=command,
                service_pid=service_pid,
                runner_pid=None,
                runner_pid_start_ticks=None,
                runner_started=False,
                runner_returncode=125,
                received_signals=(),
                started_at=started_at,
                failure_type="controls_output_root_preexisting",
                failure_message="controls output root existed at service boundary",
            ),
        )
        return 125

    received_signals: list[int] = []

    def record_signal(signum: int, _frame: object) -> None:
        received_signals.append(signum)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, record_signal)

    child: subprocess.Popen[bytes] | None = None
    try:
        child = subprocess.Popen(
            command,
            cwd=args.working_directory,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
        runner_pid_start_ticks = base._process_start_ticks(child.pid)
        base._write_new_json(
            control / "runner.started.json",
            {
                "receipt_version": RECEIPT_VERSION,
                "launcher_version": LAUNCHER_VERSION,
                "started_at_utc": base._utc_now(),
                "run_id": args.run_id,
                "unit_name": args.unit_name,
                "service_pid": service_pid,
                "runner_pid": child.pid,
                "runner_pid_start_ticks": runner_pid_start_ticks,
                "boot_id": base._boot_id(),
                "output_root_preexisting": False,
                "command_hash": args.command_hash,
                "launch_attempt": 1,
                "automatic_restart_authorized": False,
                "retry_authorized": False,
                "recovery_authorized": False,
                "replay_authorized": False,
            },
        )
        while True:
            try:
                returncode = child.wait()
                break
            except InterruptedError:
                continue
        base._write_new_json(
            control / "service.exited.json",
            _service_exit_receipt(
                args=args,
                command=command,
                service_pid=service_pid,
                runner_pid=child.pid,
                runner_pid_start_ticks=runner_pid_start_ticks,
                runner_started=True,
                runner_returncode=returncode,
                received_signals=received_signals,
                started_at=started_at,
            ),
        )
        return returncode if returncode >= 0 else 128 + (-returncode)
    except (OSError, ValueError, base.LaunchError) as exc:
        returncode = 126
        if child is not None and child.poll() is None:
            child.terminate()
            returncode = child.wait()
        if not os.path.lexists(control / "service.exited.json"):
            base._write_new_json(
                control / "service.exited.json",
                _service_exit_receipt(
                    args=args,
                    command=command,
                    service_pid=service_pid,
                    runner_pid=child.pid if child is not None else None,
                    runner_pid_start_ticks=(
                        base._process_start_ticks(child.pid)
                        if child is not None
                        else None
                    ),
                    runner_started=child is not None,
                    runner_returncode=returncode,
                    received_signals=received_signals,
                    started_at=started_at,
                    failure_type=type(exc).__name__,
                    failure_message=str(exc),
                ),
            )
        return returncode if returncode >= 0 else 128 + (-returncode)


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
            and exited.get("controls_report", {}).get("present") is True
            and exited.get("controls_failure_receipt", {}).get("present") is False
            else "failed"
        )
    elif identity["session_exists"]:
        phase = "running"
    elif receipts["launch.dispatched.json"] is None:
        phase = "launch_pending"
    else:
        phase = "orphaned_without_exit_receipt"
    output_root = Path(str(requested["output_root"]))
    return {
        "monitor_version": "tmux_detached_sec13f_controls_once_monitor_v1",
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
        "controls_report": base._regular_artifact(
            output_root / "controls.report.json"
        ),
        "controls_failure_receipt": base._regular_artifact(
            output_root / "controls.failure.json"
        ),
        "stdout_stderr_log": base._regular_artifact(
            Path(str(requested["stdout_stderr_log"]))
        ),
        "automatic_restart_authorized": False,
        "retry_authorized": False,
        "recovery_authorized": False,
        "replay_authorized": False,
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
    service_parser = commands.add_parser("_service", help=argparse.SUPPRESS)
    service_parser.add_argument("--control-directory", type=Path, required=True)
    service_parser.add_argument("--run-id", required=True)
    service_parser.add_argument("--unit-name", required=True)
    service_parser.add_argument("--working-directory", type=Path, required=True)
    service_parser.add_argument("--output-root", type=Path, required=True)
    service_parser.add_argument("--command-hash", required=True)
    service_parser.add_argument("command", nargs=argparse.REMAINDER)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--control-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "launch":
            return launch(args)
        if args.action == "_service":
            return _service(args)
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
                    "recovery_authorized": False,
                    "replay_authorized": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
