#!/usr/bin/env python3
from __future__ import annotations

"""Launch one frozen SEC-13F sealed evaluation in an isolated tmux server."""

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


LAUNCHER_VERSION = "tmux_detached_sec13f_sealed_once_launcher_v1"
RECEIPT_VERSION = "tmux_detached_sec13f_sealed_once_receipt_v1"
RUNNER_MODULE = "replication_runtime.financial_sec13f_contract_v2.sealed_runner"
CONTROL_BASE_RELATIVE_PATH = "artifacts/financial_sec13f_contract_v2_sealed_launch_control"
REQUIRED_FLAGS = (
    "--project-root",
    "--benchmark-root",
    "--measurement-view",
    "--sealed-payload",
    "--prewarm",
    "--execution-freeze",
    "--env-file",
    "--output-root",
)


def _safe_command(
    command: Sequence[str],
    *,
    working_directory: Path,
    output_root: Path,
) -> list[str]:
    result = list(command)
    if result and result[0] == "--":
        result = result[1:]
    if len(result) < 4:
        raise base.LaunchError("sealed runner command is incomplete")
    executable = Path(result[0]).expanduser().resolve(strict=True)
    if executable != Path(sys.executable).resolve(strict=True):
        raise base.LaunchError("sealed runner must use the launcher Python")
    if result[1:4] != ["-B", "-m", RUNNER_MODULE]:
        raise base.LaunchError("only the frozen sealed runner is accepted")
    for flag in REQUIRED_FLAGS:
        if result.count(flag) != 1:
            raise base.LaunchError(f"sealed runner requires exactly one {flag}")
        index = result.index(flag)
        if index + 1 == len(result) or result[index + 1].startswith("--"):
            raise base.LaunchError(f"{flag} lacks a value")
    for value in result:
        lowered = value.lower()
        if any(word in lowered for word in ("--recover", "--retry", "--replay")):
            raise base.LaunchError("sealed recovery/retry/replay flags are forbidden")
        if lowered.startswith("sk-") or re.search(r"(?:api[_-]?key|bearer)=", lowered):
            raise base.LaunchError("secret-like value appeared in sealed argv")
        if any(character in value for character in ("\x00", "\n", "\r")):
            raise base.LaunchError("sealed argv contains a control character")
    supplied_output = base._normalized(
        result[result.index("--output-root") + 1],
        relative_to=working_directory,
    )
    if supplied_output != output_root:
        raise base.LaunchError("sealed runner output differs from reservation")
    result[0] = str(executable)
    return result


def _freeze_identity(command: Sequence[str], workdir: Path) -> dict[str, str]:
    path = base._normalized(
        command[command.index("--execution-freeze") + 1],
        relative_to=workdir,
    )
    if path.is_symlink() or not path.is_file():
        raise base.LaunchError("sealed execution freeze is unavailable")
    try:
        relative = path.resolve(strict=True).relative_to(workdir.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise base.LaunchError("sealed execution freeze escaped the project") from exc
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                str(workdir),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                relative,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = subprocess.run(
            ["git", "-C", str(workdir), "rev-parse", "HEAD^{commit}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
        tracked = subprocess.run(
            [
                "git",
                "-C",
                str(workdir),
                "ls-files",
                "--error-unmatch",
                "--full-name",
                "--",
                relative,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        committed_bytes = subprocess.run(
            ["git", "-C", str(workdir), "show", f"{commit}:{tracked}"],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise base.LaunchError("sealed execution freeze Git binding failed") from exc
    if (
        status.stdout.strip()
        or re.fullmatch(r"[0-9a-f]{40,64}", commit) is None
        or not tracked
        or hashlib.sha256(committed_bytes).hexdigest() != base._sha256_file(path)
    ):
        raise base.LaunchError("sealed execution freeze is not committed and clean")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise base.LaunchError("sealed execution freeze is unreadable") from exc
    if not isinstance(value, dict):
        raise base.LaunchError("sealed execution freeze is not one object")
    body = dict(value)
    manifest_hash = body.pop("manifest_hash", None)
    source = value.get("execution_source_closure")
    rows = source.get("supplemental_files") if isinstance(source, Mapping) else None
    launcher_relative = "scripts/launch_tmux_detached_sealed_once.py"
    by_path = {
        str(row.get("relative_path")): row
        for row in rows or ()
        if isinstance(row, Mapping)
    }
    launcher = Path(__file__).resolve(strict=True)
    if (
        not isinstance(manifest_hash, str)
        or base._payload_hash(body) != manifest_hash
        or launcher_relative not in by_path
        or by_path[launcher_relative].get("file_sha256") != base._sha256_file(launcher)
    ):
        raise base.LaunchError("sealed execution freeze identity drifted")
    return {
        "sealed_execution_freeze_hash": manifest_hash,
        "sealed_execution_freeze_file_sha256": base._sha256_file(path),
        "sealed_launcher_source_sha256": base._sha256_file(launcher),
        "sealed_execution_freeze_committed_at_git_commit": commit,
    }


def _tmux_identity(socket: str, session: str) -> dict[str, Any]:
    query = subprocess.run(
        ["/usr/bin/tmux", "-L", socket, "list-panes", "-t", session, "-F", "#{pane_pid} #{pane_dead} #{pane_dead_status}"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    row = query.stdout.strip().split()
    return {
        "session_exists": query.returncode == 0,
        "pane_pid": int(row[0]) if row and row[0].isdigit() else 0,
        "pane_dead": row[1] == "1" if len(row) > 1 else None,
        "pane_dead_status": int(row[2]) if len(row) > 2 and row[2].isdigit() else None,
        "automatic_restart_authorized": False,
    }


def _common(
    *,
    args: argparse.Namespace,
    output_root: Path,
    command_hash: str,
    freeze: Mapping[str, str],
) -> dict[str, Any]:
    socket = "assumption-sealed-" + hashlib.sha256(args.run_id.encode("ascii")).hexdigest()[:16]
    return {
        "receipt_version": RECEIPT_VERSION,
        "launcher_version": LAUNCHER_VERSION,
        "run_id": args.run_id,
        "unit_name": f"tmux:{socket}:sealed-once",
        "tmux_socket_name": socket,
        "tmux_session_name": "sealed-once",
        "output_root": str(output_root),
        "output_root_hash": base._payload_hash(str(output_root)),
        "command_hash": command_hash,
        **dict(freeze),
        "launch_attempt": 1,
        "automatic_restart_authorized": False,
        "retry_authorized": False,
        "recovery_authorized": False,
        "replay_authorized": False,
        "resampling_authorized": False,
        "provider_switch_authorized": False,
    }


def launch(args: argparse.Namespace) -> int:
    if not base.RUN_ID_PATTERN.fullmatch(args.run_id):
        raise base.LaunchError("sealed run id is malformed")
    workdir = base._normalized(args.working_directory)
    output_root = base._normalized(args.output_root, relative_to=workdir)
    if os.path.lexists(output_root):
        raise base.LaunchError("sealed output root already exists")
    command = _safe_command(args.command, working_directory=workdir, output_root=output_root)
    freeze = _freeze_identity(command, workdir)
    control_base = base._normalized(args.control_base, relative_to=workdir)
    if control_base != (workdir / CONTROL_BASE_RELATIVE_PATH).resolve():
        raise base.LaunchError("sealed launcher requires the fixed control base")
    control_base.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_dir = control_base / "runs" / args.run_id
    run_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_dir.mkdir(mode=0o700)
    output_key = hashlib.sha256(os.fsencode(str(output_root))).hexdigest()
    output_reservation = control_base / "output-reservations" / output_key
    output_reservation.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_reservation.mkdir(mode=0o700)
    freeze_reservation = control_base / "freeze-reservations" / freeze["sealed_execution_freeze_hash"]
    freeze_reservation.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    freeze_reservation.mkdir(mode=0o700)
    common = _common(
        args=args,
        output_root=output_root,
        command_hash=base._payload_hash(command),
        freeze=freeze,
    )
    log = run_dir / "runner.stdout-stderr.log"
    descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    for reservation, scope in (
        (output_reservation, "sealed_output_path"),
        (freeze_reservation, "sealed_execution_freeze_hash"),
    ):
        base._write_new_json(
            reservation / "owner.json",
            {**common, "reserved_at_utc": base._utc_now(), "reservation_scope": scope, "reservation_is_permanent": True},
        )
    base._write_new_json(
        run_dir / "launch.requested.json",
        {
            **common,
            "requested_at_utc": base._utc_now(),
            "working_directory": str(workdir),
            "stdout_stderr_log": str(log),
            "command_persisted": False,
            "private_artifact_paths_persisted": False,
        },
    )
    service_command = [
        *base._sanitized_service_environment(),
        str(Path(sys.executable).resolve(strict=True)),
        str(Path(__file__).resolve(strict=True)),
        "_service",
        "--control-directory", str(run_dir),
        "--working-directory", str(workdir),
        "--output-root", str(output_root),
        "--command-hash", str(common["command_hash"]),
        "--",
        *command,
    ]
    shell = "exec " + shlex.join(service_command) + " >> " + shlex.quote(str(log)) + " 2>&1"
    dispatch = subprocess.run(
        [*base._sanitized_service_environment(), "/usr/bin/tmux", "-L", common["tmux_socket_name"], "new-session", "-d", "-s", "sealed-once", "/bin/sh", "-c", shell],
        cwd=workdir,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    identity = _tmux_identity(str(common["tmux_socket_name"]), "sealed-once")
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
        raise base.LaunchError("sealed tmux dispatch failed; reservations remain consumed")
    print(json.dumps({"run_id": args.run_id, "control_directory": str(run_dir), "dispatch_succeeded": True}, sort_keys=True))
    return 0


def _service(args: argparse.Namespace) -> int:
    control = args.control_directory.expanduser().resolve(strict=True)
    requested = base._read_receipt(control / "launch.requested.json")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if base._payload_hash(command) != args.command_hash or requested.get("command_hash") != args.command_hash:
        raise base.LaunchError("sealed service command differs from launch hash")
    output = args.output_root.expanduser().resolve()
    preexisting = os.path.lexists(output)
    base._write_new_json(
        control / "service.started.json",
        {
            "receipt_version": RECEIPT_VERSION,
            "launcher_version": LAUNCHER_VERSION,
            "started_at_utc": base._utc_now(),
            "output_root_hash": base._payload_hash(str(output)),
            "output_root_preexisting": preexisting,
            "command_hash": args.command_hash,
            "retry_authorized": False,
            "replay_authorized": False,
        },
    )
    if preexisting:
        returncode = 125
        runner_pid = None
    else:
        identity_fields = (
            "sealed_execution_freeze_hash",
            "sealed_execution_freeze_file_sha256",
            "sealed_launcher_source_sha256",
            "sealed_execution_freeze_committed_at_git_commit",
        )
        try:
            current_freeze = _freeze_identity(
                command,
                args.working_directory.expanduser().resolve(strict=True),
            )
        except (base.LaunchError, OSError, ValueError):
            current_freeze = None
        if current_freeze is None or any(
            current_freeze.get(field) != requested.get(field)
            for field in identity_fields
        ):
            runner_pid = None
            returncode = 125
            base._write_new_json(
                control / "service.freeze-rejected.json",
                {
                    "receipt_version": RECEIPT_VERSION,
                    "launcher_version": LAUNCHER_VERSION,
                    "rejected_at_utc": base._utc_now(),
                    "freeze_identity_matches_launch_request": False,
                    "runner_started": False,
                    "retry_authorized": False,
                    "replay_authorized": False,
                },
            )
        else:
            # This identity check is deliberately the final operation before
            # process creation, closing the launch-to-service freeze TOCTOU.
            child = subprocess.Popen(command, cwd=args.working_directory, stdin=subprocess.DEVNULL, close_fds=True)
            runner_pid = child.pid
            base._write_new_json(
                control / "runner.started.json",
                {
                    "receipt_version": RECEIPT_VERSION,
                    "launcher_version": LAUNCHER_VERSION,
                    "started_at_utc": base._utc_now(),
                    "runner_pid": runner_pid,
                    "runner_pid_start_ticks": base._process_start_ticks(runner_pid),
                    "command_hash": args.command_hash,
                    "freeze_identity_matches_launch_request": True,
                    "retry_authorized": False,
                    "replay_authorized": False,
                },
            )
            returncode = child.wait()
    base._write_new_json(
        control / "service.exited.json",
        {
            "receipt_version": RECEIPT_VERSION,
            "launcher_version": LAUNCHER_VERSION,
            "finished_at_utc": base._utc_now(),
            "runner_pid": runner_pid,
            "runner_returncode": returncode,
            "sealed_report": base._regular_artifact(output / "sealed.report.json"),
            "sealed_failure_receipt": base._regular_artifact(output / "sealed.failure.json"),
            "command_hash": args.command_hash,
            "retry_authorized": False,
            "recovery_authorized": False,
            "replay_authorized": False,
        },
    )
    return returncode if returncode >= 0 else 128 + (-returncode)


def status(control_directory: Path) -> dict[str, Any]:
    control = control_directory.expanduser().resolve(strict=True)
    requested = base._read_receipt(control / "launch.requested.json")
    output = Path(str(requested["output_root"]))
    exited_path = control / "service.exited.json"
    exited = base._read_receipt(exited_path) if exited_path.is_file() else None
    identity = _tmux_identity(str(requested["tmux_socket_name"]), str(requested["tmux_session_name"]))
    report = base._regular_artifact(output / "sealed.report.json")
    failure = base._regular_artifact(output / "sealed.failure.json")
    phase = "completed" if report["present"] and not failure["present"] else "failed" if exited else "running" if identity["session_exists"] else "orphaned"
    return {
        "monitor_version": "tmux_detached_sec13f_sealed_once_monitor_v1",
        "phase": phase,
        "run_id": requested["run_id"],
        "tmux_identity": identity,
        "sealed_report": report,
        "sealed_failure_receipt": failure,
        "runner_returncode": exited.get("runner_returncode") if exited else None,
        "retry_authorized": False,
        "replay_authorized": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    launch_parser = actions.add_parser("launch")
    launch_parser.add_argument("--run-id", required=True)
    launch_parser.add_argument("--control-base", type=Path, required=True)
    launch_parser.add_argument("--output-root", type=Path, required=True)
    launch_parser.add_argument("--working-directory", type=Path, required=True)
    launch_parser.add_argument("command", nargs=argparse.REMAINDER)
    service = actions.add_parser("_service", help=argparse.SUPPRESS)
    service.add_argument("--control-directory", type=Path, required=True)
    service.add_argument("--working-directory", type=Path, required=True)
    service.add_argument("--output-root", type=Path, required=True)
    service.add_argument("--command-hash", required=True)
    service.add_argument("command", nargs=argparse.REMAINDER)
    status_parser = actions.add_parser("status")
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
        print(json.dumps({"launcher_version": LAUNCHER_VERSION, "completed": False, "error_type": type(exc).__name__, "error_message_hash": base._payload_hash(str(exc)), "retry_authorized": False}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
