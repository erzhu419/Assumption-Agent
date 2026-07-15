#!/usr/bin/env python3
from __future__ import annotations

"""Launch one frozen SEC-13F contract runner as a detached user service.

The launcher is deliberately narrower than a generic process supervisor.  It
accepts only the financial-semantic formal runner, refuses recovery mode,
reserves the requested output path permanently, and dispatches the command
exactly once with systemd restart disabled.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from typing import Any, Sequence


LAUNCHER_VERSION = "detached_sec13f_contract_formal_once_launcher_v1"
RECEIPT_VERSION = "detached_sec13f_contract_formal_once_receipt_v1"
RUNNER_MODULE = "replication_runtime.financial_sec13f_contract_v2.runner"
RUN_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,47}\Z")
REQUIRED_RUNNER_FLAGS = (
    "--project-root",
    "--benchmark-root",
    "--measurement-view",
    "--prewarm",
    "--execution-freeze",
    "--env-file",
    "--output-root",
)


class LaunchError(RuntimeError):
    """A fail-closed launch invariant was not satisfied."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_hash(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hashed_receipt(body: dict[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["receipt_hash"] = _payload_hash(body)
    return result


def _write_new_json(path: Path, body: dict[str, Any]) -> None:
    """Atomically create a receipt and never replace prior evidence."""

    if os.path.lexists(path):
        raise FileExistsError(path)
    payload = _canonical_bytes(_hashed_receipt(body)) + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.lexists(path):
            raise FileExistsError(path)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def _read_receipt(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LaunchError(f"receipt is unavailable: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LaunchError(f"receipt is not an object: {path}")
    declared = payload.pop("receipt_hash", None)
    if declared != _payload_hash(payload):
        raise LaunchError(f"receipt hash mismatch: {path}")
    payload["receipt_hash"] = declared
    return payload


def _process_start_ticks(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        suffix = text[text.rfind(")") + 2 :].split()
        return int(suffix[19])
    except (OSError, ValueError, IndexError):
        return None


def _boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
    except OSError:
        return None


def _normalized(path: str | Path, *, relative_to: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (relative_to or Path.cwd()) / candidate
    return candidate.resolve(strict=False)


def _safe_runner_command(
    command: Sequence[str],
    output_root: Path,
    *,
    working_directory: Path,
) -> list[str]:
    result = list(command)
    if result and result[0] == "--":
        result = result[1:]
    if len(result) < 4:
        raise LaunchError("formal runner command is incomplete")
    executable = Path(result[0]).expanduser().resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise LaunchError("formal runner Python executable is unavailable")
    if executable != Path(sys.executable).resolve(strict=True):
        raise LaunchError("formal runner must use this launcher's Python executable")
    if result[1:4] != ["-B", "-m", RUNNER_MODULE]:
        raise LaunchError("only the frozen financial formal runner is accepted")
    if "--recover-only" in result:
        raise LaunchError("recovery mode is forbidden by the one-shot launcher")
    for flag in REQUIRED_RUNNER_FLAGS:
        if result.count(flag) != 1:
            raise LaunchError(f"formal runner must contain exactly one {flag}")
    output_index = result.index("--output-root")
    if output_index + 1 >= len(result):
        raise LaunchError("--output-root lacks a value")
    command_output = _normalized(
        result[output_index + 1], relative_to=working_directory
    )
    if command_output != output_root:
        raise LaunchError("runner --output-root differs from launcher reservation")
    for value in result:
        lowered = value.lower()
        if lowered.startswith("sk-") or re.search(
            r"(?:api[_-]?key|authorization|bearer)=", lowered
        ):
            raise LaunchError("secret-like value must not appear in command argv")
        if "\x00" in value or "\n" in value or "\r" in value:
            raise LaunchError("runner argv contains a forbidden control character")
    result[0] = str(executable)
    return result


def _sanitized_service_environment() -> list[str]:
    """Return a minimal non-secret environment for wrapper and runner.

    The formal runner obtains its selected provider credential solely by
    reading its explicit ``--env-file`` after the wrapper has spawned it.
    Neither the service wrapper nor its receipts inherit the user manager's
    API-key or Codex-session environment.
    """

    home = str(Path.home().resolve(strict=True))
    user = str(os.environ.get("USER") or os.getuid())
    logname = str(os.environ.get("LOGNAME") or user)
    runtime = str(Path(f"/run/user/{os.getuid()}").resolve(strict=True))
    return [
        "/usr/bin/env",
        "-i",
        f"HOME={home}",
        f"USER={user}",
        f"LOGNAME={logname}",
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG=C.UTF-8",
        f"XDG_RUNTIME_DIR={runtime}",
        "PYTHONDONTWRITEBYTECODE=1",
    ]


def _systemd_identity(unit_name: str) -> dict[str, Any]:
    properties = (
        "LoadState",
        "ActiveState",
        "SubState",
        "Result",
        "MainPID",
        "ExecMainCode",
        "ExecMainStatus",
        "Type",
        "Restart",
        "KillMode",
    )
    completed = subprocess.run(
        [
            "/usr/bin/systemctl",
            "--user",
            "show",
            unit_name,
            *(f"--property={name}" for name in properties),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    values: dict[str, Any] = {"query_returncode": completed.returncode}
    for line in completed.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = int(value) if key in {
            "MainPID",
            "ExecMainCode",
            "ExecMainStatus",
        } and value.isdigit() else value
    return values


def _regular_artifact(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        return {"present": False}
    return {
        "present": True,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _launch(args: argparse.Namespace) -> int:
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise LaunchError(
            "run id must be 1-48 lowercase letters, digits, underscores or hyphens"
        )
    workdir = _normalized(args.working_directory)
    if workdir.is_symlink() or not workdir.is_dir():
        raise LaunchError("working directory must be a regular existing directory")
    expected_runner_source = workdir / Path(*RUNNER_MODULE.split(".")).with_suffix(
        ".py"
    )
    if expected_runner_source.is_symlink() or not expected_runner_source.is_file():
        raise LaunchError(
            "working directory does not contain the SEC contract runner source"
        )
    output_root = _normalized(args.output_root, relative_to=workdir)
    if os.path.lexists(output_root):
        raise LaunchError("formal output root already exists")
    command = _safe_runner_command(
        args.command,
        output_root,
        working_directory=workdir,
    )

    manager = subprocess.run(
        ["/usr/bin/systemctl", "--user", "is-system-running"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if manager.stdout.strip() not in {"running", "degraded"}:
        raise LaunchError("systemd user manager is not running")
    linger = subprocess.run(
        [
            "/usr/bin/loginctl",
            "show-user",
            str(os.getuid()),
            "--property=Linger",
            "--value",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if linger.returncode != 0 or linger.stdout.strip() != "yes":
        raise LaunchError(
            "systemd user lingering is required for shell-independent survival"
        )

    control_base = _normalized(args.control_base, relative_to=workdir)
    if control_base == output_root or output_root in control_base.parents:
        raise LaunchError("control base must not be inside the formal output root")
    control_base.mkdir(parents=True, exist_ok=True, mode=0o700)
    if control_base.is_symlink() or not control_base.is_dir():
        raise LaunchError("control base is not a regular directory")
    os.chmod(control_base, 0o700)

    run_dir = control_base / "runs" / args.run_id
    run_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        run_dir.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise LaunchError("run id has already been consumed") from exc

    reservation_key = hashlib.sha256(
        os.fsencode(str(output_root))
    ).hexdigest()
    reservation = control_base / "output-reservations" / reservation_key
    reservation.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        reservation.mkdir(mode=0o700)
    except FileExistsError as exc:
        _write_new_json(
            run_dir / "launch.rejected.json",
            {
                "receipt_version": RECEIPT_VERSION,
                "launcher_version": LAUNCHER_VERSION,
                "run_id": args.run_id,
                "rejected_at_utc": _utc_now(),
                "reason": "output_path_already_reserved",
                "output_root_hash": _payload_hash(str(output_root)),
                "runner_started": False,
                "retry_authorized": False,
            },
        )
        raise LaunchError("formal output path has already been reserved") from exc

    unit_name = f"assumption-formal-{args.run_id}.service"
    log_path = run_dir / "runner.stdout-stderr.log"
    launcher_source = Path(__file__).resolve(strict=True)
    command_hash = _payload_hash(command)
    _write_new_json(
        reservation / "owner.json",
        {
            "receipt_version": RECEIPT_VERSION,
            "launcher_version": LAUNCHER_VERSION,
            "reserved_at_utc": _utc_now(),
            "run_id": args.run_id,
            "unit_name": unit_name,
            "run_control_directory": str(run_dir),
            "output_root": str(output_root),
            "output_root_hash": _payload_hash(str(output_root)),
            "command_hash": command_hash,
            "reservation_is_permanent": True,
            "retry_authorized": False,
        },
    )
    _write_new_json(
        run_dir / "launch.requested.json",
        {
            "receipt_version": RECEIPT_VERSION,
            "launcher_version": LAUNCHER_VERSION,
            "requested_at_utc": _utc_now(),
            "run_id": args.run_id,
            "unit_name": unit_name,
            "working_directory": str(workdir),
            "output_root": str(output_root),
            "output_root_preexisting": False,
            "output_reservation": str(reservation),
            "command": command,
            "command_hash": command_hash,
            "stdout_stderr_log": str(log_path),
            "launcher_source": str(launcher_source),
            "launcher_source_sha256": _sha256_file(launcher_source),
            "launcher_pid": os.getpid(),
            "launcher_pid_start_ticks": _process_start_ticks(os.getpid()),
            "boot_id": _boot_id(),
            "launch_attempt": 1,
            "automatic_restart_authorized": False,
            "retry_authorized": False,
            "recover_only_authorized": False,
        },
    )

    service_command = [
        *_sanitized_service_environment(),
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
    dispatch = subprocess.run(
        [
            "/usr/bin/systemd-run",
            "--user",
            "--quiet",
            "--unit",
            unit_name,
            "--collect",
            "--property=Type=exec",
            "--property=Restart=no",
            "--property=KillMode=control-group",
            "--property=RemainAfterExit=no",
            "--property=TimeoutStartSec=90s",
            "--property=TimeoutStopSec=90s",
            "--property=SendSIGKILL=yes",
            "--property=UMask=0077",
            f"--property=WorkingDirectory={workdir}",
            f"--property=StandardOutput=append:{log_path}",
            f"--property=StandardError=append:{log_path}",
            *service_command,
        ],
        cwd=workdir,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    identity = _systemd_identity(unit_name)
    dispatch_body = {
        "receipt_version": RECEIPT_VERSION,
        "launcher_version": LAUNCHER_VERSION,
        "dispatched_at_utc": _utc_now(),
        "run_id": args.run_id,
        "unit_name": unit_name,
        "systemd_run_returncode": dispatch.returncode,
        "systemd_stderr_hash": _payload_hash(dispatch.stderr),
        "systemd_identity": identity,
        "dispatch_succeeded": dispatch.returncode == 0,
        "launch_attempt": 1,
        "automatic_restart_authorized": False,
        "retry_authorized": False,
    }
    _write_new_json(run_dir / "launch.dispatched.json", dispatch_body)
    if dispatch.returncode != 0:
        raise LaunchError(
            "systemd dispatch failed; reservation remains consumed and no retry is authorized"
        )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "unit_name": unit_name,
                "control_directory": str(run_dir),
                "output_root": str(output_root),
                "main_pid": identity.get("MainPID", 0),
                "restart": identity.get("Restart"),
                "dispatch_succeeded": True,
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
    report = _regular_artifact(output_root / "measurement.report.json")
    failure = _regular_artifact(output_root / "measurement.failure.json")
    body: dict[str, Any] = {
        "receipt_version": RECEIPT_VERSION,
        "launcher_version": LAUNCHER_VERSION,
        "run_id": args.run_id,
        "unit_name": args.unit_name,
        "service_pid": service_pid,
        "service_pid_start_ticks": _process_start_ticks(service_pid),
        "runner_pid": runner_pid,
        "runner_pid_start_ticks": runner_pid_start_ticks,
        "runner_started": runner_started,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "runner_returncode": runner_returncode,
        "runner_exit_status": (
            runner_returncode if runner_returncode >= 0 else None
        ),
        "runner_term_signal": (
            -runner_returncode if runner_returncode < 0 else None
        ),
        "received_service_signals": list(received_signals),
        "output_root": str(output_root),
        "formal_report": report,
        "formal_failure_receipt": failure,
        "command_hash": _payload_hash(list(command)),
        "launch_attempt": 1,
        "automatic_restart_authorized": False,
        "retry_authorized": False,
        "recover_only_authorized": False,
    }
    if failure_type is not None:
        body["wrapper_failure_type"] = failure_type
        body["wrapper_failure_message_hash"] = _payload_hash(
            failure_message or ""
        )
    return body


def _service(args: argparse.Namespace) -> int:
    control = Path(args.control_directory).resolve(strict=True)
    requested = _read_receipt(control / "launch.requested.json")
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
        or _payload_hash(command) != args.command_hash
    ):
        raise LaunchError("service arguments differ from immutable launch receipt")

    started_at = _utc_now()
    service_pid = os.getpid()
    output_preexisting = os.path.lexists(output_root)
    _write_new_json(
        control / "service.started.json",
        {
            "receipt_version": RECEIPT_VERSION,
            "launcher_version": LAUNCHER_VERSION,
            "started_at_utc": started_at,
            "run_id": args.run_id,
            "unit_name": args.unit_name,
            "service_pid": service_pid,
            "service_pid_start_ticks": _process_start_ticks(service_pid),
            "boot_id": _boot_id(),
            "output_root": str(output_root),
            "output_root_preexisting": output_preexisting,
            "command_hash": args.command_hash,
            "launch_attempt": 1,
            "automatic_restart_authorized": False,
            "retry_authorized": False,
        },
    )
    if output_preexisting:
        _write_new_json(
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
                failure_type="formal_output_root_preexisting",
                failure_message="formal output root existed at service boundary",
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
        runner_pid_start_ticks = _process_start_ticks(child.pid)
        _write_new_json(
            control / "runner.started.json",
            {
                "receipt_version": RECEIPT_VERSION,
                "launcher_version": LAUNCHER_VERSION,
                "started_at_utc": _utc_now(),
                "run_id": args.run_id,
                "unit_name": args.unit_name,
                "service_pid": service_pid,
                "runner_pid": child.pid,
                "runner_pid_start_ticks": runner_pid_start_ticks,
                "boot_id": _boot_id(),
                "output_root_preexisting": False,
                "command_hash": args.command_hash,
                "launch_attempt": 1,
                "automatic_restart_authorized": False,
                "retry_authorized": False,
            },
        )
        while True:
            try:
                returncode = child.wait()
                break
            except InterruptedError:
                continue
        _write_new_json(
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
    except (OSError, ValueError, LaunchError) as exc:
        returncode = 126
        if child is not None and child.poll() is None:
            child.terminate()
            returncode = child.wait()
        if not os.path.lexists(control / "service.exited.json"):
            _write_new_json(
                control / "service.exited.json",
                _service_exit_receipt(
                    args=args,
                    command=command,
                    service_pid=service_pid,
                    runner_pid=child.pid if child is not None else None,
                    runner_pid_start_ticks=(
                        _process_start_ticks(child.pid)
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


def monitor(control_directory: Path) -> dict[str, Any]:
    control = control_directory.expanduser().resolve(strict=True)
    requested = _read_receipt(control / "launch.requested.json")
    unit_name = str(requested["unit_name"])
    output_root = Path(str(requested["output_root"]))
    identity = _systemd_identity(unit_name)
    receipts: dict[str, Any] = {}
    for name in (
        "launch.dispatched.json",
        "service.started.json",
        "runner.started.json",
        "service.exited.json",
    ):
        path = control / name
        receipts[name] = (
            _read_receipt(path) if path.is_file() and not path.is_symlink() else None
        )
    exited = receipts["service.exited.json"]
    active = identity.get("ActiveState") in {"activating", "active", "reloading"}
    if isinstance(exited, dict):
        phase = "completed" if (
            exited.get("runner_returncode") == 0
            and exited.get("formal_report", {}).get("present") is True
        ) else "failed"
    elif active:
        phase = "running"
    elif receipts["launch.dispatched.json"] is None:
        phase = "launch_pending"
    else:
        phase = "orphaned_without_exit_receipt"
    return {
        "monitor_version": "detached_sec13f_contract_formal_once_monitor_v1",
        "observed_at_utc": _utc_now(),
        "run_id": requested["run_id"],
        "unit_name": unit_name,
        "control_directory": str(control),
        "output_root": str(output_root),
        "phase": phase,
        "systemd_identity": identity,
        "receipt_presence": {
            name: value is not None for name, value in receipts.items()
        },
        "runner_pid": (
            receipts["runner.started.json"].get("runner_pid")
            if isinstance(receipts["runner.started.json"], dict)
            else None
        ),
        "runner_returncode": (
            exited.get("runner_returncode")
            if isinstance(exited, dict)
            else None
        ),
        "formal_report": _regular_artifact(
            output_root / "measurement.report.json"
        ),
        "formal_failure_receipt": _regular_artifact(
            output_root / "measurement.failure.json"
        ),
        "stdout_stderr_log": _regular_artifact(
            Path(str(requested["stdout_stderr_log"]))
        ),
        "automatic_restart_authorized": False,
        "retry_authorized": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    launch = subparsers.add_parser("launch")
    launch.add_argument("--run-id", required=True)
    launch.add_argument("--control-base", type=Path, required=True)
    launch.add_argument("--output-root", type=Path, required=True)
    launch.add_argument("--working-directory", type=Path, required=True)
    launch.add_argument("command", nargs=argparse.REMAINDER)

    service = subparsers.add_parser("_service", help=argparse.SUPPRESS)
    service.add_argument("--control-directory", type=Path, required=True)
    service.add_argument("--run-id", required=True)
    service.add_argument("--unit-name", required=True)
    service.add_argument("--working-directory", type=Path, required=True)
    service.add_argument("--output-root", type=Path, required=True)
    service.add_argument("--command-hash", required=True)
    service.add_argument("command", nargs=argparse.REMAINDER)

    status = subparsers.add_parser("status")
    status.add_argument("--control-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "launch":
            return _launch(args)
        if args.action == "_service":
            return _service(args)
        if args.action == "status":
            print(json.dumps(monitor(args.control_directory), sort_keys=True))
            return 0
        raise AssertionError(args.action)
    except (LaunchError, FileExistsError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "launcher_version": LAUNCHER_VERSION,
                    "completed": False,
                    "error_type": type(exc).__name__,
                    "error_message_hash": _payload_hash(str(exc)),
                    "retry_authorized": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
