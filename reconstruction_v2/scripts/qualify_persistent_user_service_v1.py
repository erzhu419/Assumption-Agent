#!/usr/bin/env python3
from __future__ import annotations

"""Source-free qualification for a persistent systemd user-manager topology.

The one-shot parent is launched as a transient user service.  It launches two
independent transient child services, waits for both to finish naturally, and
refuses to succeed unless a later SSH session records that the original launch
session is gone while the same user manager and all three units remain active.

No benchmark source, item, query, document, label, qrel, action, model, GPU,
provider, API, or online evaluator is opened by this diagnostic.
"""

import argparse
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


QUALIFICATION_ID = "RQ_PERSISTENT_USER_SERVICE_P0_V1"
SCHEMA_PREFIX = "rq_persistent_user_service_p0_v1"
PARENT_UNIT = "rq-persistent-user-service-p0-v1.service"
CHILD_UNITS = {
    "coordinate": "rq-persistent-coordinate-p0-v1.service",
    "hipporag": "rq-persistent-hipporag-p0-v1.service",
}
SYSTEMD_RUN = "/usr/bin/systemd-run"
SYSTEMCTL = "/usr/bin/systemctl"
LOGINCTL = "/usr/bin/loginctl"
ENV = "/usr/bin/env"
PYTHON = "/usr/bin/python3"


class QualificationError(RuntimeError):
    """The persistent-service qualification failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_receipt(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    if "self_sha256" in body:
        raise QualificationError("receipt body supplied self_sha256")
    if os.path.lexists(path):
        raise QualificationError(f"receipt already exists: {path.name}")
    value = dict(body)
    value["self_sha256"] = _stable_hash(body)
    raw = _canonical_bytes(value) + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.lexists(path):
            raise QualificationError(f"receipt raced: {path.name}")
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)
    return value


def _read_receipt(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"receipt unavailable: {path.name}")
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii"))
    if not isinstance(value, dict):
        raise QualificationError(f"receipt is not an object: {path.name}")
    if raw != _canonical_bytes(value) + b"\n":
        raise QualificationError(f"receipt is not canonical: {path.name}")
    body = dict(value)
    declared = body.pop("self_sha256", None)
    if declared != _stable_hash(body):
        raise QualificationError(f"receipt self hash drifted: {path.name}")
    return value


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()


def _boot_monotonic_seconds() -> float:
    return float(
        Path("/proc/uptime").read_text(encoding="ascii").split()[0]
    )


def _user_manager_pid() -> int:
    uid = os.getuid()
    matches: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            command = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if (
            command
            and command[0].endswith(b"/systemd")
            and b"--user" in command[1:]
        ):
            matches.append(int(entry.name))
    if len(matches) != 1:
        raise QualificationError(
            f"expected one user systemd manager, found {len(matches)}"
        )
    return matches[0]


def _loginctl_value(property_name: str) -> str:
    completed = subprocess.run(
        [
            LOGINCTL,
            "show-user",
            str(os.getuid()),
            f"--property={property_name}",
            "--value",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise QualificationError(f"loginctl failed for {property_name}")
    return completed.stdout.strip()


def _linger() -> str:
    return _loginctl_value("Linger")


def _sessions() -> list[str]:
    raw = _loginctl_value("Sessions")
    return sorted(value for value in raw.split() if value)


def _network_denial_probe() -> dict[str, bool]:
    result: dict[str, bool] = {}
    for name, family in (("AF_INET", socket.AF_INET), ("AF_INET6", socket.AF_INET6)):
        denied = False
        try:
            probe = socket.socket(family, socket.SOCK_STREAM)
        except OSError as exc:
            denied = exc.errno in {
                errno.EAFNOSUPPORT,
                errno.EPERM,
                errno.EACCES,
            }
        else:
            probe.close()
        result[name] = denied
    if not all(result.values()):
        raise QualificationError("network address-family denial was not enforced")
    return result


def _unit_identity(unit: str) -> dict[str, Any]:
    properties = (
        "LoadState",
        "ActiveState",
        "SubState",
        "Result",
        "MainPID",
        "Type",
        "Restart",
        "KillMode",
    )
    completed = subprocess.run(
        [
            SYSTEMCTL,
            "--user",
            "show",
            unit,
            *(f"--property={name}" for name in properties),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise QualificationError(f"systemctl show failed: {unit}")
    result: dict[str, Any] = {}
    for row in completed.stdout.splitlines():
        if "=" not in row:
            continue
        key, value = row.split("=", 1)
        result[key] = int(value) if key == "MainPID" and value.isdigit() else value
    return result


def _assert_active_unit(identity: Mapping[str, Any], unit: str) -> None:
    expected = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "Type": "exec",
        "Restart": "no",
        "KillMode": "control-group",
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise QualificationError(
                f"{unit} has unexpected {key}: {identity.get(key)!r}"
            )
    if not isinstance(identity.get("MainPID"), int) or identity["MainPID"] <= 0:
        raise QualificationError(f"{unit} lacks a live MainPID")


def _clean_environment() -> list[str]:
    uid = os.getuid()
    home = str(Path.home().resolve(strict=True))
    user = os.environ.get("USER") or str(uid)
    return [
        ENV,
        "-i",
        f"HOME={home}",
        f"USER={user}",
        f"LOGNAME={user}",
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG=C.UTF-8",
        f"XDG_RUNTIME_DIR=/run/user/{uid}",
        f"DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/{uid}/bus",
        "PYTHONDONTWRITEBYTECODE=1",
    ]


def _child_command(
    *,
    script: Path,
    expected_script_sha256: str,
    work_root: Path,
    name: str,
    duration_seconds: int,
) -> list[str]:
    return [
        *_clean_environment(),
        PYTHON,
        "-B",
        str(script),
        "child",
        "--expected-script-sha256",
        expected_script_sha256,
        "--work-root",
        str(work_root),
        "--name",
        name,
        "--duration-seconds",
        str(duration_seconds),
    ]


def _launch_child(
    *,
    script: Path,
    expected_script_sha256: str,
    work_root: Path,
    name: str,
    duration_seconds: int,
) -> None:
    unit = CHILD_UNITS[name]
    command = [
        SYSTEMD_RUN,
        "--user",
        "--quiet",
        f"--unit={unit}",
        "--collect",
        "--property=Type=exec",
        "--property=Restart=no",
        "--property=KillMode=control-group",
        "--property=NoNewPrivileges=yes",
        "--property=PrivateTmp=yes",
        "--property=ProtectSystem=strict",
        "--property=ProtectHome=read-only",
        f"--property=ReadWritePaths={work_root}",
        "--property=RestrictAddressFamilies=AF_UNIX",
        "--property=UMask=0077",
        "--property=MemoryMax=256M",
        *_child_command(
            script=script,
            expected_script_sha256=expected_script_sha256,
            work_root=work_root,
            name=name,
            duration_seconds=duration_seconds,
        ),
    ]
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise QualificationError(f"failed to launch child unit {unit}")


def _verify_script(expected_sha256: str) -> Path:
    script = Path(__file__).resolve(strict=True)
    if _file_sha256(script) != expected_sha256:
        raise QualificationError("qualification script SHA-256 drifted")
    return script


def _child(args: argparse.Namespace) -> int:
    _verify_script(args.expected_script_sha256)
    work_root = Path(args.work_root).resolve(strict=True)
    name = args.name
    if name not in CHILD_UNITS:
        raise QualificationError("unknown child name")
    network_denial = _network_denial_probe()
    started = _write_new_receipt(
        work_root / f"child.{name}.started.json",
        {
            "schema": f"{SCHEMA_PREFIX}_child_started_v1",
            "qualification_id": QUALIFICATION_ID,
            "status": "child_started",
            "name": name,
            "unit": CHILD_UNITS[name],
            "started_at_utc": _utc_now(),
            "boot_id": _boot_id(),
            "user_manager_pid": _user_manager_pid(),
            "network_denial": network_denial,
            "source_item_query_document_label_qrel_action_open_count": 0,
            "model_gpu_provider_api_online_evaluator_call_count": 0,
        },
    )
    deadline = time.monotonic() + args.duration_seconds
    iterations = 0
    digest = hashlib.sha256(name.encode("ascii"))
    while time.monotonic() < deadline:
        digest.update(f"{iterations}:{name}".encode("ascii"))
        iterations += 1
        time.sleep(0.25)
    _write_new_receipt(
        work_root / f"child.{name}.completed.json",
        {
            "schema": f"{SCHEMA_PREFIX}_child_completed_v1",
            "qualification_id": QUALIFICATION_ID,
            "status": "child_completed_naturally",
            "name": name,
            "unit": CHILD_UNITS[name],
            "completed_at_utc": _utc_now(),
            "boot_id": _boot_id(),
            "user_manager_pid": _user_manager_pid(),
            "started_receipt_self_sha256": started["self_sha256"],
            "duration_seconds": args.duration_seconds,
            "iterations": iterations,
            "synthetic_digest": digest.hexdigest(),
            "source_item_query_document_label_qrel_action_open_count": 0,
            "model_gpu_provider_api_online_evaluator_call_count": 0,
        },
    )
    return 0


def _parent(args: argparse.Namespace) -> int:
    script = _verify_script(args.expected_script_sha256)
    qualification_root = Path(args.qualification_root).resolve(strict=True)
    work_root = qualification_root / "work"
    if os.path.lexists(work_root):
        raise QualificationError("one-shot work root was already consumed")
    work_root.mkdir(mode=0o700)
    started_path = work_root / "parent.started.json"
    try:
        if _linger() != "yes":
            raise QualificationError("Linger is not enabled")
        started = _write_new_receipt(
            started_path,
            {
                "schema": f"{SCHEMA_PREFIX}_parent_started_v1",
                "qualification_id": QUALIFICATION_ID,
                "status": "parent_started",
                "parent_unit": PARENT_UNIT,
                "child_units": CHILD_UNITS,
                "started_at_utc": _utc_now(),
                "boot_id": _boot_id(),
                "boot_monotonic_seconds": _boot_monotonic_seconds(),
                "user_manager_pid": _user_manager_pid(),
                "launch_sessions": _sessions(),
                "linger": "yes",
                "network_denial": _network_denial_probe(),
                "child_duration_seconds": args.child_duration_seconds,
                "observer_min_seconds": args.observer_min_seconds,
                "observer_max_seconds": args.observer_max_seconds,
                "script_sha256": args.expected_script_sha256,
                "source_item_query_document_label_qrel_action_open_count": 0,
                "model_gpu_provider_api_online_evaluator_call_count": 0,
            },
        )
        for name in CHILD_UNITS:
            _launch_child(
                script=script,
                expected_script_sha256=args.expected_script_sha256,
                work_root=work_root,
                name=name,
                duration_seconds=args.child_duration_seconds,
            )
        dispatched = _write_new_receipt(
            work_root / "children.dispatched.json",
            {
                "schema": f"{SCHEMA_PREFIX}_children_dispatched_v1",
                "qualification_id": QUALIFICATION_ID,
                "status": "two_child_units_dispatched_once",
                "dispatched_at_utc": _utc_now(),
                "parent_started_self_sha256": started["self_sha256"],
                "child_units": CHILD_UNITS,
                "launch_attempt_count": 1,
                "restart_retry_replay_count": 0,
            },
        )
        start_deadline = time.monotonic() + 20
        child_started_paths = [
            work_root / f"child.{name}.started.json" for name in CHILD_UNITS
        ]
        while not all(path.is_file() for path in child_started_paths):
            if time.monotonic() >= start_deadline:
                raise QualificationError("both child start receipts did not appear")
            time.sleep(0.1)

        deadline = (
            time.monotonic()
            + args.child_duration_seconds
            + args.observer_max_seconds
            + 30
        )
        observer_path = work_root / "detached.observer.json"
        child_completed_paths = [
            work_root / f"child.{name}.completed.json" for name in CHILD_UNITS
        ]
        while not (
            observer_path.is_file()
            and all(path.is_file() for path in child_completed_paths)
        ):
            if time.monotonic() >= deadline:
                raise QualificationError(
                    "observer or natural child completion receipt was absent"
                )
            time.sleep(0.25)

        observer = _read_receipt(observer_path)
        if observer.get("status") != (
            "observed_persistent_after_launch_session_disconnect"
        ):
            raise QualificationError("detached observer did not pass")
        if observer.get("boot_id") != started["boot_id"]:
            raise QualificationError("boot changed during qualification")
        if observer.get("user_manager_pid") != started["user_manager_pid"]:
            raise QualificationError("user manager changed during qualification")

        completions = {
            name: _read_receipt(
                work_root / f"child.{name}.completed.json"
            )["self_sha256"]
            for name in CHILD_UNITS
        }
        _write_new_receipt(
            work_root / "qualification.terminal_success.json",
            {
                "schema": f"{SCHEMA_PREFIX}_terminal_success_v1",
                "qualification_id": QUALIFICATION_ID,
                "status": "qualified_persistent_user_service_topology",
                "completed_at_utc": _utc_now(),
                "boot_id": started["boot_id"],
                "user_manager_pid": started["user_manager_pid"],
                "linger": "yes",
                "parent_started_self_sha256": started["self_sha256"],
                "children_dispatched_self_sha256": dispatched["self_sha256"],
                "observer_self_sha256": observer["self_sha256"],
                "child_completion_self_sha256": completions,
                "parent_launch_attempt_count": 1,
                "child_launch_attempt_count": 2,
                "natural_child_completion_count": 2,
                "restart_retry_replay_count": 0,
                "source_item_query_document_label_qrel_action_open_count": 0,
                "model_gpu_provider_api_online_evaluator_call_count": 0,
                "effect_study_or_cohort_consumed": False,
            },
        )
        return 0
    except BaseException as exc:
        if started_path.exists():
            failure_path = work_root / "qualification.terminal_failure.json"
            if not os.path.lexists(failure_path):
                _write_new_receipt(
                    failure_path,
                    {
                        "schema": f"{SCHEMA_PREFIX}_terminal_failure_v1",
                        "qualification_id": QUALIFICATION_ID,
                        "status": "terminal_infrastructure_qualification_failure",
                        "failed_at_utc": _utc_now(),
                        "error_type": type(exc).__name__,
                        "error_message_sha256": hashlib.sha256(
                            str(exc).encode("utf-8")
                        ).hexdigest(),
                        "restart_retry_replay_count": 0,
                        "source_item_query_document_label_qrel_action_open_count": 0,
                        "model_gpu_provider_api_online_evaluator_call_count": 0,
                    },
                )
        raise


def _observe(args: argparse.Namespace) -> int:
    _verify_script(args.expected_script_sha256)
    qualification_root = Path(args.qualification_root).resolve(strict=True)
    work_root = qualification_root / "work"
    started = _read_receipt(work_root / "parent.started.json")
    if started.get("qualification_id") != QUALIFICATION_ID:
        raise QualificationError("parent receipt qualification ID drifted")
    if _linger() != "yes":
        raise QualificationError("Linger changed before detached observation")
    current_boot_id = _boot_id()
    current_manager_pid = _user_manager_pid()
    if current_boot_id != started.get("boot_id"):
        raise QualificationError("boot changed before detached observation")
    if current_manager_pid != started.get("user_manager_pid"):
        raise QualificationError("user manager did not persist")
    elapsed = _boot_monotonic_seconds() - float(
        started["boot_monotonic_seconds"]
    )
    if not (
        float(started["observer_min_seconds"])
        <= elapsed
        <= float(started["observer_max_seconds"])
    ):
        raise QualificationError("detached observation is outside frozen window")
    launch_sessions = set(started.get("launch_sessions", []))
    observation_sessions = set(_sessions())
    launch_session_disconnected = not launch_sessions or launch_sessions.isdisjoint(
        observation_sessions
    )
    if not launch_session_disconnected:
        raise QualificationError("original launch session is still active")

    identities = {
        PARENT_UNIT: _unit_identity(PARENT_UNIT),
        **{
            unit: _unit_identity(unit)
            for unit in CHILD_UNITS.values()
        },
    }
    for unit, identity in identities.items():
        _assert_active_unit(identity, unit)
    child_starts = {
        name: _read_receipt(
            work_root / f"child.{name}.started.json"
        )["self_sha256"]
        for name in CHILD_UNITS
    }
    _write_new_receipt(
        work_root / "detached.observer.json",
        {
            "schema": f"{SCHEMA_PREFIX}_detached_observer_v1",
            "qualification_id": QUALIFICATION_ID,
            "status": "observed_persistent_after_launch_session_disconnect",
            "observed_at_utc": _utc_now(),
            "elapsed_boot_seconds": round(elapsed, 3),
            "boot_id": current_boot_id,
            "user_manager_pid": current_manager_pid,
            "linger": "yes",
            "launch_sessions": sorted(launch_sessions),
            "observation_sessions": sorted(observation_sessions),
            "launch_session_disconnected": True,
            "unit_identities": identities,
            "child_started_self_sha256": child_starts,
            "source_item_query_document_label_qrel_action_open_count": 0,
            "model_gpu_provider_api_online_evaluator_call_count": 0,
        },
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    parent = subparsers.add_parser("parent")
    parent.add_argument("--expected-script-sha256", required=True)
    parent.add_argument("--qualification-root", required=True)
    parent.add_argument("--child-duration-seconds", type=int, required=True)
    parent.add_argument("--observer-min-seconds", type=int, required=True)
    parent.add_argument("--observer-max-seconds", type=int, required=True)

    child = subparsers.add_parser("child")
    child.add_argument("--expected-script-sha256", required=True)
    child.add_argument("--work-root", required=True)
    child.add_argument("--name", choices=sorted(CHILD_UNITS), required=True)
    child.add_argument("--duration-seconds", type=int, required=True)

    observe = subparsers.add_parser("observe")
    observe.add_argument("--expected-script-sha256", required=True)
    observe.add_argument("--qualification-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "parent":
        return _parent(args)
    if args.mode == "child":
        return _child(args)
    if args.mode == "observe":
        return _observe(args)
    raise QualificationError("unknown mode")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QualificationError as exc:
        print(f"qualification error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
