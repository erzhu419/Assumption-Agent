#!/usr/bin/env python3
"""Controller for the Q5a public signer-genesis candidate ceremony.

The controller invokes the local one-shot worker exactly once per signer.  It
never opens private-key contents: recovery proves custody only from lstat
metadata, then replays the worker's public JSON receipt and public DER.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
import time
import types
from typing import Any, Final, Iterator, Mapping, NoReturn, Sequence


sys.dont_write_bytecode = True
PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
PACKAGE_ROOT: Final = PROJECT_ROOT / "src" / "hegel_machine"
if "hegel_machine" not in sys.modules:
    package = types.ModuleType("hegel_machine")
    package.__path__ = [str(PACKAGE_ROOT)]
    package.__package__ = "hegel_machine"
    sys.modules["hegel_machine"] = package

state = importlib.import_module("hegel_machine.phase3_q5a_completion_replay_v1")
wire = importlib.import_module(
    "hegel_machine.phase3_q5a_purpose5_signer_genesis_v1"
)

DEFAULT_CONTRACT: Final = PROJECT_ROOT / (
    "config/phase3_q5a_purpose5_signer_genesis_candidate_actual_v1.json"
)
PENDING_EXIT_CODE: Final = 75
NOT_AUTHORIZED_EXIT_CODE: Final = 4
TERMINAL_EXIT_CODE: Final = 2
PRIVATE_KEY_NAME: Final = "ed25519-private.pem"
PUBLIC_DER_NAME: Final = "ed25519-public.der"
WORKER_RECEIPT_NAME: Final = "purpose5-keygen-public-receipt.json"
OUTER_RECEIPT_NAME: Final = "q5a-public-receipt.json"
MAX_PRIVATE_KEY_BYTES: Final = 16 * 1024
MAX_PUBLIC_FILE_BYTES: Final = 64 * 1024
MAX_ARTIFACT_BYTES: Final = 1024 * 1024
PARTIAL_ARTIFACT_TO_TERMINAL: Final = "ARTIFACT_TO_TERMINAL"
PARTIAL_TERMINAL_TO_TRAJECTORY_SUFFIX: Final = (
    "TERMINAL_TO_TRAJECTORY_SUFFIX"
)


class ControllerError(RuntimeError):
    """Base class for classified controller outcomes."""


class ControllerPending(ControllerError):
    def __init__(self, phase: str, condition: str, detail: str) -> None:
        super().__init__(f"{phase}/{condition}: {detail}")
        self.phase = phase
        self.condition = condition
        self.detail = detail


class ControllerTerminal(ControllerError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class ControllerPartial(ControllerError):
    """The candidate is sealed and only exact finalization may continue."""

    def __init__(self, kind: str, detail: str) -> None:
        if kind not in {
            PARTIAL_ARTIFACT_TO_TERMINAL,
            PARTIAL_TERMINAL_TO_TRAJECTORY_SUFFIX,
        }:
            raise ValueError("Q5a partial-finalization kind differs")
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


class PublicResultPending(ControllerPending):
    """A worker result is valid, but its public wrapper is not yet durable."""

    def __init__(self, detail: str, receipt: Mapping[str, Any]) -> None:
        super().__init__(
            "PUBLIC_RECEIPT_PUBLICATION",
            "Q5A_PUBLIC_RECEIPT_NOT_DURABLE",
            detail,
        )
        self.receipt = dict(receipt)
        self.completed_receipts: list[dict[str, Any]] = [dict(receipt)]


def _terminal(code: str, detail: str) -> NoReturn:
    raise ControllerTerminal(code, detail)


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular(path: Path, label: str, maximum_bytes: int) -> bytes:
    try:
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise OSError("not one regular non-symlink file")
        if metadata.st_size > maximum_bytes:
            raise OSError(f"exceeds {maximum_bytes}-byte bound")
        return path.read_bytes()
    except OSError as error:
        raise ControllerPending(
            "LOCAL_READ_ONLY_REPLAY", "BOUND_LOCAL_FILE_UNAVAILABLE",
            f"{label}: {error}",
        ) from error


def _read_json_if_present(path: Path, label: str) -> dict[str, Any] | None:
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ControllerPending(
            "LOCAL_READ_ONLY_REPLAY", "BOUND_LOCAL_FILE_UNAVAILABLE",
            f"{label}: {error}",
        ) from error
    return state.strict_json_bytes(
        _read_regular(path, label, state.MAX_JSON_BYTES), label
    )


def _publish_bytes(
    path: Path, payload: bytes, label: str, *, mode: int = 0o600
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.lstat()
    except FileNotFoundError:
        pass
    except OSError as error:
        raise OSError(f"{label} existence probe failed: {error}") from error
    else:
        if _read_regular(path, label, max(len(payload), 1)) == payload:
            return
        raise OSError(f"{label} already exists with different bytes")
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    descriptor = -1
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode
        )
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        _fsync_parent(path)
        if _read_regular(path, label, max(len(payload), 1)) != payload:
            raise OSError(f"{label} readback differs")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _replace_bytes(path: Path, payload: bytes, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    descriptor = -1
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError(f"{label} short write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        _fsync_parent(path)
        if _read_regular(path, label, max(len(payload), 1)) != payload:
            raise OSError(f"{label} readback differs")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _publish_json(
    path: Path,
    value: Mapping[str, Any],
    label: str,
    *,
    mode: int = 0o600,
) -> None:
    _publish_bytes(
        path, state.canonical_json_bytes(dict(value)), label, mode=mode
    )


def _replace_json(path: Path, value: Mapping[str, Any], label: str) -> None:
    _replace_bytes(path, state.canonical_json_bytes(dict(value)), label)


@contextmanager
def _campaign_lock(contract: Any) -> Iterator[None]:
    path = Path(contract.value["fixed_paths"]["campaign_lock_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _load_contract(path: Path) -> tuple[Any, Any]:
    try:
        contract = state.load_contract(path)
        preregistration = state.load_preregistration(
            Path(contract.value["preregistration_path"])
        )
    except (OSError, ValueError) as error:
        raise ControllerPending(
            "LOCAL_CONTRACT", "Q5A_CONTRACT_OR_PREREGISTRATION_UNAVAILABLE",
            str(error),
        ) from error
    fixed = contract.value["fixed_paths"]
    worker = contract.value.get("worker")
    expected_worker = {
        "python_executable": "/usr/bin/python3.10",
        "worker_path": str(
            Path(fixed["project_root"])
            / "tools/phase3_q5a_purpose5_keygen_worker_v1.py"
        ),
        "openssl_executable": "/usr/bin/openssl",
        "expected_uid": 1000,
        "expected_gid": 1000,
        "signer_order": [1, 2, 3],
        "signer_roles": list(state.SIGNER_ROLE_NAMES),
    }
    if worker != expected_worker:
        raise ControllerPending(
            "LOCAL_CONTRACT",
            "Q5A_ACTUAL_WORKER_CONFIGURATION_DIFFERS",
            str(path),
        )
    return contract, preregistration


def _source_closure_rows(contract: Any) -> list[dict[str, Any]]:
    project_root = Path(contract.value["fixed_paths"]["project_root"])
    rows: list[dict[str, Any]] = []
    for relative in state.Q5_SOURCE_PATHS:
        path = project_root / relative
        payload = _read_regular(path, f"Q5a source {relative}", state.MAX_JSON_BYTES)
        try:
            if path.resolve(strict=True) != path:
                raise OSError("source path resolves elsewhere")
        except OSError as error:
            raise ControllerPending(
                "SOURCE_CLOSURE", "Q5A_SOURCE_PATH_IDENTITY_DIFFERS",
                f"{relative}: {error}",
            ) from error
        rows.append(
            {
                "relative_path": relative,
                "byte_count": len(payload),
                "raw_sha256": state.raw_sha256(payload),
            }
        )
    return state.validate_source_closure_rows(rows)


def _read_q4_inputs(contract: Any) -> dict[str, bytes]:
    values: dict[str, bytes] = {}
    for kind in (
        "capsule", "authorization", "consumption", "terminal", "artifact",
        "trajectory",
    ):
        maximum = (
            MAX_ARTIFACT_BYTES
            if kind == "artifact"
            else state.MAX_TRAJECTORY_BYTES
            if kind == "trajectory"
            else state.MAX_JSON_BYTES
        )
        values[kind] = _read_regular(
            Path(contract.value["q4_inputs"][kind]), f"sealed Q4 {kind}", maximum
        )
    return values


def _replay_q4(contract: Any, preregistration: Any) -> Any:
    payloads = _read_q4_inputs(contract)
    try:
        return state.replay_q4_predecessor_v1(
            preregistration.value,
            capsule_payload=payloads["capsule"],
            authorization_payload=payloads["authorization"],
            consumption_payload=payloads["consumption"],
            terminal_payload=payloads["terminal"],
            artifact_payload=payloads["artifact"],
            trajectory_payload=payloads["trajectory"],
        )
    except ValueError as error:
        raise ControllerPending(
            "SEALED_Q4_REPLAY", "Q5A_SEALED_Q4_REPLAY_DIFFERS", str(error)
        ) from error


def _worker_command(
    contract: Any, ordinal: int, *, startup_smoke: bool
) -> list[str]:
    worker = contract.value["worker"]
    command = [
        worker["python_executable"], "-I", "-S", "-B", worker["worker_path"],
    ]
    if startup_smoke:
        command.append("--startup-smoke")
    else:
        fixed = contract.value["fixed_paths"]
        command.extend(
            [
                "generate",
                "--state-root", fixed[f"signer_{ordinal}_custody_root"],
                "--receipt-root", fixed[f"signer_{ordinal}_receipt_root"],
                "--output-root", fixed[f"signer_{ordinal}_output_root"],
            ]
        )
    command.extend(
        [
            "--signer-ordinal", str(ordinal),
            "--signer-role", state.SIGNER_ROLE_NAMES[ordinal - 1],
            "--expected-uid", str(worker["expected_uid"]),
            "--expected-gid", str(worker["expected_gid"]),
        ]
    )
    return command


def _invoke_worker(command: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(command), check=False, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180,
    )


def _worker_result(command: Sequence[str], label: str) -> dict[str, Any]:
    try:
        result = _invoke_worker(command)
    except (OSError, subprocess.SubprocessError) as error:
        raise ControllerPending(
            label, "Q5A_WORKER_PROCESS_OUTCOME_UNKNOWN", str(error)
        ) from error
    if result.returncode != 0:
        detail = result.stderr.decode("ascii", errors="replace").strip()
        raise ControllerPending(label, "Q5A_WORKER_REJECTED", detail)
    if result.stderr:
        raise ControllerPending(label, "Q5A_WORKER_STDERR_NOT_EMPTY", "worker stderr")
    try:
        return state.strict_json_bytes(result.stdout, label)
    except ValueError as error:
        raise ControllerPending(label, "Q5A_WORKER_RESULT_SHAPE_DIFFERS", str(error)) from error


def _startup_smokes(
    contract: Any, q4_predecessor_receipt: Mapping[str, Any]
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for ordinal in state.REQUIRED_SIGNER_ORDINALS:
        receipts.append(_worker_result(
            _worker_command(contract, ordinal, startup_smoke=True),
            f"Q5A_SIGNER_{ordinal}_STARTUP_SMOKE",
        ))
    try:
        preflight = state.build_preflight_receipt(
            q4_predecessor_receipt,
            receipts,
            fresh_root_admission_completed=True,
        )
    except ValueError as error:
        raise ControllerPending(
            "WORKER_STARTUP_SMOKE",
            "Q5A_WORKER_STARTUP_SMOKE_DIFFERS",
            str(error),
        ) from error
    return preflight["worker_startup_smoke_receipts"]


def _future_paths(contract: Any) -> list[Path]:
    fixed = contract.value["fixed_paths"]
    paths = [
        Path(fixed[key])
        for key in ("consumption_path", "checkpoint_path", "terminal_path", "artifact_path")
    ]
    for ordinal in state.REQUIRED_SIGNER_ORDINALS:
        paths.extend(
            Path(fixed[f"signer_{ordinal}_{suffix}"])
            for suffix in ("intent_path", "custody_root", "receipt_root", "output_root")
        )
    return paths


def _path_present(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ControllerPending(
            "LOCAL_STATE_REPLAY",
            "Q5A_LOCAL_PATH_PRESENCE_UNPROVEN",
            f"{path}: {error}",
        ) from error
    return True


def _require_initial_fresh(contract: Any) -> None:
    fixed = contract.value["fixed_paths"]
    paths = [
        Path(fixed[key])
        for key in (
            "run_root",
            "ledger_root",
            "capsule_path",
            "authorization_path",
            "pending_path",
            "trajectory_path",
        )
    ]
    paths.extend(_future_paths(contract))
    for path in paths:
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ControllerPending(
                "FRESH_ROOT_ADMISSION",
                "Q5A_INITIAL_ROOT_ABSENCE_UNPROVEN",
                f"{path}: {error}",
            ) from error
        raise ControllerPending(
            "FRESH_ROOT_ADMISSION",
            "Q5A_INITIAL_ROOT_ALREADY_EXISTS",
            str(path),
        )


def _require_preconsumption_fresh(contract: Any) -> None:
    for path in _future_paths(contract):
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ControllerPending(
                "FRESH_ROOT_ADMISSION", "Q5A_FUTURE_ROOT_ABSENCE_UNPROVEN",
                f"{path}: {error}",
            ) from error
        raise ControllerPending(
            "FRESH_ROOT_ADMISSION", "Q5A_FUTURE_ROOT_ALREADY_EXISTS", str(path)
        )


def _preconsumption_replay(
    contract: Any, preregistration: Any, capsule: Mapping[str, Any]
) -> None:
    rows = _source_closure_rows(contract)
    if (
        rows != capsule["source_closure_rows"]
        or state.source_closure_root(rows) != capsule["source_closure_root"]
    ):
        raise ControllerPending(
            "SOURCE_CLOSURE", "Q5A_CAPSULE_SOURCE_CLOSURE_DIFFERS",
            capsule["source_closure_root"],
        )
    replay = _replay_q4(contract, preregistration)
    smokes = _startup_smokes(contract, replay.preflight_receipt)
    _require_preconsumption_fresh(contract)
    expected = state.build_preflight_receipt(
        replay.preflight_receipt, smokes, fresh_root_admission_completed=True
    )
    if expected != capsule["preflight_receipt"]:
        raise ControllerPending(
            "PREFLIGHT_REPLAY", "Q5A_CAPSULE_PREFLIGHT_RECEIPT_DIFFERS",
            capsule["capsule_digest"],
        )


def _append_trajectory(contract: Any, event: Mapping[str, Any]) -> None:
    path = Path(contract.value["fixed_paths"]["trajectory_path"])
    payload = state.canonical_json_bytes(dict(event))
    existing = b""
    if _path_present(path):
        existing = _read_regular(
            path, "Q5a trajectory", state.MAX_TRAJECTORY_BYTES
        )
    combined = existing + payload
    if len(combined) > state.MAX_TRAJECTORY_BYTES:
        raise OSError("Q5a trajectory exceeds its byte bound")
    # Replace the complete verified prefix plus one new row atomically.  An
    # interruption can therefore leave only the old complete prefix, never a
    # partial JSON line that would make deterministic suffix repair impossible.
    _replace_bytes(path, combined, "Q5a trajectory")


def _event(
    name: str, capsule: Mapping[str, Any], *, authorization: Mapping[str, Any] | None = None,
    consumption: Mapping[str, Any] | None = None,
    receipts: Sequence[Mapping[str, Any]] = (),
    terminal: Mapping[str, Any] | None = None,
    time_ns: int | None = None,
) -> dict[str, Any]:
    audit = state.q5a_action_audit(tuple(range(1, len(receipts) + 1)))
    if terminal is not None:
        audit = terminal["outcome"]["q5a_action_audit"]
    event = {
        "schema_version": "hegel-phase3a-q5a-trajectory/1",
        "event": name,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": None if authorization is None else authorization["authorization_digest"],
        "consumption_record_digest": None if consumption is None else consumption["record_digest"],
        "completed_prefix": list(range(1, len(receipts) + 1)),
        "authorization_consumption_count": 0 if consumption is None else 1,
        "logical_campaign_count": 0 if consumption is None else 1,
        "q5a_action_audit": audit,
        "time_ns": time.time_ns() if time_ns is None else time_ns,
    }
    if terminal is not None:
        outcome = terminal["outcome"]
        event.update(
            {
                "terminal_record_digest": terminal["record_digest"],
                "candidate_root": outcome.get("candidate_root"),
                "failure_code": outcome.get("failure_code"),
            }
        )
    return event


def _trajectory_expected(
    capsule: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any] | None = None,
    consumption: Mapping[str, Any] | None = None,
    receipts: Sequence[Mapping[str, Any]] = (),
    terminal: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    events = [_event("CAPSULE_PREPARED_NOT_AUTHORIZED", capsule, time_ns=0)]
    if authorization is not None:
        events.append(
            _event(
                "AUTHORIZATION_RECORDED_NOT_CONSUMED",
                capsule,
                authorization=authorization,
                time_ns=0,
            )
        )
    if consumption is not None:
        events.append(
            _event(
                "Q5A_AUTHORIZATION_CONSUMED",
                capsule,
                authorization=authorization,
                consumption=consumption,
                time_ns=0,
            )
        )
        for ordinal in range(1, len(receipts) + 1):
            events.append(
                _event(
                    f"SIGNER_{ordinal}_PUBLIC_RECEIPT_COMPLETE",
                    capsule,
                    authorization=authorization,
                    consumption=consumption,
                    receipts=receipts[:ordinal],
                    time_ns=0,
                )
            )
    if terminal is not None:
        events.append(
            _event(
                terminal["status"],
                capsule,
                authorization=authorization,
                consumption=consumption,
                receipts=receipts,
                terminal=terminal,
                time_ns=0,
            )
        )
    return events


def _event_without_time(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    observed = result.pop("time_ns", None)
    if type(observed) is not int or observed < 0:
        raise ValueError("trajectory time_ns differs")
    return result


def _reconcile_trajectory(
    contract: Any,
    capsule: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any] | None = None,
    consumption: Mapping[str, Any] | None = None,
    receipts: Sequence[Mapping[str, Any]] = (),
    terminal: Mapping[str, Any] | None = None,
) -> None:
    path = Path(contract.value["fixed_paths"]["trajectory_path"])
    expected = _trajectory_expected(
        capsule,
        authorization=authorization,
        consumption=consumption,
        receipts=receipts,
        terminal=terminal,
    )
    try:
        if _path_present(path):
            payload = _read_regular(
                path, "Q5a trajectory", state.MAX_TRAJECTORY_BYTES
            )
            raw_lines = payload.splitlines(keepends=True)
            if b"".join(raw_lines) != payload:
                raise ValueError("trajectory line framing differs")
            existing = [
                state.strict_json_bytes(line, f"Q5a trajectory event {index}")
                for index, line in enumerate(raw_lines, 1)
            ]
        else:
            existing = []
        if len(existing) > len(expected):
            raise ValueError("trajectory leads durable state")
        for observed, wanted in zip(existing, expected, strict=False):
            if _event_without_time(observed) != _event_without_time(wanted):
                raise ValueError("trajectory is not the exact durable-state prefix")
        for wanted in expected[len(existing):]:
            wanted["time_ns"] = time.time_ns()
            _append_trajectory(contract, wanted)
    except ValueError as error:
        _terminal("Q5A_TRAJECTORY_REPLAY_DIFFERS", str(error))
    except (OSError, ControllerPending) as error:
        detail = str(error)
        if terminal is not None:
            raise ControllerPartial(
                PARTIAL_TERMINAL_TO_TRAJECTORY_SUFFIX,
                f"terminal is durable but trajectory finalization failed: {detail}",
            ) from error
        raise ControllerPending(
            "TRAJECTORY_RECONCILIATION",
            "Q5A_TRAJECTORY_SUFFIX_NOT_DURABLE",
            detail,
        ) from error


def _clear_pending(contract: Any) -> None:
    path = Path(contract.value["fixed_paths"]["pending_path"])
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_parent(path)


def _persist_pending(
    contract: Any, preregistration: Any, capsule: Mapping[str, Any],
    error: ControllerPending, *, authorization: Mapping[str, Any] | None = None,
    consumption: Mapping[str, Any] | None = None,
    receipts: Sequence[Mapping[str, Any]] = (),
) -> None:
    pending = state.build_pending(
        contract, preregistration.value, capsule,
        phase=error.phase, condition=error.condition, detail=error.detail,
        authorization=authorization, consumption=consumption,
        completed_receipts=receipts,
    )
    _replace_json(
        Path(contract.value["fixed_paths"]["pending_path"]), pending, "Q5a pending"
    )


def _execution_state_exists(contract: Any) -> bool:
    return any(_path_present(path) for path in _future_paths(contract))


def _read_capsule(
    contract: Any, preregistration: Any, digest: str
) -> dict[str, Any]:
    path = Path(contract.value["fixed_paths"]["capsule_path"])
    value = _read_json_if_present(path, "Q5a capsule")
    if value is None:
        raise ControllerPending("CAPSULE_REPLAY", "Q5A_CAPSULE_UNAVAILABLE", str(path))
    try:
        checked = state.validate_capsule(value, contract, preregistration.value)
    except ValueError as error:
        raise ControllerPending(
            "CAPSULE_REPLAY", "Q5A_CAPSULE_REPLAY_DIFFERS", str(error)
        ) from error
    if checked["capsule_digest"] != digest:
        raise ControllerPending(
            "CAPSULE_REPLAY", "Q5A_CAPSULE_DIGEST_DIFFERS", digest
        )
    return checked


def _read_authorization(
    contract: Any, preregistration: Any, capsule: Mapping[str, Any]
) -> dict[str, Any] | None:
    value = _read_json_if_present(
        Path(contract.value["fixed_paths"]["authorization_path"]),
        "Q5a authorization",
    )
    if value is None:
        return None
    try:
        return state.validate_authorization(
            value, contract, preregistration.value, capsule
        )
    except ValueError as error:
        _terminal("Q5A_AUTHORIZATION_REPLAY_DIFFERS", str(error))


def _read_consumption(
    contract: Any, preregistration: Any, capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any] | None:
    value = _read_json_if_present(
        Path(contract.value["fixed_paths"]["consumption_path"]), "Q5a consumption"
    )
    if value is None:
        return None
    try:
        return state.validate_consumption(
            value, contract, preregistration.value, capsule, authorization
        )
    except ValueError as error:
        _terminal("Q5A_CONSUMPTION_REPLAY_DIFFERS", str(error))


def prepare_capsule(
    contract_path: Path, source_commit: str, *, run_id: str | None = None,
    created_at_unix_seconds: int | None = None,
) -> dict[str, Any]:
    contract, preregistration = _load_contract(contract_path)
    fixed = contract.value["fixed_paths"]
    capsule_path = Path(fixed["capsule_path"])
    with _campaign_lock(contract):
        existing = _read_json_if_present(capsule_path, "Q5a capsule")
        if existing is not None:
            try:
                capsule = state.validate_capsule(
                    existing, contract, preregistration.value
                )
            except ValueError as error:
                if _execution_state_exists(contract):
                    _terminal("Q5A_CAPSULE_REPLAY_FAILED_AFTER_EXECUTION_STATE", str(error))
                raise ControllerPending(
                    "CAPSULE_REPLAY", "Q5A_CAPSULE_REPLAY_DIFFERS", str(error)
                ) from error
            if _execution_state_exists(contract):
                return capsule
            try:
                _preconsumption_replay(contract, preregistration, capsule)
                authorization = _read_authorization(
                    contract, preregistration, capsule
                )
                _reconcile_trajectory(
                    contract, capsule, authorization=authorization
                )
            except ControllerPending as error:
                authorization = _read_authorization(
                    contract, preregistration, capsule
                )
                _persist_pending(
                    contract, preregistration, capsule, error,
                    authorization=authorization,
                )
                raise
            _clear_pending(contract)
            return capsule

        _require_initial_fresh(contract)
        rows = _source_closure_rows(contract)
        replay = _replay_q4(contract, preregistration)
        smokes = _startup_smokes(contract, replay.preflight_receipt)
        preflight = state.build_preflight_receipt(
            replay.preflight_receipt, smokes, fresh_root_admission_completed=True
        )
        capsule = state.build_capsule(
            contract,
            preregistration.value,
            {
                "source_commit": source_commit,
                "source_closure_rows": rows,
                "source_closure_root": state.source_closure_root(rows),
                "run_id": run_id or secrets.token_hex(16),
                "created_at_unix_seconds": int(time.time()) if created_at_unix_seconds is None else created_at_unix_seconds,
                "preflight_receipt": preflight,
            },
        )
        _publish_json(capsule_path, capsule, "Q5a capsule")
        try:
            _reconcile_trajectory(contract, capsule)
        except ControllerPending as error:
            _persist_pending(contract, preregistration, capsule, error)
            raise
        return capsule


def record_authorization(contract_path: Path, capsule_digest: str) -> dict[str, Any]:
    contract, preregistration = _load_contract(contract_path)
    fixed = contract.value["fixed_paths"]
    with _campaign_lock(contract):
        execution = _execution_state_exists(contract)
        try:
            capsule = _read_capsule(contract, preregistration, capsule_digest)
            existing = _read_authorization(contract, preregistration, capsule)
        except ControllerPending as error:
            if execution:
                _terminal("Q5A_AUTHORIZATION_REPLAY_FAILED_AFTER_EXECUTION_STATE", str(error))
            raise
        if execution:
            if existing is None:
                _terminal("Q5A_AUTHORIZATION_MISSING_AFTER_EXECUTION_STATE", capsule_digest)
            return existing
        try:
            _preconsumption_replay(contract, preregistration, capsule)
        except ControllerPending as error:
            _persist_pending(
                contract, preregistration, capsule, error,
                authorization=existing,
            )
            raise
        if existing is not None:
            try:
                _reconcile_trajectory(
                    contract, capsule, authorization=existing
                )
            except ControllerPending as error:
                _persist_pending(
                    contract,
                    preregistration,
                    capsule,
                    error,
                    authorization=existing,
                )
                raise
            _clear_pending(contract)
            return existing
        authorization = state.build_authorization(
            contract, preregistration.value, capsule
        )
        _publish_json(Path(fixed["authorization_path"]), authorization, "Q5a authorization")
        try:
            _reconcile_trajectory(
                contract, capsule, authorization=authorization
            )
        except ControllerPending as error:
            _persist_pending(
                contract,
                preregistration,
                capsule,
                error,
                authorization=authorization,
            )
            raise
        _clear_pending(contract)
        return authorization


def _root_metadata(path: Path, label: str, expected_uid: int, expected_gid: int) -> None:
    try:
        metadata = path.lstat()
        if (
            path.is_symlink() or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != expected_uid or metadata.st_gid != expected_gid
        ):
            raise OSError("root metadata differs")
    except OSError as error:
        _terminal("Q5A_PUBLIC_RESULT_ROOT_REPLAY_DIFFERS", f"{label}: {error}")


def _private_custody_metadata(
    path: Path, expected_uid: int, expected_gid: int
) -> None:
    """Prove custody metadata without opening the private-key file."""
    try:
        metadata = path.lstat()
        if (
            path.is_symlink() or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != expected_uid or metadata.st_gid != expected_gid
            or metadata.st_nlink != 1
            or not (0 < metadata.st_size <= MAX_PRIVATE_KEY_BYTES)
        ):
            raise OSError("private custody metadata differs")
    except OSError as error:
        _terminal("Q5A_PRIVATE_CUSTODY_METADATA_DIFFERS", str(error))


def _read_public_regular(
    path: Path,
    label: str,
    expected_uid: int,
    expected_gid: int,
) -> bytes:
    try:
        metadata = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o644
            or metadata.st_uid != expected_uid
            or metadata.st_gid != expected_gid
            or metadata.st_nlink != 1
        ):
            raise OSError("public file metadata differs")
    except OSError as error:
        raise ControllerTerminal(
            "Q5A_PUBLIC_FILE_METADATA_DIFFERS", f"{label}: {error}"
        ) from error
    return _read_regular(path, label, MAX_PUBLIC_FILE_BYTES)


def _public_paths(contract: Any, ordinal: int) -> dict[str, Path]:
    fixed = contract.value["fixed_paths"]
    custody = Path(fixed[f"signer_{ordinal}_custody_root"])
    receipt = Path(fixed[f"signer_{ordinal}_receipt_root"])
    output = Path(fixed[f"signer_{ordinal}_output_root"])
    return {
        "intent": Path(fixed[f"signer_{ordinal}_intent_path"]),
        "custody": custody,
        "receipt_root": receipt,
        "output_root": output,
        "private": custody / PRIVATE_KEY_NAME,
        "worker_receipt": receipt / WORKER_RECEIPT_NAME,
        "outer_receipt": receipt / OUTER_RECEIPT_NAME,
        "public_der": output / PUBLIC_DER_NAME,
    }


def _read_intent(
    contract: Any, preregistration: Any, capsule: Mapping[str, Any],
    authorization: Mapping[str, Any], consumption: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]], ordinal: int,
) -> dict[str, Any] | None:
    path = _public_paths(contract, ordinal)["intent"]
    value = _read_json_if_present(path, f"Q5a signer {ordinal} intent")
    if value is None:
        return None
    try:
        return state.validate_keygen_intent(
            value, contract, preregistration.value, capsule, authorization,
            consumption, receipts,
        )
    except ValueError as error:
        _terminal("Q5A_KEYGEN_INTENT_REPLAY_DIFFERS", str(error))


def _recover_public_receipt(
    contract: Any, preregistration: Any, capsule: Mapping[str, Any],
    authorization: Mapping[str, Any], consumption: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]], intent: Mapping[str, Any],
) -> dict[str, Any]:
    ordinal = intent["signer_ordinal"]
    paths = _public_paths(contract, ordinal)
    worker = contract.value["worker"]
    _root_metadata(paths["custody"], "custody", worker["expected_uid"], worker["expected_gid"])
    _root_metadata(paths["receipt_root"], "receipt", worker["expected_uid"], worker["expected_gid"])
    _root_metadata(paths["output_root"], "output", worker["expected_uid"], worker["expected_gid"])
    _private_custody_metadata(paths["private"], worker["expected_uid"], worker["expected_gid"])
    try:
        worker_raw = _read_public_regular(
            paths["worker_receipt"], f"Q5a signer {ordinal} worker receipt",
            worker["expected_uid"], worker["expected_gid"],
        )
        worker_receipt = state.strict_json_bytes(
            worker_raw, f"Q5a signer {ordinal} worker receipt"
        )
        der = _read_public_regular(
            paths["public_der"], f"Q5a signer {ordinal} public DER",
            worker["expected_uid"], worker["expected_gid"],
        )
        receipt = state.build_public_receipt(
            contract, preregistration.value, capsule, authorization, consumption,
            receipts, intent, worker_receipt=worker_receipt,
            worker_public_receipt_raw_sha256=state.raw_sha256(worker_raw),
            public_key_der_raw_sha256=state.raw_sha256(der),
        )
    except (ControllerPending, ControllerTerminal, ValueError) as error:
        _terminal("Q5A_PUBLIC_WORKER_RESULT_REPLAY_DIFFERS", str(error))
    try:
        existing = _read_json_if_present(
            paths["outer_receipt"], f"Q5a signer {ordinal} public receipt"
        )
    except ControllerPending as error:
        raise PublicResultPending(str(error), receipt) from error
    if existing is not None and existing != receipt:
        _terminal("Q5A_OUTER_PUBLIC_RECEIPT_REPLAY_DIFFERS", str(paths["outer_receipt"]))
    if existing is None:
        try:
            _publish_json(
                paths["outer_receipt"],
                receipt,
                "Q5a public receipt",
                mode=0o644,
            )
        except OSError as error:
            try:
                replay = _read_json_if_present(
                    paths["outer_receipt"],
                    f"Q5a signer {ordinal} public receipt after publication error",
                )
            except ControllerPending:
                replay = None
            if replay == receipt:
                return receipt
            raise PublicResultPending(str(error), receipt) from error
    return receipt


def _any_worker_state(paths: Mapping[str, Path]) -> bool:
    return any(
        _path_present(paths[key])
        for key in (
            "custody", "receipt_root", "output_root", "private",
            "worker_receipt", "outer_receipt", "public_der",
        )
    )


def _recover_prefix(
    contract: Any, preregistration: Any, capsule: Mapping[str, Any],
    authorization: Mapping[str, Any], consumption: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str | None]:
    receipts: list[dict[str, Any]] = []
    unresolved: dict[str, Any] | None = None
    unresolved_detail: str | None = None
    for ordinal in state.REQUIRED_SIGNER_ORDINALS:
        paths = _public_paths(contract, ordinal)
        intent = _read_intent(
            contract, preregistration, capsule, authorization, consumption,
            receipts, ordinal,
        )
        if intent is None:
            if _any_worker_state(paths):
                _terminal("Q5A_WORKER_STATE_WITHOUT_INTENT", str(ordinal))
            break
        if not _any_worker_state(paths):
            unresolved = intent
            unresolved_detail = "durable intent exists without any worker result"
            break
        try:
            receipt = _recover_public_receipt(
                contract, preregistration, capsule, authorization, consumption,
                receipts, intent,
            )
        except PublicResultPending as error:
            error.completed_receipts = [*receipts, error.receipt]
            raise
        except (ControllerPending, ControllerTerminal, ValueError) as error:
            unresolved = intent
            unresolved_detail = str(error)
            break
        receipts.append(receipt)
    for ordinal in range(len(receipts) + 2, 4):
        paths = _public_paths(contract, ordinal)
        if _path_present(paths["intent"]) or _any_worker_state(paths):
            _terminal("Q5A_NONPREFIX_SIGNER_STATE", str(ordinal))
    return receipts, unresolved, unresolved_detail


def _checkpoint(
    contract: Any, preregistration: Any, capsule: Mapping[str, Any],
    authorization: Mapping[str, Any], consumption: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = state.build_checkpoint(
        contract, preregistration.value, capsule, authorization, consumption,
        receipts,
    )
    path = Path(contract.value["fixed_paths"]["checkpoint_path"])
    try:
        existing = _read_json_if_present(path, "Q5a checkpoint")
    except ControllerPending as error:
        raise ControllerPending(
            "CLEAN_CHECKPOINT_REPLAY",
            "Q5A_CHECKPOINT_UNAVAILABLE",
            str(error),
        ) from error
    if existing is not None:
        prefix = existing.get("completed_prefix")
        if type(prefix) is not list or len(prefix) > len(receipts):
            raise ControllerPending(
                "CLEAN_CHECKPOINT_REPLAY",
                "Q5A_CHECKPOINT_REPLAY_DIFFERS",
                "checkpoint leads public receipts",
            )
        try:
            state.validate_checkpoint(
                existing, contract, preregistration.value, capsule,
                authorization, consumption, receipts[: len(prefix)],
            )
        except ValueError as error:
            raise ControllerPending(
                "CLEAN_CHECKPOINT_REPLAY",
                "Q5A_CHECKPOINT_REPLAY_DIFFERS",
                str(error),
            ) from error
    if existing != expected:
        try:
            _replace_json(path, expected, "Q5a checkpoint")
        except OSError as error:
            raise ControllerPending(
                "CLEAN_CHECKPOINT_PUBLICATION",
                "Q5A_CHECKPOINT_NOT_DURABLE",
                str(error),
            ) from error
    return expected


def _admit_next_roots(contract: Any, ordinal: int) -> None:
    paths = _public_paths(contract, ordinal)
    for key in ("intent", "custody", "receipt_root", "output_root"):
        path = paths[key]
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ControllerPending(
                "NEXT_SIGNER_ROOT_ADMISSION", "Q5A_NEXT_SIGNER_ROOT_ABSENCE_UNPROVEN",
                f"{path}: {error}",
            ) from error
        raise ControllerPending(
            "NEXT_SIGNER_ROOT_ADMISSION", "Q5A_NEXT_SIGNER_ROOT_ALREADY_EXISTS",
            str(path),
        )


def _seal_invalidated(
    contract: Any, preregistration: Any, capsule: Mapping[str, Any],
    authorization: Mapping[str, Any], consumption: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    intent: Mapping[str, Any] | None,
    *,
    phase: str | None = None,
    code: str,
    detail: str,
) -> dict[str, Any]:
    failure_phase = phase
    if failure_phase is None:
        failure_phase = (
            "FINALIZATION"
            if intent is None
            else f"SIGNER_{intent['signer_ordinal']}_WORKER"
        )
    terminal = state.build_invalidated_terminal(
        contract, preregistration.value, capsule, authorization, consumption,
        receipts, failure_phase=failure_phase,
        failure_code=code, failure_detail=detail, unknown_intent=intent,
    )
    try:
        _publish_json(
            Path(contract.value["fixed_paths"]["terminal_path"]), terminal,
            "Q5a invalidated terminal",
        )
    except OSError as error:
        _terminal(
            "Q5A_INVALIDATED_TERMINAL_PUBLICATION_FAILED_NO_RESUME",
            str(error),
        )
    _reconcile_trajectory(
        contract,
        capsule,
        authorization=authorization,
        consumption=consumption,
        receipts=receipts,
        terminal=terminal,
    )
    _clear_pending(contract)
    return terminal


def _read_terminal(
    contract: Any, preregistration: Any, capsule: Mapping[str, Any],
    authorization: Mapping[str, Any], consumption: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]], unresolved: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    path = Path(contract.value["fixed_paths"]["terminal_path"])
    try:
        value = _read_json_if_present(path, "Q5a terminal")
    except ControllerPending as error:
        _terminal("Q5A_TERMINAL_REPLAY_UNAVAILABLE", str(error))
    if value is None:
        return None
    artifact: bytes | None = None
    unknown: Mapping[str, Any] | None = None
    if value.get("status") == wire.Q5A_SUCCESS_STATUS:
        try:
            artifact = _read_regular(
                Path(contract.value["fixed_paths"]["artifact_path"]),
                "Q5a candidate artifact", MAX_ARTIFACT_BYTES,
            )
        except ControllerPending as error:
            _terminal("Q5A_TERMINAL_ARTIFACT_REPLAY_DIFFERS", str(error))
    elif value.get("status") == state.INVALIDATED_TERMINAL_NO_RESUME:
        unknown = unresolved
    try:
        return state.validate_terminal(
            value, contract, preregistration.value, capsule, authorization,
            consumption, receipts, artifact_payload=artifact,
            unknown_intent=unknown,
        )
    except ValueError as error:
        _terminal("Q5A_TERMINAL_REPLAY_DIFFERS", str(error))


def _not_authorized(capsule: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "hegel-phase3a-q5a-controller-not-authorized/1",
        "status": state.NOT_AUTHORIZED,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_consumption_count": 0,
        "logical_campaign_count": 0,
        "q5a_action_audit": state.ZERO_ACTION_AUDIT,
    }


def run_v1(contract_path: Path, capsule_digest: str) -> dict[str, Any]:
    contract, preregistration = _load_contract(contract_path)
    fixed = contract.value["fixed_paths"]
    with _campaign_lock(contract):
        execution = _execution_state_exists(contract)
        try:
            capsule = _read_capsule(contract, preregistration, capsule_digest)
            authorization = _read_authorization(contract, preregistration, capsule)
        except ControllerPending as error:
            if execution:
                _terminal("Q5A_PREFIX_REPLAY_FAILED_AFTER_EXECUTION_STATE", str(error))
            raise
        if authorization is None:
            if execution:
                _terminal("Q5A_AUTHORIZATION_MISSING_AFTER_EXECUTION_STATE", capsule_digest)
            return _not_authorized(capsule)

        consumption_path = Path(fixed["consumption_path"])
        consumption_was_present = _path_present(consumption_path)
        try:
            consumption = _read_consumption(
                contract, preregistration, capsule, authorization
            )
        except ControllerPending as error:
            if consumption_was_present:
                _terminal("Q5A_CONSUMPTION_REPLAY_UNAVAILABLE", str(error))
            raise
        if consumption is None:
            if execution:
                _terminal("Q5A_CONSUMPTION_MISSING_AFTER_EXECUTION_STATE", capsule_digest)
            try:
                _preconsumption_replay(contract, preregistration, capsule)
            except ControllerPending as error:
                _persist_pending(
                    contract, preregistration, capsule, error,
                    authorization=authorization,
                )
                raise
            consumption = state.build_consumption(
                contract, preregistration.value, capsule, authorization
            )
            try:
                _publish_json(consumption_path, consumption, "Q5a consumption")
            except OSError as error:
                try:
                    durable = _read_consumption(
                        contract, preregistration, capsule, authorization
                    )
                except (ControllerPending, ControllerTerminal):
                    durable = None
                if durable is None and not _path_present(consumption_path):
                    pending = ControllerPending(
                        "CONSUMPTION_PUBLICATION", "Q5A_CONSUMPTION_NOT_DURABLE",
                        str(error),
                    )
                    _persist_pending(
                        contract, preregistration, capsule, pending,
                        authorization=authorization,
                    )
                    raise pending
                if durable is None:
                    _terminal(
                        "Q5A_CONSUMPTION_PUBLICATION_OUTCOME_UNKNOWN",
                        str(error),
                    )
                consumption = durable
            _clear_pending(contract)

        try:
            receipts, unresolved, unresolved_detail = _recover_prefix(
                contract, preregistration, capsule, authorization, consumption
            )
        except PublicResultPending as error:
            _persist_pending(
                contract,
                preregistration,
                capsule,
                error,
                authorization=authorization,
                consumption=consumption,
                receipts=error.completed_receipts,
            )
            raise
        except ControllerPending as error:
            _terminal("Q5A_POSTCONSUMPTION_PREFIX_REPLAY_UNAVAILABLE", str(error))
        terminal = _read_terminal(
            contract, preregistration, capsule, authorization, consumption,
            receipts, unresolved,
        )
        if terminal is not None:
            _reconcile_trajectory(
                contract,
                capsule,
                authorization=authorization,
                consumption=consumption,
                receipts=receipts,
                terminal=terminal,
            )
            return terminal
        if unresolved is not None:
            return _seal_invalidated(
                contract, preregistration, capsule, authorization, consumption,
                receipts, unresolved, code="Q5A_KEYGEN_OUTCOME_UNKNOWN",
                detail=(
                    unresolved_detail
                    or "durable intent exists without a complete public result"
                ),
            )

        try:
            _reconcile_trajectory(
                contract,
                capsule,
                authorization=authorization,
                consumption=consumption,
                receipts=receipts,
            )
            _checkpoint(
                contract, preregistration, capsule, authorization, consumption,
                receipts,
            )
        except ControllerPending as error:
            _persist_pending(
                contract,
                preregistration,
                capsule,
                error,
                authorization=authorization,
                consumption=consumption,
                receipts=receipts,
            )
            raise
        artifact_path = Path(fixed["artifact_path"])
        if _path_present(artifact_path):
            if len(receipts) != 3:
                _terminal("Q5A_ARTIFACT_EXISTS_BEFORE_COMPLETE_PREFIX", str(len(receipts)))
            try:
                artifact = _read_regular(
                    artifact_path, "Q5a candidate artifact", MAX_ARTIFACT_BYTES
                )
                state.validate_artifact_binding(artifact, capsule, receipts)
            except ControllerPending as error:
                pending = ControllerPending(
                    "CANDIDATE_ARTIFACT_REPLAY",
                    "Q5A_CANDIDATE_ARTIFACT_TEMPORARILY_UNAVAILABLE",
                    str(error),
                )
                _persist_pending(
                    contract,
                    preregistration,
                    capsule,
                    pending,
                    authorization=authorization,
                    consumption=consumption,
                    receipts=receipts,
                )
                raise pending
            except ValueError as error:
                _terminal("Q5A_ARTIFACT_REPLAY_DIFFERS", str(error))
            terminal = state.build_complete_terminal(
                contract, preregistration.value, capsule, authorization,
                consumption, receipts, artifact,
            )
            try:
                _publish_json(
                    Path(fixed["terminal_path"]), terminal, "Q5a terminal"
                )
            except OSError as error:
                raise ControllerPartial(
                    PARTIAL_ARTIFACT_TO_TERMINAL, str(error)
                ) from error
            _reconcile_trajectory(
                contract,
                capsule,
                authorization=authorization,
                consumption=consumption,
                receipts=receipts,
                terminal=terminal,
            )
            _clear_pending(contract)
            return terminal

        while len(receipts) < 3:
            ordinal = len(receipts) + 1
            try:
                _admit_next_roots(contract, ordinal)
            except ControllerPending as error:
                _persist_pending(
                    contract, preregistration, capsule, error,
                    authorization=authorization, consumption=consumption,
                    receipts=receipts,
                )
                raise
            intent = state.build_keygen_intent(
                contract, preregistration.value, capsule, authorization,
                consumption, receipts, created_at_unix_seconds=int(time.time()),
            )
            intent_path = _public_paths(contract, ordinal)["intent"]
            try:
                _publish_json(
                    intent_path, intent, f"Q5a signer {ordinal} intent"
                )
            except OSError as error:
                try:
                    durable_intent = _read_intent(
                        contract,
                        preregistration,
                        capsule,
                        authorization,
                        consumption,
                        receipts,
                        ordinal,
                    )
                except (ControllerPending, ControllerTerminal):
                    durable_intent = None
                if durable_intent is None and not _path_present(intent_path):
                    pending = ControllerPending(
                        "KEYGEN_INTENT_PUBLICATION",
                        "Q5A_KEYGEN_INTENT_NOT_DURABLE",
                        str(error),
                    )
                    _persist_pending(
                        contract,
                        preregistration,
                        capsule,
                        pending,
                        authorization=authorization,
                        consumption=consumption,
                        receipts=receipts,
                    )
                    raise pending
                if durable_intent is None:
                    _terminal(
                        "Q5A_KEYGEN_INTENT_PUBLICATION_OUTCOME_UNKNOWN",
                        str(error),
                    )
                intent = durable_intent
            worker_error: ControllerPending | None = None
            try:
                returned = _worker_result(
                    _worker_command(contract, ordinal, startup_smoke=False),
                    f"Q5A_SIGNER_{ordinal}_WORKER",
                )
            except ControllerPending as error:
                worker_error = error
                returned = None
            try:
                receipt = _recover_public_receipt(
                    contract, preregistration, capsule, authorization,
                    consumption, receipts, intent,
                )
            except PublicResultPending as error:
                error.completed_receipts = [*receipts, error.receipt]
                _persist_pending(
                    contract,
                    preregistration,
                    capsule,
                    error,
                    authorization=authorization,
                    consumption=consumption,
                    receipts=error.completed_receipts,
                )
                raise
            except (ControllerPending, ControllerTerminal, ValueError) as error:
                return _seal_invalidated(
                    contract, preregistration, capsule, authorization,
                    consumption, receipts, intent,
                    code=(
                        worker_error.condition
                        if worker_error is not None
                        else "Q5A_KEYGEN_OUTCOME_UNKNOWN"
                    ),
                    detail=(
                        worker_error.detail
                        if worker_error is not None
                        else str(error)
                    ),
                )
            if (
                returned is not None
                and returned != receipt["worker_public_receipt"]
            ):
                return _seal_invalidated(
                    contract, preregistration, capsule, authorization,
                    consumption, receipts, intent,
                    code="Q5A_WORKER_STDOUT_DIFFERS_FROM_DURABLE_RECEIPT",
                    detail=f"signer {ordinal}",
                )
            receipts.append(receipt)
            try:
                _checkpoint(
                    contract, preregistration, capsule, authorization,
                    consumption, receipts,
                )
                _reconcile_trajectory(
                    contract,
                    capsule,
                    authorization=authorization,
                    consumption=consumption,
                    receipts=receipts,
                )
            except ControllerPending as error:
                _persist_pending(
                    contract,
                    preregistration,
                    capsule,
                    error,
                    authorization=authorization,
                    consumption=consumption,
                    receipts=receipts,
                )
                raise

        artifact = state.build_candidate_from_receipts(capsule, receipts).canonical_bytes
        try:
            _publish_bytes(artifact_path, artifact, "Q5a candidate artifact")
        except OSError as error:
            try:
                artifact_present = _path_present(artifact_path)
            except ControllerPending as presence_error:
                artifact_present = False
                error = OSError(
                    f"{error}; artifact presence unproven: {presence_error}"
                )
            if not artifact_present:
                pending = ControllerPending(
                    "CANDIDATE_ARTIFACT_PUBLICATION",
                    "Q5A_CANDIDATE_ARTIFACT_NOT_DURABLE",
                    str(error),
                )
                _persist_pending(
                    contract,
                    preregistration,
                    capsule,
                    pending,
                    authorization=authorization,
                    consumption=consumption,
                    receipts=receipts,
                )
                raise pending
            try:
                durable_artifact = _read_regular(
                    artifact_path, "Q5a candidate artifact", MAX_ARTIFACT_BYTES
                )
                state.validate_artifact_binding(
                    durable_artifact, capsule, receipts
                )
            except ControllerPending as replay_error:
                pending = ControllerPending(
                    "CANDIDATE_ARTIFACT_REPLAY",
                    "Q5A_CANDIDATE_ARTIFACT_TEMPORARILY_UNAVAILABLE",
                    str(replay_error),
                )
                _persist_pending(
                    contract,
                    preregistration,
                    capsule,
                    pending,
                    authorization=authorization,
                    consumption=consumption,
                    receipts=receipts,
                )
                raise pending
            except ValueError:
                return _seal_invalidated(
                    contract,
                    preregistration,
                    capsule,
                    authorization,
                    consumption,
                    receipts,
                    None,
                    phase="ARTIFACT_PUBLICATION",
                    code="Q5A_CANDIDATE_ARTIFACT_PUBLICATION_FAILED",
                    detail=str(error),
                )
            artifact = durable_artifact
        terminal = state.build_complete_terminal(
            contract, preregistration.value, capsule, authorization, consumption,
            receipts, artifact,
        )
        try:
            _publish_json(
                Path(fixed["terminal_path"]), terminal, "Q5a terminal"
            )
        except OSError as error:
            raise ControllerPartial(
                PARTIAL_ARTIFACT_TO_TERMINAL, str(error)
            ) from error
        _reconcile_trajectory(
            contract,
            capsule,
            authorization=authorization,
            consumption=consumption,
            receipts=receipts,
            terminal=terminal,
        )
        _clear_pending(contract)
        return terminal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-capsule")
    prepare.add_argument("--source-commit", required=True)
    prepare.add_argument("--run-id")
    prepare.add_argument("--created-at-unix-seconds", type=int)
    for name in ("record-authorization", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--capsule-digest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "prepare-capsule":
            result = prepare_capsule(
                arguments.contract, arguments.source_commit,
                run_id=arguments.run_id,
                created_at_unix_seconds=arguments.created_at_unix_seconds,
            )
        elif arguments.command == "record-authorization":
            result = record_authorization(
                arguments.contract, arguments.capsule_digest
            )
        else:
            result = run_v1(arguments.contract, arguments.capsule_digest)
    except ControllerPending as error:
        print(
            state.canonical_json_bytes(
                {"status": state.PENDING, "phase": error.phase,
                 "condition": error.condition, "detail": error.detail}
            ).decode("ascii"),
            end="",
        )
        return PENDING_EXIT_CODE
    except ControllerTerminal as error:
        print(
            state.canonical_json_bytes(
                {"status": state.INVALIDATED_TERMINAL_NO_RESUME,
                 "failure_code": error.code, "failure_detail": error.detail}
            ).decode("ascii"),
            end="",
        )
        return TERMINAL_EXIT_CODE
    except ControllerPartial as error:
        print(
            state.canonical_json_bytes(
                {
                    "status": state.PARTIAL_FINALIZATION_REQUIRED,
                    "partial_finalization_kind": error.kind,
                    "detail": error.detail,
                    "exact_artifact_to_terminal_reentry_allowed": (
                        error.kind == PARTIAL_ARTIFACT_TO_TERMINAL
                    ),
                    "exact_terminal_to_trajectory_suffix_reentry_allowed": (
                        error.kind == PARTIAL_TERMINAL_TO_TRAJECTORY_SUFFIX
                    ),
                    "authorization_consumption_count": 1,
                    "logical_campaign_count": 1,
                }
            ).decode("ascii"),
            end="",
        )
        return TERMINAL_EXIT_CODE
    print(state.canonical_json_bytes(result).decode("ascii"), end="")
    if result.get("status") == state.NOT_AUTHORIZED:
        return NOT_AUTHORIZED_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
