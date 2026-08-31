#!/usr/bin/env python3
"""Zero-process controller for Q4a capability admission, never issuance."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
import time
import types
from typing import Any, Final, Iterator, Mapping, NoReturn, Sequence


sys.dont_write_bytecode = True
PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
PACKAGE_ROOT: Final = PROJECT_ROOT / "src" / "hegel_machine"
PACKAGE_NAME: Final = "hegel_machine"
if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(PACKAGE_ROOT)]
    package.__package__ = PACKAGE_NAME
    sys.modules[PACKAGE_NAME] = package

state = importlib.import_module("hegel_machine.phase3_q4_completion_replay_v1")
wire = importlib.import_module(
    "hegel_machine.phase3_q4_formal_quotient_certificate_capability_v1"
)

DEFAULT_CONTRACT: Final = PROJECT_ROOT / (
    "config/phase3_q4_formal_quotient_certificate_capability_actual_v1.json"
)
PENDING_EXIT_CODE: Final = 75
NOT_AUTHORIZED_EXIT_CODE: Final = 4
TERMINAL_EXIT_CODE: Final = 2


class ControllerError(RuntimeError):
    """Base class for classified Q4a controller outcomes."""


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
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def _terminal(code: str, detail: str) -> NoReturn:
    raise ControllerTerminal(code, detail)


def _read_regular(
    path: Path,
    label: str,
    maximum_bytes: int = state.MAX_JSON_BYTES,
) -> bytes:
    try:
        info = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise OSError("not one regular non-symlink file")
        if info.st_size > maximum_bytes:
            raise OSError(f"exceeds {maximum_bytes}-byte bound")
        return path.read_bytes()
    except OSError as error:
        raise ControllerPending(
            "LOCAL_READ_ONLY_REPLAY",
            "BOUND_LOCAL_FILE_UNAVAILABLE",
            f"{label}: {error}",
        ) from error


def _load_contract(path: Path) -> tuple[Any, Any]:
    try:
        contract = state.load_contract(path)
        preregistration = state.load_preregistration(
            Path(contract.value["preregistration_path"])
        )
    except (OSError, ValueError) as error:
        raise ControllerPending(
            "LOCAL_CONTRACT",
            "Q4_CONTRACT_OR_PREREGISTRATION_UNAVAILABLE",
            str(error),
        ) from error
    if preregistration.raw_sha256 != contract.value["preregistration_raw_sha256"]:
        raise ControllerPending(
            "LOCAL_CONTRACT",
            "Q4_PREREGISTRATION_RAW_IDENTITY_DIFFERS",
            str(preregistration.path),
        )
    return contract, preregistration


def _source_closure_rows(contract: Any) -> list[dict[str, Any]]:
    project_root = Path(contract.value["fixed_paths"]["project_root"])
    rows: list[dict[str, Any]] = []
    for relative in state.Q4_SOURCE_PATHS:
        path = project_root / relative
        payload = _read_regular(path, f"Q4 source {relative}")
        try:
            if path.resolve(strict=True) != path:
                raise OSError("source path resolves elsewhere")
        except OSError as error:
            raise ControllerPending(
                "SOURCE_CLOSURE", "Q4_SOURCE_PATH_IDENTITY_DIFFERS",
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


def _read_q3_selected(contract: Any) -> dict[str, bytes]:
    selected: dict[str, bytes] = {}
    for kind in ("capsule", "authorization", "consumption", "terminal", "artifact"):
        bound = Path(contract.value["q3_inputs"][kind])
        selected[kind] = _read_regular(
            bound,
            f"sealed Q3 {kind}",
            state.MAX_Q3_ARTIFACT_BYTES if kind == "artifact" else state.MAX_JSON_BYTES,
        )
    return selected


def _replay_q3(
    contract: Any,
    preregistration: Any,
    *,
    q4_source_commit: str,
    q4_source_closure_root: str,
) -> Any:
    selected = _read_q3_selected(contract)
    try:
        return state.replay_q3_predecessor_v1(
            preregistration.value,
            q4_source_commit=q4_source_commit,
            q4_source_closure_root=q4_source_closure_root,
            capsule_payload=selected["capsule"],
            authorization_payload=selected["authorization"],
            consumption_payload=selected["consumption"],
            terminal_payload=selected["terminal"],
            artifact_payload=selected["artifact"],
        )
    except ValueError as error:
        raise ControllerPending(
            "SEALED_Q3_REPLAY", "Q4_SEALED_Q3_REPLAY_DIFFERS", str(error)
        ) from error


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_bytes(path: Path, payload: bytes, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            if path.read_bytes() == payload:
                return
        except OSError as error:
            raise OSError(f"{label} readback failed: {error}") from error
        raise OSError(f"{label} already exists with different bytes")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        _fsync_parent(path)
        if path.read_bytes() != payload:
            raise OSError("readback differs")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _publish_json(path: Path, value: Mapping[str, Any], label: str) -> None:
    _publish_bytes(path, state.canonical_json_bytes(dict(value)), label)


def _replace_bytes(path: Path, payload: bytes, label: str) -> None:
    """Atomically replace one transient latest-state file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError(f"{label} short write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        _fsync_parent(path)
        if path.read_bytes() != payload:
            raise OSError(f"{label} readback differs")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _clear_pending(contract: Any) -> None:
    path = Path(contract.value["fixed_paths"]["pending_path"])
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_parent(path)


def _persist_pending(
    contract: Any,
    preregistration: Any,
    capsule: Mapping[str, Any],
    error: ControllerPending,
    authorization: Mapping[str, Any] | None = None,
) -> None:
    pending = state.build_pending(
        contract,
        preregistration.value,
        capsule,
        phase=error.phase,
        condition=error.condition,
        detail=error.detail,
        authorization=authorization,
    )
    path = Path(contract.value["fixed_paths"]["pending_path"])
    _replace_bytes(path, state.canonical_json_bytes(pending), "Q4 pending")


def _append_trajectory(contract: Any, event: Mapping[str, Any]) -> None:
    path = Path(contract.value["fixed_paths"]["trajectory_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = state.canonical_json_bytes(dict(event))
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("short trajectory write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_parent(path)


def _read_json_if_present(path: Path, label: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return state.strict_json_bytes(_read_regular(path, label), label)


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


def _read_capsule(
    contract: Any, preregistration: Any, digest: str
) -> dict[str, Any]:
    path = Path(contract.value["fixed_paths"]["capsule_path"])
    value = _read_json_if_present(path, "Q4 capsule")
    if value is None:
        raise ControllerPending("CAPSULE_REPLAY", "Q4_CAPSULE_UNAVAILABLE", str(path))
    try:
        checked = state.validate_capsule(value, contract, preregistration.value)
    except ValueError as error:
        raise ControllerPending(
            "CAPSULE_REPLAY", "Q4_CAPSULE_REPLAY_DIFFERS", str(error)
        ) from error
    if checked["capsule_digest"] != digest:
        raise ControllerPending(
            "CAPSULE_REPLAY", "Q4_CAPSULE_DIGEST_DIFFERS", digest
        )
    return checked


def _read_authorization(
    contract: Any, preregistration: Any, capsule: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = Path(contract.value["fixed_paths"]["authorization_path"])
    value = _read_json_if_present(path, "Q4 authorization")
    if value is None:
        return None
    try:
        return state.validate_authorization(
            value, contract, preregistration.value, capsule
        )
    except ValueError as error:
        _terminal("Q4_AUTHORIZATION_REPLAY_DIFFERS", str(error))


def _read_consumption(
    contract: Any,
    preregistration: Any,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = Path(contract.value["fixed_paths"]["consumption_path"])
    value = _read_json_if_present(path, "Q4 consumption")
    if value is None:
        return None
    try:
        return state.validate_consumption(
            value, contract, preregistration.value, capsule, authorization
        )
    except ValueError as error:
        _terminal("Q4_CONSUMPTION_REPLAY_DIFFERS", str(error))


def _read_terminal(
    contract: Any,
    preregistration: Any,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
) -> dict[str, Any] | None:
    fixed = contract.value["fixed_paths"]
    value = _read_json_if_present(Path(fixed["terminal_path"]), "Q4 terminal")
    if value is None:
        return None
    artifact_payload: bytes | None = None
    if value.get("status") == wire.Q4_SUCCESS_STATUS:
        try:
            artifact_payload = _read_regular(
                Path(fixed["artifact_path"]), "Q4 capability receipt",
                state.MAX_Q3_ARTIFACT_BYTES,
            )
            state.validate_artifact_binding(artifact_payload, capsule)
        except (ControllerPending, ValueError) as error:
            _terminal("Q4_TERMINAL_ARTIFACT_REPLAY_DIFFERS", str(error))
    try:
        return state.validate_terminal(
            value, contract, preregistration.value, capsule, authorization,
            consumption, artifact_payload=artifact_payload,
        )
    except ValueError as error:
        _terminal("Q4_TERMINAL_REPLAY_DIFFERS", str(error))


def _preconsumption_replay(
    contract: Any, preregistration: Any, capsule: Mapping[str, Any]
) -> Any:
    rows = _source_closure_rows(contract)
    root = state.source_closure_root(rows)
    if rows != capsule["source_closure_rows"] or root != capsule["source_closure_root"]:
        raise ControllerPending(
            "SOURCE_CLOSURE", "Q4_CAPSULE_SOURCE_CLOSURE_DIFFERS", root
        )
    replay = _replay_q3(
        contract,
        preregistration,
        q4_source_commit=capsule["source_commit"],
        q4_source_closure_root=root,
    )
    if replay.preflight_receipt != capsule["preflight_receipt"]:
        raise ControllerPending(
            "SEALED_Q3_REPLAY", "Q4_PREFLIGHT_RECEIPT_DIFFERS",
            capsule["capsule_digest"],
        )
    return replay


def prepare_capsule(
    contract_path: Path,
    source_commit: str,
    *,
    run_id: str | None = None,
    created_at_unix_seconds: int | None = None,
) -> dict[str, Any]:
    contract, preregistration = _load_contract(contract_path)
    fixed = contract.value["fixed_paths"]
    capsule_path = Path(fixed["capsule_path"])
    postconsumption_state = any(
        Path(fixed[key]).exists()
        for key in ("consumption_path", "artifact_path", "terminal_path")
    )
    if postconsumption_state and not capsule_path.exists():
        _terminal(
            "Q4_CAPSULE_MISSING_AFTER_EXECUTION_STATE", str(capsule_path)
        )
    if capsule_path.exists():
        try:
            value = _read_json_if_present(capsule_path, "Q4 capsule")
            assert value is not None
            capsule = state.validate_capsule(
                value, contract, preregistration.value
            )
        except (ControllerPending, ValueError) as error:
            if postconsumption_state:
                _terminal(
                    "Q4_CAPSULE_REPLAY_FAILED_AFTER_EXECUTION_STATE",
                    str(error),
                )
            raise ControllerPending(
                "CAPSULE_REPLAY", "Q4_CAPSULE_REPLAY_DIFFERS", str(error)
            ) from error
        if postconsumption_state:
            # Capsule preparation is already complete.  Once execution state
            # exists, this entry must not reopen pre-consumption validation or
            # create a zero-count PENDING receipt.
            return capsule
        try:
            _preconsumption_replay(contract, preregistration, capsule)
        except ControllerPending as error:
            authorization = _read_authorization(
                contract, preregistration, capsule
            )
            _persist_pending(
                contract, preregistration, capsule, error, authorization
            )
            raise
        _clear_pending(contract)
        return capsule
    rows = _source_closure_rows(contract)
    root = state.source_closure_root(rows)
    replay = _replay_q3(
        contract, preregistration, q4_source_commit=source_commit,
        q4_source_closure_root=root,
    )
    inputs = {
        "source_commit": source_commit,
        "source_closure_rows": rows,
        "source_closure_root": root,
        "run_id": run_id or secrets.token_hex(16),
        "created_at_unix_seconds": (
            int(time.time()) if created_at_unix_seconds is None
            else created_at_unix_seconds
        ),
        "preflight_receipt": replay.preflight_receipt,
    }
    capsule = state.build_capsule(contract, preregistration.value, inputs)
    with _campaign_lock(contract):
        for key in ("authorization_path", "consumption_path", "artifact_path", "terminal_path"):
            if Path(fixed[key]).exists():
                raise ControllerPending(
                    "CAPSULE_PREPARATION", "Q4_FUTURE_STATE_ALREADY_EXISTS",
                    fixed[key],
                )
        existing = _read_json_if_present(capsule_path, "Q4 capsule")
        if existing is not None:
            checked = state.validate_capsule(
                existing, contract, preregistration.value
            )
            if checked != capsule:
                raise ControllerPending(
                    "CAPSULE_PREPARATION", "Q4_EXISTING_CAPSULE_DIFFERS",
                    fixed["capsule_path"],
                )
            return checked
        _publish_json(capsule_path, capsule, "Q4 capsule")
        _append_trajectory(
            contract,
            {
                "schema_version": "hegel-phase3a-q4-trajectory/1",
                "event": "CAPSULE_PREPARED_NOT_AUTHORIZED",
                "capsule_digest": capsule["capsule_digest"],
                "authorization_consumption_count": 0,
                "logical_campaign_count": 0,
                "q4_action_audit": state.ZERO_ACTION_AUDIT,
                "time_ns": time.time_ns(),
            },
        )
    return capsule


def record_authorization(contract_path: Path, capsule_digest: str) -> dict[str, Any]:
    contract, preregistration = _load_contract(contract_path)
    fixed = contract.value["fixed_paths"]
    with _campaign_lock(contract):
        future_state = any(
            Path(fixed[key]).exists()
            for key in ("consumption_path", "artifact_path", "terminal_path")
        )
        try:
            capsule = _read_capsule(contract, preregistration, capsule_digest)
            existing = _read_authorization(
                contract, preregistration, capsule
            )
        except (ControllerPending, ValueError) as error:
            if future_state:
                _terminal(
                    "Q4_AUTHORIZATION_REPLAY_FAILED_AFTER_EXECUTION_STATE",
                    str(error),
                )
            raise
        if future_state:
            if existing is not None:
                return existing
            _terminal("Q4_AUTHORIZATION_AFTER_EXECUTION_STATE", capsule_digest)
        try:
            _preconsumption_replay(contract, preregistration, capsule)
        except ControllerPending as error:
            _persist_pending(
                contract, preregistration, capsule, error, existing
            )
            raise
        if existing is not None:
            _clear_pending(contract)
            return existing
        authorization = state.build_authorization(
            contract, preregistration.value, capsule
        )
        _publish_json(
            Path(fixed["authorization_path"]), authorization, "Q4 authorization"
        )
        _append_trajectory(
            contract,
            {
                "schema_version": "hegel-phase3a-q4-trajectory/1",
                "event": "AUTHORIZATION_RECORDED_NOT_CONSUMED",
                "capsule_digest": capsule_digest,
                "authorization_digest": authorization["authorization_digest"],
                "authorization_consumption_count": 0,
                "logical_campaign_count": 0,
                "q4_action_audit": state.ZERO_ACTION_AUDIT,
                "time_ns": time.time_ns(),
            },
        )
        _clear_pending(contract)
        return authorization


def _seal_invalidated(
    contract: Any,
    preregistration: Any,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    *,
    phase: str,
    code: str,
    detail: str,
) -> dict[str, Any]:
    terminal = state.build_invalidated_terminal(
        contract,
        preregistration.value,
        capsule,
        authorization,
        consumption,
        failure_phase=phase,
        failure_code=code,
        failure_detail=detail,
    )
    _publish_json(
        Path(contract.value["fixed_paths"]["terminal_path"]),
        terminal,
        "Q4 invalidated terminal",
    )
    _clear_pending(contract)
    return terminal


def run_v1(contract_path: Path, capsule_digest: str) -> dict[str, Any]:
    contract, preregistration = _load_contract(contract_path)
    fixed = contract.value["fixed_paths"]
    with _campaign_lock(contract):
        consumption_exists = Path(fixed["consumption_path"]).exists()
        artifact_exists = Path(fixed["artifact_path"]).exists()
        terminal_exists = Path(fixed["terminal_path"]).exists()
        postconsumption_state = (
            consumption_exists or artifact_exists or terminal_exists
        )
        try:
            capsule = _read_capsule(
                contract, preregistration, capsule_digest
            )
            authorization = _read_authorization(
                contract, preregistration, capsule
            )
        except (ControllerPending, ValueError) as error:
            if postconsumption_state:
                _terminal(
                    "Q4_PREFIX_REPLAY_FAILED_AFTER_EXECUTION_STATE",
                    str(error),
                )
            raise
        if authorization is None:
            if postconsumption_state:
                _terminal(
                    "Q4_AUTHORIZATION_MISSING_AFTER_EXECUTION_STATE",
                    capsule_digest,
                )
            return {
                "schema_version": "hegel-phase3a-q4-controller-not-authorized/1",
                "status": state.NOT_AUTHORIZED,
                "capsule_digest": capsule_digest,
                "authorization_consumption_count": 0,
                "logical_campaign_count": 0,
                "q4_action_audit": state.ZERO_ACTION_AUDIT,
            }
        if not consumption_exists and (artifact_exists or terminal_exists):
            _terminal(
                "Q4_CONSUMPTION_MISSING_AFTER_EXECUTION_STATE",
                capsule_digest,
            )
        try:
            consumption = _read_consumption(
                contract, preregistration, capsule, authorization
            )
        except (ControllerPending, ValueError) as error:
            if consumption_exists:
                _terminal(
                    "Q4_CONSUMPTION_REPLAY_FAILED_AFTER_PUBLICATION",
                    str(error),
                )
            raise
        if consumption is not None:
            try:
                terminal = _read_terminal(
                    contract,
                    preregistration,
                    capsule,
                    authorization,
                    consumption,
                )
            except (ControllerPending, ValueError) as error:
                _terminal(
                    "Q4_TERMINAL_REPLAY_FAILED_AFTER_CONSUMPTION",
                    str(error),
                )
            if terminal is not None:
                return terminal
        replay: Any | None = None
        if consumption is None:
            try:
                replay = _preconsumption_replay(
                    contract, preregistration, capsule
                )
            except ControllerPending as error:
                _persist_pending(
                    contract, preregistration, capsule, error, authorization
                )
                raise
            consumption = state.build_consumption(
                contract, preregistration.value, capsule, authorization
            )
            try:
                consumption_path = Path(fixed["consumption_path"])
                _publish_json(
                    consumption_path, consumption, "Q4 consumption"
                )
            except OSError as error:
                try:
                    consumption_path.lstat()
                except FileNotFoundError:
                    durable = None
                except OSError as probe_error:
                    _terminal(
                        "Q4_CONSUMPTION_PUBLICATION_OUTCOME_UNKNOWN",
                        str(probe_error),
                    )
                else:
                    try:
                        durable = state.strict_json_bytes(
                            _read_regular(
                                consumption_path,
                                "Q4 consumption after publication error",
                            ),
                            "Q4 consumption after publication error",
                        )
                        consumption = state.validate_consumption(
                            durable,
                            contract,
                            preregistration.value,
                            capsule,
                            authorization,
                        )
                    except (ControllerPending, ValueError) as replay_error:
                        _terminal(
                            "Q4_CONSUMPTION_PUBLICATION_OUTCOME_UNKNOWN",
                            str(replay_error),
                        )
                    _clear_pending(contract)
                if durable is None:
                    pending = ControllerPending(
                        "CONSUMPTION_PUBLICATION",
                        "Q4_CONSUMPTION_NOT_DURABLE",
                        str(error),
                    )
                    _persist_pending(
                        contract,
                        preregistration,
                        capsule,
                        pending,
                        authorization,
                    )
                    raise pending
            _clear_pending(contract)

        artifact_path = Path(fixed["artifact_path"])
        artifact_payload: bytes
        if artifact_path.exists():
            try:
                artifact_payload = _read_regular(
                    artifact_path, "Q4 capability receipt", state.MAX_Q3_ARTIFACT_BYTES
                )
                state.validate_artifact_binding(artifact_payload, capsule)
            except (ControllerPending, ValueError) as error:
                return _seal_invalidated(
                    contract,
                    preregistration,
                    capsule,
                    authorization,
                    consumption,
                    phase="ARTIFACT_REPLAY",
                    code="Q4_CAPABILITY_ARTIFACT_REPLAY_DIFFERS",
                    detail=str(error),
                )
        else:
            if replay is None:
                return _seal_invalidated(
                    contract,
                    preregistration,
                    capsule,
                    authorization,
                    consumption,
                    phase="ARTIFACT_REPLAY",
                    code="Q4_CAPABILITY_ARTIFACT_MISSING_AFTER_CONSUMPTION",
                    detail=str(artifact_path),
                )
            artifact_payload = replay.artifact_payload
            try:
                _publish_bytes(
                    artifact_path, artifact_payload, "Q4 capability receipt"
                )
            except OSError as error:
                return _seal_invalidated(
                    contract,
                    preregistration,
                    capsule,
                    authorization,
                    consumption,
                    phase="ARTIFACT_PUBLICATION",
                    code="Q4_CAPABILITY_ARTIFACT_PUBLICATION_FAILED",
                    detail=str(error),
                )

        terminal = state.build_complete_terminal(
            contract, preregistration.value, capsule, authorization,
            consumption, artifact_payload,
        )
        try:
            _publish_json(Path(fixed["terminal_path"]), terminal, "Q4 terminal")
        except OSError as error:
            raise ControllerPartial(str(error)) from error
        _append_trajectory(
            contract,
            {
                "schema_version": "hegel-phase3a-q4-trajectory/1",
                "event": "Q4_CAPABILITY_ADMISSION_COMPLETE_NOT_ISSUED",
                "capsule_digest": capsule_digest,
                "capability_receipt_root": terminal["outcome"][
                    "capability_receipt_root"
                ],
                "authorization_consumption_count": 1,
                "logical_campaign_count": 1,
                "q4_action_audit": state.ZERO_ACTION_AUDIT,
                "time_ns": time.time_ns(),
            },
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
        command.add_argument("--capsule-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare-capsule":
            result = prepare_capsule(
                args.contract, args.source_commit, run_id=args.run_id,
                created_at_unix_seconds=args.created_at_unix_seconds,
            )
        elif args.command == "record-authorization":
            result = record_authorization(args.contract, args.capsule_sha256)
        else:
            result = run_v1(args.contract, args.capsule_sha256)
    except ControllerPending as error:
        result = {
            "schema_version": "hegel-phase3a-q4-controller-pending/1",
            "status": state.PENDING_NOT_CONSUMED,
            "phase": error.phase,
            "condition": error.condition,
            "detail": error.detail,
            "authorization_consumption_count": 0,
            "logical_campaign_count": 0,
            "q4_action_audit": state.ZERO_ACTION_AUDIT,
        }
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return PENDING_EXIT_CODE
    except ControllerPartial as error:
        result = {
            "schema_version": "hegel-phase3a-q4-controller-partial/1",
            "status": state.PARTIAL_FINALIZATION_REQUIRED,
            "detail": error.detail,
            "exact_artifact_to_terminal_reentry_allowed": True,
            "authorization_consumption_count": 1,
            "logical_campaign_count": 1,
            "q4_action_audit": state.ZERO_ACTION_AUDIT,
        }
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return TERMINAL_EXIT_CODE
    except ControllerTerminal as error:
        result = {
            "schema_version": "hegel-phase3a-q4-controller-terminal/1",
            "status": state.INVALIDATED_TERMINAL_NO_RESUME,
            "failure_code": error.code,
            "failure_detail": error.detail,
        }
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return TERMINAL_EXIT_CODE
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    if result.get("status") == state.NOT_AUTHORIZED:
        return NOT_AUTHORIZED_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
