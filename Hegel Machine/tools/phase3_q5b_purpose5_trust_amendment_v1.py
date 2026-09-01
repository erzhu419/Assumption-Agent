#!/usr/bin/env python3
"""Zero-process controller for the public-only Q5b trust amendment."""

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

state = importlib.import_module("hegel_machine.phase3_q5b_completion_replay_v1")
wire = importlib.import_module("hegel_machine.phase3_q5b_purpose5_trust_amendment_v1")

DEFAULT_CONTRACT: Final = PROJECT_ROOT / (
    "config/phase3_q5b_purpose5_trust_amendment_actual_v1.json"
)
PENDING_EXIT_CODE: Final = 75
NOT_AUTHORIZED_EXIT_CODE: Final = 4
TERMINAL_EXIT_CODE: Final = 2


class ControllerError(RuntimeError):
    """Base class for classified Q5b controller outcomes."""


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
            "BOUND_PUBLIC_FILE_UNAVAILABLE",
            f"{label}: {error}",
        ) from error


def _load_contract(path: Path) -> tuple[Any, Any]:
    try:
        contract = state.load_contract(path)
        preregistration = state.load_preregistration(
            Path(contract.value["preregistration_path"])
        )
    except (OSError, ValueError) as error:
        raise ControllerTerminal(
            "Q5B_BOOTSTRAP_CONTRACT_OR_PREREGISTRATION_UNAVAILABLE_NO_STATE_CLAIM",
            str(error),
        ) from error
    if preregistration.raw_sha256 != contract.value["preregistration_raw_sha256"]:
        raise ControllerTerminal(
            "Q5B_BOOTSTRAP_PREREGISTRATION_RAW_IDENTITY_DIFFERS_NO_STATE_CLAIM",
            str(preregistration.path),
        )
    return contract, preregistration


def _source_closure_rows(contract: Any) -> list[dict[str, Any]]:
    project_root = Path(contract.value["fixed_paths"]["project_root"])
    rows: list[dict[str, Any]] = []
    for relative in state.Q5B_SOURCE_PATHS:
        path = project_root / relative
        payload = _read_regular(path, f"Q5b source {relative}")
        try:
            if path.resolve(strict=True) != path:
                raise OSError("source path resolves elsewhere")
        except OSError as error:
            raise ControllerPending(
                "SOURCE_CLOSURE",
                "Q5B_SOURCE_PATH_IDENTITY_DIFFERS",
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


def _read_q5a_selected(contract: Any) -> dict[str, bytes]:
    selected: dict[str, bytes] = {}
    for kind in (
        "capsule",
        "authorization",
        "consumption",
        "terminal",
        "artifact",
        "trajectory",
    ):
        path = Path(contract.value["q5a_inputs"][kind])
        maximum = (
            state.MAX_Q5A_ARTIFACT_BYTES
            if kind == "artifact"
            else state.MAX_TRAJECTORY_BYTES
            if kind == "trajectory"
            else state.MAX_JSON_BYTES
        )
        selected[kind] = _read_regular(path, f"sealed public Q5a {kind}", maximum)
    return selected


def _replay_q5a(
    contract: Any,
    preregistration: Any,
    *,
    source_commit: str,
    source_closure_root: str,
) -> Any:
    selected = _read_q5a_selected(contract)
    try:
        return state.replay_q5a_predecessor_v1(
            preregistration.value,
            q5b_source_commit=source_commit,
            q5b_source_closure_root=source_closure_root,
            capsule_payload=selected["capsule"],
            authorization_payload=selected["authorization"],
            consumption_payload=selected["consumption"],
            terminal_payload=selected["terminal"],
            artifact_payload=selected["artifact"],
            trajectory_payload=selected["trajectory"],
        )
    except (TypeError, ValueError) as error:
        raise ControllerPending(
            "SEALED_Q5A_REPLAY",
            "Q5B_SEALED_Q5A_REPLAY_DIFFERS",
            str(error),
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
        if path.is_symlink() or not path.is_file():
            raise OSError(f"{label} exists but is not a regular file")
        if path.read_bytes() == payload:
            return
        raise OSError(f"{label} already exists with different bytes")
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


def _publish_json(path: Path, value: Mapping[str, Any], label: str) -> None:
    _publish_bytes(path, state.canonical_json_bytes(dict(value)), label)


def _replace_bytes(path: Path, payload: bytes, label: str) -> None:
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
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _clear_pending(contract: Any) -> None:
    path = Path(contract.value["fixed_paths"]["pending_path"])
    try:
        path.unlink()
        _fsync_parent(path)
    except FileNotFoundError:
        pass


def _trajectory_event(
    name: str,
    capsule: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any] | None = None,
    consumption: Mapping[str, Any] | None = None,
    terminal: Mapping[str, Any] | None = None,
    time_ns: int,
) -> dict[str, Any]:
    if type(time_ns) is not int or time_ns < 0:
        raise ValueError("Q5b trajectory time differs")
    audit = state.ZERO_ACTION_AUDIT
    if terminal is not None:
        audit = terminal["outcome"]["q5b_action_audit"]
    event: dict[str, Any] = {
        "schema_version": "hegel-phase3a-q5b-trajectory/1",
        "event": name,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": (
            None if authorization is None else authorization["authorization_digest"]
        ),
        "consumption_record_digest": (
            None if consumption is None else consumption["record_digest"]
        ),
        "authorization_consumption_count": 0 if consumption is None else 1,
        "logical_campaign_count": 0 if consumption is None else 1,
        "q5b_action_audit": audit,
        "time_ns": time_ns,
    }
    if terminal is not None:
        outcome = terminal["outcome"]
        event.update(
            {
                "terminal_record_digest": terminal["record_digest"],
                "artifact_raw_sha256": outcome.get("artifact_raw_sha256"),
                "trust_amendment_receipt_root": outcome.get(
                    "trust_amendment_receipt_root"
                ),
                "pinned_trust_genesis_root": outcome.get(
                    "pinned_trust_genesis_root"
                ),
                "current_key_status_root": outcome.get("current_key_status_root"),
                "purpose5_active_signer_count": outcome.get(
                    "purpose5_active_signer_count"
                ),
                "global_active_transition_performed": outcome.get(
                    "global_active_transition_performed"
                ),
                "failure_code": outcome.get("failure_code"),
            }
        )
    return event


def _trajectory_expected(
    capsule: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any] | None = None,
    consumption: Mapping[str, Any] | None = None,
    terminal: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = [
        _trajectory_event(
            "CAPSULE_PREPARED_NOT_AUTHORIZED", capsule, time_ns=0
        )
    ]
    if authorization is not None:
        rows.append(
            _trajectory_event(
                "AUTHORIZATION_RECORDED_NOT_CONSUMED",
                capsule,
                authorization=authorization,
                time_ns=0,
            )
        )
    if terminal is not None:
        if authorization is None or consumption is None:
            raise ValueError("Q5b terminal trajectory lacks consumed authority")
        rows.append(
            _trajectory_event(
                terminal["status"],
                capsule,
                authorization=authorization,
                consumption=consumption,
                terminal=terminal,
                time_ns=0,
            )
        )
    return rows


def _without_time(value: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(value)
    observed = row.pop("time_ns", None)
    if type(observed) is not int or observed < 0:
        raise ValueError("Q5b trajectory time differs")
    return row


def _reconcile_trajectory(
    contract: Any,
    capsule: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any] | None = None,
    consumption: Mapping[str, Any] | None = None,
    terminal: Mapping[str, Any] | None = None,
) -> None:
    path = Path(contract.value["fixed_paths"]["trajectory_path"])
    expected = _trajectory_expected(
        capsule,
        authorization=authorization,
        consumption=consumption,
        terminal=terminal,
    )
    observed: list[dict[str, Any]] = []
    if path.exists():
        try:
            payload = _read_regular(
                path, "Q5b trajectory", state.MAX_TRAJECTORY_BYTES
            )
            if not payload.endswith(b"\n"):
                raise ValueError("Q5b trajectory framing differs")
            observed = [
                state.strict_json_bytes(line, f"Q5b trajectory event {index}")
                for index, line in enumerate(payload.splitlines(keepends=True), 1)
            ]
        except ControllerPending as error:
            if terminal is None:
                raise
            raise ControllerPartial(
                f"terminal is durable; trajectory replay unavailable: {error}"
            ) from error
        except ValueError as error:
            _terminal("Q5B_TRAJECTORY_REPLAY_DIFFERS", str(error))
    if len(observed) > len(expected):
        _terminal("Q5B_TRAJECTORY_LEADS_DURABLE_STATE", str(len(observed)))
    prior_time = -1
    for index, row in enumerate(observed):
        try:
            if _without_time(row) != _without_time(expected[index]):
                raise ValueError(f"Q5b trajectory event {index + 1} differs")
            if row["time_ns"] < prior_time:
                raise ValueError("Q5b trajectory time order differs")
            prior_time = row["time_ns"]
        except ValueError as error:
            _terminal("Q5B_TRAJECTORY_REPLAY_DIFFERS", str(error))
    if len(observed) == len(expected):
        return
    next_time = max(time.time_ns(), prior_time + 1)
    completed = list(observed)
    for row in expected[len(observed) :]:
        with_time = dict(row)
        with_time["time_ns"] = next_time
        completed.append(with_time)
        next_time += 1
    payload = b"".join(state.canonical_json_bytes(row) for row in completed)
    if len(payload) > state.MAX_TRAJECTORY_BYTES:
        _terminal("Q5B_TRAJECTORY_EXCEEDS_BOUND", str(len(payload)))
    try:
        _replace_bytes(path, payload, "Q5b trajectory")
    except OSError as error:
        if terminal is not None:
            raise ControllerPartial(
                f"terminal is durable; trajectory suffix publication failed: {error}"
            ) from error
        raise ControllerPending(
            "TRAJECTORY_PUBLICATION",
            "Q5B_TRAJECTORY_SUFFIX_NOT_DURABLE",
            str(error),
        ) from error
    try:
        readback = _read_regular(path, "Q5b trajectory readback", state.MAX_TRAJECTORY_BYTES)
        rows = [
            state.strict_json_bytes(line, f"Q5b trajectory readback {index}")
            for index, line in enumerate(readback.splitlines(keepends=True), 1)
        ]
        if len(rows) != len(expected) or any(
            _without_time(row) != _without_time(wanted)
            for row, wanted in zip(rows, expected, strict=True)
        ):
            raise ValueError("Q5b trajectory readback differs")
    except (ControllerPending, ValueError) as error:
        if terminal is not None:
            raise ControllerPartial(
                f"terminal is durable; trajectory suffix readback failed: {error}"
            ) from error
        raise ControllerPending(
            "TRAJECTORY_PUBLICATION",
            "Q5B_TRAJECTORY_SUFFIX_NOT_DURABLE",
            str(error),
        ) from error


def _publish_pending(
    contract: Any,
    preregistration: Any,
    capsule: Mapping[str, Any],
    error: ControllerPending,
    *,
    authorization: Mapping[str, Any] | None = None,
) -> None:
    consumption_path = Path(contract.value["fixed_paths"]["consumption_path"])
    if consumption_path.exists():
        _terminal(
            "Q5B_POSTCONSUMPTION_PENDING_FORBIDDEN",
            f"{error.phase}/{error.condition}: {error.detail}",
        )
    pending = state.build_pending(
        contract,
        preregistration.value,
        capsule,
        phase=error.phase,
        condition=error.condition,
        detail=error.detail,
        authorization=authorization,
    )
    try:
        _replace_json(
            Path(contract.value["fixed_paths"]["pending_path"]),
            pending,
            "Q5b pre-consumption pending",
        )
    except OSError as publication_error:
        raise ControllerPending(
            error.phase,
            error.condition,
            f"{error.detail}; pending publication failed: {publication_error}",
        ) from publication_error


def _require_preconsumption_artifact_absent(contract: Any) -> None:
    path = Path(contract.value["fixed_paths"]["artifact_path"])
    if os.path.lexists(path):
        raise ControllerPending(
            "FRESH_ROOT_ADMISSION",
            "Q5B_ARTIFACT_EXISTS_BEFORE_CONSUMPTION",
            str(path),
        )


def _load_capsule(
    contract: Any,
    preregistration: Any,
    expected_digest: str | None = None,
) -> dict[str, Any]:
    path = Path(contract.value["fixed_paths"]["capsule_path"])
    try:
        loaded = state.load_pretty_json(path, "Q5b capsule")
        capsule = state.validate_capsule(
            loaded.value, contract, preregistration.value
        )
    except ValueError as error:
        _terminal("Q5B_CAPSULE_REPLAY_DIFFERS", str(error))
    if expected_digest is not None and capsule["capsule_digest"] != expected_digest:
        _terminal("Q5B_CAPSULE_DIGEST_DIFFERS", expected_digest)
    return capsule


def _load_authorization(
    contract: Any,
    preregistration: Any,
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(contract.value["fixed_paths"]["authorization_path"])
    try:
        loaded = state.load_pretty_json(path, "Q5b authorization")
        return state.validate_authorization(
            loaded.value, contract, preregistration.value, capsule
        )
    except ValueError as error:
        _terminal("Q5B_AUTHORIZATION_REPLAY_DIFFERS", str(error))


def _load_consumption(
    contract: Any,
    preregistration: Any,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(contract.value["fixed_paths"]["consumption_path"])
    try:
        loaded = state.load_pretty_json(path, "Q5b consumption")
        return state.validate_consumption(
            loaded.value,
            contract,
            preregistration.value,
            capsule,
            authorization,
        )
    except ValueError as error:
        _terminal("Q5B_CONSUMPTION_REPLAY_DIFFERS", str(error))


def _preconsumption_replay(
    contract: Any,
    preregistration: Any,
    capsule: Mapping[str, Any],
) -> Any:
    rows = _source_closure_rows(contract)
    closure = state.source_closure_root(rows)
    if (
        rows != capsule["source_closure_rows"]
        or closure != capsule["source_closure_root"]
    ):
        raise ControllerPending(
            "SOURCE_CLOSURE",
            "Q5B_LIVE_SOURCE_CLOSURE_DIFFERS",
            closure,
        )
    replay = _replay_q5a(
        contract,
        preregistration,
        source_commit=capsule["source_commit"],
        source_closure_root=closure,
    )
    if replay.preflight_receipt != capsule["preflight_receipt"]:
        raise ControllerPending(
            "SEALED_Q5A_REPLAY",
            "Q5B_CAPSULE_PREFLIGHT_RECEIPT_DIFFERS",
            capsule["capsule_digest"],
        )
    return replay


def _q5a_artifact_postconsumption(contract: Any) -> bytes:
    path = Path(contract.value["q5a_inputs"]["artifact"])
    try:
        return _read_regular(path, "sealed public Q5a artifact", state.MAX_Q5A_ARTIFACT_BYTES)
    except ControllerPending as error:
        _terminal("Q5B_Q5A_ARTIFACT_UNAVAILABLE_AFTER_CONSUMPTION", str(error))


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
    try:
        _publish_json(
            Path(contract.value["fixed_paths"]["terminal_path"]),
            terminal,
            "Q5b invalidated terminal",
        )
    except OSError as error:
        _terminal("Q5B_INVALIDATED_TERMINAL_PUBLICATION_FAILED", str(error))
    _reconcile_trajectory(
        contract,
        capsule,
        authorization=authorization,
        consumption=consumption,
        terminal=terminal,
    )
    return terminal


def _replay_existing_terminal(
    contract: Any,
    preregistration: Any,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
) -> dict[str, Any]:
    terminal_path = Path(contract.value["fixed_paths"]["terminal_path"])
    try:
        terminal_loaded = state.load_pretty_json(terminal_path, "Q5b terminal")
    except ValueError as error:
        _terminal("Q5B_TERMINAL_REPLAY_DIFFERS", str(error))
    terminal = terminal_loaded.value
    if terminal.get("status") == wire.Q5B_SUCCESS_STATUS:
        artifact_path = Path(contract.value["fixed_paths"]["artifact_path"])
        try:
            artifact = _read_regular(
                artifact_path, "sealed Q5b artifact", state.MAX_Q5A_ARTIFACT_BYTES
            )
        except ControllerPending as error:
            _terminal("Q5B_TERMINAL_ARTIFACT_UNAVAILABLE", str(error))
        q5a_artifact = _q5a_artifact_postconsumption(contract)
        try:
            return state.validate_terminal(
                terminal,
                contract,
                preregistration.value,
                capsule,
                authorization,
                consumption,
                artifact_payload=artifact,
                q5a_artifact_payload=q5a_artifact,
            )
        except ValueError as error:
            _terminal("Q5B_TERMINAL_REPLAY_DIFFERS", str(error))
    try:
        return state.validate_terminal(
            terminal,
            contract,
            preregistration.value,
            capsule,
            authorization,
            consumption,
        )
    except ValueError as error:
        _terminal("Q5B_TERMINAL_REPLAY_DIFFERS", str(error))


def prepare_capsule(
    contract_path: Path,
    source_commit: str,
    *,
    run_id: str | None = None,
    created_at_unix_seconds: int | None = None,
) -> dict[str, Any]:
    contract, preregistration = _load_contract(contract_path)
    with _campaign_lock(contract):
        capsule_path = Path(contract.value["fixed_paths"]["capsule_path"])
        if capsule_path.exists():
            capsule = _load_capsule(contract, preregistration)
            fixed = contract.value["fixed_paths"]
            consumption_exists = Path(fixed["consumption_path"]).exists()
            terminal_exists = Path(fixed["terminal_path"]).exists()
            if terminal_exists and not consumption_exists:
                _terminal(
                    "Q5B_TERMINAL_WITHOUT_CONSUMPTION",
                    fixed["terminal_path"],
                )
            if consumption_exists:
                return capsule
            authorization = None
            if Path(fixed["authorization_path"]).exists():
                authorization = _load_authorization(
                    contract, preregistration, capsule
                )
            try:
                _preconsumption_replay(contract, preregistration, capsule)
                _require_preconsumption_artifact_absent(contract)
                _reconcile_trajectory(
                    contract, capsule, authorization=authorization
                )
            except ControllerPending as error:
                _publish_pending(
                    contract,
                    preregistration,
                    capsule,
                    error,
                    authorization=authorization,
                )
                raise
            _clear_pending(contract)
            return capsule
        rows = _source_closure_rows(contract)
        closure = state.source_closure_root(rows)
        replay = _replay_q5a(
            contract,
            preregistration,
            source_commit=source_commit,
            source_closure_root=closure,
        )
        fixed = contract.value["fixed_paths"]
        for key in (
            "authorization_path",
            "consumption_path",
            "pending_path",
            "artifact_path",
            "terminal_path",
        ):
            if Path(fixed[key]).exists():
                raise ControllerPending(
                    "FRESH_ROOT_ADMISSION",
                    "Q5B_FUTURE_STATE_ALREADY_EXISTS",
                    str(fixed[key]),
                )
        capsule = state.build_capsule(
            contract,
            preregistration.value,
            {
                "source_commit": source_commit,
                "source_closure_rows": rows,
                "source_closure_root": closure,
                "run_id": secrets.token_hex(16) if run_id is None else run_id,
                "created_at_unix_seconds": (
                    int(time.time())
                    if created_at_unix_seconds is None
                    else created_at_unix_seconds
                ),
                "preflight_receipt": replay.preflight_receipt,
            },
        )
        try:
            _publish_json(capsule_path, capsule, "Q5b capsule")
        except OSError as error:
            raise ControllerPending(
                "CAPSULE_PUBLICATION",
                "Q5B_CAPSULE_NOT_DURABLE",
                str(error),
            ) from error
        try:
            _reconcile_trajectory(contract, capsule)
        except ControllerPending as error:
            _publish_pending(contract, preregistration, capsule, error)
            raise
        return capsule


def record_authorization(
    contract_path: Path,
    capsule_digest: str,
) -> dict[str, Any]:
    contract, preregistration = _load_contract(contract_path)
    with _campaign_lock(contract):
        capsule = _load_capsule(contract, preregistration, capsule_digest)
        fixed = contract.value["fixed_paths"]
        authorization_path = Path(fixed["authorization_path"])
        if authorization_path.exists():
            authorization = _load_authorization(
                contract, preregistration, capsule
            )
            if Path(fixed["consumption_path"]).exists():
                return authorization
            try:
                _preconsumption_replay(contract, preregistration, capsule)
                _require_preconsumption_artifact_absent(contract)
                _reconcile_trajectory(
                    contract, capsule, authorization=authorization
                )
            except ControllerPending as error:
                _publish_pending(
                    contract,
                    preregistration,
                    capsule,
                    error,
                    authorization=authorization,
                )
                raise
            _clear_pending(contract)
            return authorization
        if Path(fixed["consumption_path"]).exists() or Path(fixed["terminal_path"]).exists():
            _terminal("Q5B_AUTHORIZATION_MISSING_AFTER_EXECUTION_STATE", capsule_digest)
        try:
            _preconsumption_replay(contract, preregistration, capsule)
            _require_preconsumption_artifact_absent(contract)
        except ControllerPending as error:
            _publish_pending(contract, preregistration, capsule, error)
            raise
        authorization = state.build_authorization(
            contract, preregistration.value, capsule
        )
        try:
            _publish_json(authorization_path, authorization, "Q5b authorization")
        except OSError as error:
            pending = ControllerPending(
                "AUTHORIZATION_PUBLICATION",
                "Q5B_AUTHORIZATION_NOT_DURABLE",
                str(error),
            )
            _publish_pending(contract, preregistration, capsule, pending)
            raise pending
        _clear_pending(contract)
        try:
            _reconcile_trajectory(
                contract, capsule, authorization=authorization
            )
        except ControllerPending as error:
            _publish_pending(
                contract,
                preregistration,
                capsule,
                error,
                authorization=authorization,
            )
            raise
        return authorization


def _not_authorized(capsule: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": state.NOT_AUTHORIZED,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_consumption_count": 0,
        "logical_campaign_count": 0,
        "q5b_action_audit": state.ZERO_ACTION_AUDIT,
        "purpose5_active_signer_count": 0,
        "purpose5_trust_genesis_pinned": False,
        "purpose5_current_key_status_accepted": False,
        "global_active_transition_performed": False,
    }


def run_v1(contract_path: Path, capsule_digest: str) -> dict[str, Any]:
    contract, preregistration = _load_contract(contract_path)
    with _campaign_lock(contract):
        capsule = _load_capsule(contract, preregistration, capsule_digest)
        fixed = contract.value["fixed_paths"]
        authorization_path = Path(fixed["authorization_path"])
        consumption_path = Path(fixed["consumption_path"])
        terminal_path = Path(fixed["terminal_path"])
        artifact_path = Path(fixed["artifact_path"])

        if not authorization_path.exists():
            if consumption_path.exists() or terminal_path.exists():
                _terminal("Q5B_AUTHORIZATION_MISSING_AFTER_EXECUTION_STATE", capsule_digest)
            try:
                _preconsumption_replay(contract, preregistration, capsule)
                _require_preconsumption_artifact_absent(contract)
                _reconcile_trajectory(contract, capsule)
            except ControllerPending as error:
                _publish_pending(contract, preregistration, capsule, error)
                raise
            _clear_pending(contract)
            return _not_authorized(capsule)
        authorization = _load_authorization(contract, preregistration, capsule)

        if terminal_path.exists():
            if not consumption_path.exists():
                _terminal("Q5B_TERMINAL_WITHOUT_CONSUMPTION", str(terminal_path))
            consumption = _load_consumption(
                contract, preregistration, capsule, authorization
            )
            _clear_pending(contract)
            terminal = _replay_existing_terminal(
                contract, preregistration, capsule, authorization, consumption
            )
            _reconcile_trajectory(
                contract,
                capsule,
                authorization=authorization,
                consumption=consumption,
                terminal=terminal,
            )
            return terminal

        if consumption_path.exists():
            consumption = _load_consumption(
                contract, preregistration, capsule, authorization
            )
            _clear_pending(contract)
            if not artifact_path.exists():
                return _seal_invalidated(
                    contract,
                    preregistration,
                    capsule,
                    authorization,
                    consumption,
                    phase="POST_CONSUMPTION_ARTIFACT",
                    code="Q5B_AMENDMENT_ARTIFACT_MISSING_AFTER_CONSUMPTION",
                    detail=str(artifact_path),
                )
            try:
                artifact = _read_regular(
                    artifact_path,
                    "sealed Q5b artifact",
                    state.MAX_Q5A_ARTIFACT_BYTES,
                )
                q5a_artifact = _q5a_artifact_postconsumption(contract)
                state.validate_artifact_binding(artifact, q5a_artifact, capsule)
            except (ControllerPending, ControllerTerminal, ValueError) as error:
                return _seal_invalidated(
                    contract,
                    preregistration,
                    capsule,
                    authorization,
                    consumption,
                    phase="POST_CONSUMPTION_ARTIFACT",
                    code="Q5B_AMENDMENT_ARTIFACT_INVALID_AFTER_CONSUMPTION",
                    detail=str(error),
                )
        else:
            try:
                replay = _preconsumption_replay(contract, preregistration, capsule)
                _require_preconsumption_artifact_absent(contract)
            except ControllerPending as error:
                _publish_pending(
                    contract,
                    preregistration,
                    capsule,
                    error,
                    authorization=authorization,
                )
                raise
            consumption = state.build_consumption(
                contract,
                preregistration.value,
                capsule,
                authorization,
            )
            try:
                _publish_json(consumption_path, consumption, "Q5b consumption")
            except OSError as error:
                if not consumption_path.exists():
                    pending = ControllerPending(
                        "CONSUMPTION_PUBLICATION",
                        "Q5B_CONSUMPTION_ABSENCE_PROVEN",
                        str(error),
                    )
                    _publish_pending(
                        contract,
                        preregistration,
                        capsule,
                        pending,
                        authorization=authorization,
                    )
                    raise pending
                _terminal("Q5B_CONSUMPTION_PUBLICATION_OUTCOME_UNKNOWN", str(error))
            consumption = _load_consumption(
                contract, preregistration, capsule, authorization
            )
            _clear_pending(contract)
            artifact = replay.artifact_payload
            q5a_artifact = _q5a_artifact_postconsumption(contract)
            try:
                _publish_bytes(artifact_path, artifact, "Q5b amendment artifact")
            except OSError as error:
                try:
                    durable = _read_regular(
                        artifact_path,
                        "Q5b artifact after publication error",
                        state.MAX_Q5A_ARTIFACT_BYTES,
                    )
                    state.validate_artifact_binding(durable, q5a_artifact, capsule)
                    artifact = durable
                except (ControllerPending, ValueError) as replay_error:
                    return _seal_invalidated(
                        contract,
                        preregistration,
                        capsule,
                        authorization,
                        consumption,
                        phase="ARTIFACT_PUBLICATION",
                        code="Q5B_AMENDMENT_ARTIFACT_PUBLICATION_FAILED",
                        detail=f"{error}; {replay_error}",
                    )

        terminal = state.build_complete_terminal(
            contract,
            preregistration.value,
            capsule,
            authorization,
            consumption,
            artifact,
            q5a_artifact,
        )
        try:
            _publish_json(terminal_path, terminal, "Q5b complete terminal")
        except OSError as error:
            raise ControllerPartial(
                f"sealed artifact is durable; terminal publication failed: {error}"
            ) from error
        _reconcile_trajectory(
            contract,
            capsule,
            authorization=authorization,
            consumption=consumption,
            terminal=terminal,
        )
        return terminal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-capsule")
    prepare.add_argument("--source-commit", required=True)
    prepare.add_argument("--run-id")
    prepare.add_argument("--created-at-unix-seconds", type=int)
    record = sub.add_parser("record-authorization")
    record.add_argument("--capsule-digest", required=True)
    run = sub.add_parser("run")
    run.add_argument("--capsule-digest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare-capsule":
            result = prepare_capsule(
                args.contract,
                args.source_commit,
                run_id=args.run_id,
                created_at_unix_seconds=args.created_at_unix_seconds,
            )
        elif args.command == "record-authorization":
            result = record_authorization(args.contract, args.capsule_digest)
        else:
            result = run_v1(args.contract, args.capsule_digest)
    except ControllerPending as error:
        print(
            json.dumps(
                {
                    "status": state.PENDING_NOT_CONSUMED,
                    "phase": error.phase,
                    "condition": error.condition,
                    "detail": error.detail,
                },
                sort_keys=True,
            )
        )
        return PENDING_EXIT_CODE
    except ControllerPartial as error:
        print(
            json.dumps(
                {"status": state.PARTIAL_FINALIZATION_REQUIRED, "detail": error.detail},
                sort_keys=True,
            )
        )
        return TERMINAL_EXIT_CODE
    except ControllerTerminal as error:
        print(
            json.dumps(
                {"status": "Q5B_TERMINAL_ERROR", "code": error.code, "detail": error.detail},
                sort_keys=True,
            )
        )
        return TERMINAL_EXIT_CODE
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return NOT_AUTHORIZED_EXIT_CODE if result.get("status") == state.NOT_AUTHORIZED else 0


if __name__ == "__main__":
    raise SystemExit(main())
