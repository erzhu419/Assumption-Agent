#!/usr/bin/env python3
"""Zero-Popen controller for the Q3 bounded certificate admission replay.

``prepare-capsule`` is target-free and does not create an authorization,
consumption record, artifact, or terminal.  The other entry points exist for a
later exact owner authorization; this engineering-freeze invocation must stop
after capsule preparation.
"""

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

state = importlib.import_module("hegel_machine.phase3_q3_completion_replay_v1")
wire = importlib.import_module(
    "hegel_machine.phase3_q3_bounded_quotient_certificate_v1"
)

DEFAULT_CONTRACT: Final = (
    PROJECT_ROOT / "config/phase3_q3_bounded_quotient_certificate_actual_v1.json"
)
PENDING_EXIT_CODE: Final = 75
NOT_AUTHORIZED_EXIT_CODE: Final = 4
TERMINAL_EXIT_CODE: Final = 2


class ControllerError(RuntimeError):
    """Base class for classified Q3 controller failures."""


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
        payload = path.read_bytes()
    except OSError as error:
        raise ControllerPending(
            "LOCAL_READ_ONLY_REPLAY",
            "BOUND_LOCAL_FILE_UNAVAILABLE",
            f"{label}: {error}",
        ) from error
    return payload


def _load_contract(path: Path) -> Any:
    try:
        contract = state.load_contract(path)
        preregistration = state.load_preregistration(
            Path(contract.value["preregistration_path"])
        )
    except (OSError, ValueError) as error:
        raise ControllerPending(
            "LOCAL_CONTRACT",
            "Q3_CONTRACT_OR_PREREGISTRATION_UNAVAILABLE",
            str(error),
        ) from error
    if preregistration.raw_sha256 != contract.value["preregistration_raw_sha256"]:
        raise ControllerPending(
            "LOCAL_CONTRACT",
            "Q3_PREREGISTRATION_RAW_IDENTITY_DIFFERS",
            str(preregistration.path),
        )
    return contract, preregistration


def _source_closure_rows(contract: Any) -> list[dict[str, Any]]:
    project_root = Path(contract.value["fixed_paths"]["project_root"])
    rows: list[dict[str, Any]] = []
    for relative in state.Q3_SOURCE_PATHS:
        path = project_root / relative
        payload = _read_regular(path, f"Q3 source {relative}")
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ControllerPending(
                "SOURCE_CLOSURE",
                "Q3_SOURCE_PATH_UNAVAILABLE",
                f"{relative}: {error}",
            ) from error
        if resolved != path:
            raise ControllerPending(
                "SOURCE_CLOSURE",
                "Q3_SOURCE_PATH_IDENTITY_DIFFERS",
                relative,
            )
        rows.append(
            {
                "relative_path": relative,
                "byte_count": len(payload),
                "raw_sha256": state.raw_sha256(payload),
            }
        )
    state.validate_source_closure_rows(rows)
    state.source_closure_root(rows)
    return rows


def _read_q2_selected(contract: Any) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for key in ("capsule", "authorization", "consumption", "terminal", "artifact"):
        maximum = (
            state.MAX_Q2_ARTIFACT_BYTES
            if key == "artifact"
            else state.MAX_JSON_BYTES
        )
        result[key] = _read_regular(
            Path(contract.value["q2_inputs"][key]),
            f"sealed Q2 {key}",
            maximum,
        )
    return result


def _replay_q2(contract: Any, preregistration: Any) -> Any:
    selected = _read_q2_selected(contract)
    try:
        return state.replay_q2_predecessor_v1(
            preregistration.value,
            capsule_payload=selected["capsule"],
            authorization_payload=selected["authorization"],
            consumption_payload=selected["consumption"],
            terminal_payload=selected["terminal"],
            artifact_payload=selected["artifact"],
        )
    except ValueError as error:
        raise ControllerPending(
            "SEALED_Q2_REPLAY",
            "Q2_SEALED_CHAIN_REPLAY_DIFFERS",
            str(error),
        ) from error


def _replay_capsule_source(contract: Any, capsule: Mapping[str, Any]) -> None:
    rows = _source_closure_rows(contract)
    if (
        rows != capsule["source_closure_rows"]
        or state.source_closure_root(rows) != capsule["source_closure_root"]
    ):
        raise ControllerPending(
            "SOURCE_CLOSURE",
            "Q3_CAPSULE_SOURCE_CLOSURE_DIFFERS",
            str(contract.path),
        )


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_new(path: Path, payload: bytes, label: str) -> None:
    descriptor: int | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short durable write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            if descriptor is not None:
                os.close(descriptor)
                descriptor = None
        _fsync_parent(path)
    except OSError as error:
        raise ControllerPending(
            "LOCAL_PUBLICATION",
            "Q3_LOCAL_PUBLICATION_UNAVAILABLE",
            f"{label}: {error}",
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _publish_json(path: Path, value: Mapping[str, Any], label: str) -> None:
    _publish_new(path, state.canonical_json_bytes(dict(value)), label)


def _append_trajectory(contract: Any, event: str, **fields: object) -> None:
    path = Path(contract.value["fixed_paths"]["trajectory_path"])
    row = {
        "schema_version": "hegel-phase3a-q3-trajectory/1",
        "event": event,
        "time_ns": time.time_ns(),
        **fields,
    }
    payload = state.canonical_json_bytes(row)
    descriptor: int | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short trajectory write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        _fsync_parent(path)
    except OSError as error:
        raise ControllerPending(
            "TRAJECTORY_PUBLICATION",
            "Q3_TRAJECTORY_PUBLICATION_UNAVAILABLE",
            f"{event}: {error}",
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_json_if_present(path: Path, label: str) -> dict[str, Any] | None:
    try:
        exists = path.exists()
    except OSError as error:
        raise ControllerPending(
            "LOCAL_READ_ONLY_REPLAY",
            "Q3_LOCAL_PATH_STATUS_UNAVAILABLE",
            f"{label}: {error}",
        ) from error
    if not exists:
        return None
    return state.strict_json_bytes(_read_regular(path, label), label)


def _read_capsule(contract: Any, preregistration: Any, digest: str) -> dict[str, Any]:
    path = Path(contract.value["fixed_paths"]["capsule_path"])
    try:
        value = _read_json_if_present(path, "Q3 capsule")
    except (ControllerPending, ValueError) as error:
        raise ControllerPending(
            "CAPSULE_REPLAY",
            "Q3_CAPSULE_REPLAY_DIFFERS",
            str(error),
        ) from error
    if value is None:
        raise ControllerPending(
            "CAPSULE_REPLAY",
            "Q3_CAPSULE_UNAVAILABLE",
            str(path),
        )
    try:
        checked = state.validate_capsule(value, contract, preregistration.value)
    except ValueError as error:
        raise ControllerPending(
            "CAPSULE_REPLAY",
            "Q3_CAPSULE_REPLAY_DIFFERS",
            str(error),
        ) from error
    if checked["capsule_digest"] != digest:
        raise ControllerPending(
            "CAPSULE_REPLAY",
            "Q3_CAPSULE_DIGEST_DIFFERS",
            digest,
        )
    return checked


def _read_authorization(
    contract: Any,
    preregistration: Any,
    capsule: Mapping[str, Any],
) -> dict[str, Any] | None:
    try:
        value = _read_json_if_present(
            Path(contract.value["fixed_paths"]["authorization_path"]),
            "Q3 authorization",
        )
    except (ControllerPending, ValueError) as error:
        _terminal("Q3_AUTHORIZATION_REPLAY_DIFFERS", str(error))
    if value is None:
        return None
    try:
        return state.validate_authorization(
            value, contract, preregistration.value, capsule
        )
    except ValueError as error:
        _terminal("Q3_AUTHORIZATION_REPLAY_DIFFERS", str(error))


def _read_consumption(
    contract: Any,
    preregistration: Any,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any] | None:
    try:
        value = _read_json_if_present(
            Path(contract.value["fixed_paths"]["consumption_path"]),
            "Q3 consumption",
        )
    except (ControllerPending, ValueError) as error:
        _terminal("Q3_CONSUMPTION_REPLAY_DIFFERS", str(error))
    if value is None:
        return None
    try:
        return state.validate_consumption(
            value,
            contract,
            preregistration.value,
            capsule,
            authorization,
        )
    except ValueError as error:
        _terminal("Q3_CONSUMPTION_REPLAY_DIFFERS", str(error))


def _read_terminal(
    contract: Any,
    preregistration: Any,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = Path(contract.value["fixed_paths"]["terminal_path"])
    try:
        value = _read_json_if_present(path, "Q3 terminal")
    except (ControllerPending, ValueError) as error:
        _terminal("Q3_TERMINAL_REPLAY_DIFFERS", str(error))
    if value is None:
        return None

    artifact_path = Path(contract.value["fixed_paths"]["artifact_path"])
    artifact_payload: bytes | None = None
    if value.get("status") == wire.Q3_SUCCESS_STATUS:
        try:
            artifact_payload = _read_regular(
                artifact_path,
                "Q3 candidate artifact",
                state.MAX_Q2_ARTIFACT_BYTES,
            )
            state.validate_candidate_binding(artifact_payload, capsule)
        except (ControllerPending, ValueError) as error:
            _terminal("Q3_TERMINAL_ARTIFACT_REPLAY_DIFFERS", str(error))
    elif artifact_path.exists():
        try:
            possible = _read_regular(
                artifact_path,
                "Q3 candidate artifact",
                state.MAX_Q2_ARTIFACT_BYTES,
            )
            state.validate_candidate_binding(possible, capsule)
        except (ControllerPending, ValueError):
            pass
        else:
            _terminal(
                "Q3_INVALIDATED_TERMINAL_VALID_ARTIFACT_COEXISTS",
                str(artifact_path),
            )
    try:
        return state.validate_terminal(
            value,
            contract,
            preregistration.value,
            capsule,
            authorization,
            consumption,
            artifact_payload=artifact_payload,
        )
    except ValueError as error:
        _terminal("Q3_TERMINAL_REPLAY_DIFFERS", str(error))


@contextmanager
def _campaign_lock(contract: Any) -> Iterator[None]:
    path = Path(contract.value["fixed_paths"]["campaign_lock_path"])
    descriptor: int | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise ControllerPending(
            "CAMPAIGN_LOCK",
            "Q3_CAMPAIGN_LOCK_UNAVAILABLE",
            str(error),
        ) from error
    try:
        yield
    finally:
        assert descriptor is not None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(descriptor)
        except OSError:
            pass


def _fresh_prepare_absence(contract: Any) -> None:
    fixed = contract.value["fixed_paths"]
    for key in (
        "authorization_path",
        "consumption_path",
        "terminal_path",
        "artifact_path",
    ):
        path = Path(fixed[key])
        if path.exists():
            raise ControllerPending(
                "CAPSULE_PREPARATION",
                "Q3_FUTURE_STATE_ALREADY_EXISTS",
                str(path),
            )


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
    if capsule_path.exists():
        try:
            value = state.strict_json_bytes(
                _read_regular(capsule_path, "Q3 capsule"), "Q3 capsule"
            )
            capsule = state.validate_capsule(
                value, contract, preregistration.value
            )
        except (ControllerPending, ValueError) as error:
            raise ControllerPending(
                "CAPSULE_REPLAY",
                "Q3_CAPSULE_REPLAY_DIFFERS",
                str(error),
            ) from error
        _replay_capsule_source(contract, capsule)
        replay = _replay_q2(contract, preregistration)
        if replay.preflight_receipt != capsule["preflight_receipt"]:
            raise ControllerPending(
                "CAPSULE_REPLAY",
                "Q3_PREFLIGHT_RECEIPT_DIFFERS",
                str(capsule_path),
            )
        return capsule
    _fresh_prepare_absence(contract)
    replay = _replay_q2(contract, preregistration)
    rows = _source_closure_rows(contract)
    inputs = {
        "schema_version": state.CAPSULE_INPUTS_SCHEMA,
        "source_closure_rows": rows,
        "source_closure_root": state.source_closure_root(rows),
        "source_commit": source_commit,
        "run_id": run_id or secrets.token_hex(16),
        "created_at_unix_seconds": (
            int(time.time())
            if created_at_unix_seconds is None
            else created_at_unix_seconds
        ),
        "preflight_receipt": replay.preflight_receipt,
    }
    capsule = state.build_capsule(contract, preregistration.value, inputs)
    _publish_json(capsule_path, capsule, "Q3 capsule")
    published = state.strict_json_bytes(
        _read_regular(capsule_path, "Q3 capsule"), "Q3 capsule"
    )
    state.validate_capsule(published, contract, preregistration.value)
    _append_trajectory(
        contract,
        "CAPSULE_PREPARED_NOT_AUTHORIZED",
        capsule_digest=capsule["capsule_digest"],
        authorization_consumption_count=0,
        logical_campaign_count=0,
        q3_action_audit=state.ZERO_ACTION_AUDIT,
    )
    return capsule


def record_authorization(contract_path: Path, capsule_digest: str) -> dict[str, Any]:
    contract, preregistration = _load_contract(contract_path)
    with _campaign_lock(contract):
        capsule = _read_capsule(contract, preregistration, capsule_digest)
        _replay_capsule_source(contract, capsule)
        replay = _replay_q2(contract, preregistration)
        if replay.preflight_receipt != capsule["preflight_receipt"]:
            raise ControllerPending(
                "AUTHORIZATION_PRECONSUMPTION",
                "Q3_PREFLIGHT_RECEIPT_DIFFERS",
                capsule_digest,
            )
        fixed = contract.value["fixed_paths"]
        if any(
            Path(fixed[key]).exists()
            for key in ("consumption_path", "terminal_path", "artifact_path")
        ):
            _terminal(
                "Q3_AUTHORIZATION_RECORDING_AFTER_EXECUTION_STATE",
                capsule_digest,
            )
        existing = _read_authorization(
            contract, preregistration, capsule
        )
        if existing is not None:
            return existing
        authorization = state.build_authorization(
            contract, preregistration.value, capsule
        )
        _publish_json(
            Path(fixed["authorization_path"]),
            authorization,
            "Q3 authorization",
        )
        checked = _read_authorization(contract, preregistration, capsule)
        if checked != authorization:
            _terminal("Q3_AUTHORIZATION_READBACK_DIFFERS", capsule_digest)
        _append_trajectory(
            contract,
            "AUTHORIZATION_RECORDED_NOT_CONSUMED",
            capsule_digest=capsule_digest,
            authorization_digest=authorization["authorization_digest"],
            authorization_consumption_count=0,
            logical_campaign_count=0,
            q3_action_audit=state.ZERO_ACTION_AUDIT,
        )
        return authorization


def _publish_pending(
    contract: Any,
    preregistration: Any,
    capsule: Mapping[str, Any],
    error: ControllerPending,
    authorization: Mapping[str, Any] | None,
) -> dict[str, Any]:
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
    try:
        if path.exists():
            path.unlink()
            _fsync_parent(path)
        _publish_json(path, pending, "Q3 pending")
        _append_trajectory(
            contract,
            "PRECONSUMPTION_PENDING",
            capsule_digest=capsule["capsule_digest"],
            condition=error.condition,
            authorization_consumption_count=0,
            logical_campaign_count=0,
            q3_action_audit=state.ZERO_ACTION_AUDIT,
        )
    except (ControllerPending, OSError) as publication_error:
        return {
            "schema_version": "hegel-phase3a-q3-pending-publication-unavailable/1",
            "status": state.PENDING_NOT_CONSUMED,
            "capsule_digest": capsule["capsule_digest"],
            "authorization_digest": (
                authorization["authorization_digest"]
                if authorization is not None
                else None
            ),
            "authorization_consumption_count": 0,
            "logical_campaign_count": 0,
            "q3_action_audit": state.ZERO_ACTION_AUDIT,
            "pending_publication_completed": False,
            "detail": str(publication_error),
        }
    return pending


def _clear_pending_before_consumption(contract: Any) -> None:
    path = Path(contract.value["fixed_paths"]["pending_path"])
    try:
        if path.exists():
            path.unlink()
            _fsync_parent(path)
    except OSError as error:
        raise ControllerPending(
            "PRECONSUMPTION_PENDING_CLEAR",
            "Q3_STALE_PENDING_CLEAR_UNAVAILABLE",
            str(error),
        ) from error


def _partial_finalization_result(
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    error: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": "hegel-phase3a-q3-partial-finalization/1",
        "status": state.PARTIAL_FINALIZATION_REQUIRED,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "consumption_record_digest": consumption["record_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "exact_artifact_to_terminal_reentry_allowed": True,
        "q2_replay_or_candidate_reentry_allowed": False,
        "q3_action_audit": state.ZERO_ACTION_AUDIT,
        "detail": str(error),
    }


def _unsealed_terminal_result(
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    error: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": "hegel-phase3a-q3-unsealed-terminal/1",
        "status": state.INVALIDATED_TERMINAL_NO_RESUME,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "consumption_record_digest": consumption["record_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "durable_terminal_published": False,
        "terminal_no_resume": True,
        "q3_action_audit": state.ZERO_ACTION_AUDIT,
        "detail": str(error),
    }


def _consumed_state_unreadable_result(
    capsule: Mapping[str, Any] | str,
    error: Exception,
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    capsule_digest = (
        capsule["capsule_digest"] if not isinstance(capsule, str) else capsule
    )
    return {
        "schema_version": "hegel-phase3a-q3-consumed-state-unreadable/1",
        "status": state.INVALIDATED_TERMINAL_NO_RESUME,
        "capsule_digest": capsule_digest,
        "authorization_digest": (
            authorization["authorization_digest"]
            if authorization is not None
            else None
        ),
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "counts_are_conservative": True,
        "durable_terminal_published": False,
        "terminal_no_resume": True,
        "q3_action_audit": state.ZERO_ACTION_AUDIT,
        "detail": str(error),
    }


def _publish_invalidated(
    contract: Any,
    preregistration: Any,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    error: ControllerTerminal,
) -> dict[str, Any]:
    artifact_path = Path(contract.value["fixed_paths"]["artifact_path"])
    if artifact_path.exists():
        try:
            artifact = _read_regular(
                artifact_path,
                "Q3 candidate artifact",
                state.MAX_Q2_ARTIFACT_BYTES,
            )
            state.validate_candidate_binding(artifact, capsule)
        except (ControllerPending, ValueError):
            pass
        else:
            return _partial_finalization_result(
                capsule, authorization, consumption, error
            )
    terminal = state.build_invalidated_terminal(
        contract,
        preregistration.value,
        capsule,
        authorization,
        consumption,
        failure_phase="POST_CONSUMPTION_FINALIZATION",
        failure_code=error.code,
        failure_detail=error.detail,
    )
    path = Path(contract.value["fixed_paths"]["terminal_path"])
    try:
        if not path.exists():
            _publish_json(path, terminal, "Q3 invalidated terminal")
        replayed = _read_terminal(
            contract,
            preregistration,
            capsule,
            authorization,
            consumption,
        )
        if replayed is None:
            raise ControllerTerminal(
                "Q3_INVALIDATED_TERMINAL_READBACK_MISSING", str(path)
            )
    except (ControllerPending, ControllerTerminal, OSError) as publication_error:
        return _unsealed_terminal_result(
            capsule, authorization, consumption, publication_error
        )
    try:
        _append_trajectory(
            contract,
            "INVALIDATED_TERMINAL_NO_RESUME",
            capsule_digest=capsule["capsule_digest"],
            failure_code=error.code,
            authorization_consumption_count=1,
            logical_campaign_count=1,
            q3_action_audit=state.ZERO_ACTION_AUDIT,
        )
    except (ControllerPending, OSError):
        pass
    return replayed


def _run_v1_locked(
    contract: Any,
    preregistration: Any,
    capsule_digest: str,
) -> dict[str, Any]:
    with _campaign_lock(contract):
        fixed = contract.value["fixed_paths"]
        consumption_path = Path(fixed["consumption_path"])
        artifact_path = Path(fixed["artifact_path"])
        terminal_path = Path(fixed["terminal_path"])
        execution_state_observed = any(
            path.exists() for path in (consumption_path, artifact_path, terminal_path)
        )
        try:
            capsule = _read_capsule(contract, preregistration, capsule_digest)
        except ControllerPending as error:
            if execution_state_observed:
                return _consumed_state_unreadable_result(
                    capsule_digest, error
                )
            raise
        try:
            authorization = _read_authorization(
                contract, preregistration, capsule
            )
        except ControllerTerminal as error:
            if execution_state_observed:
                return _consumed_state_unreadable_result(capsule, error)
            raise
        if authorization is None:
            if execution_state_observed:
                return _consumed_state_unreadable_result(
                    capsule,
                    ControllerTerminal(
                        "Q3_EXECUTION_STATE_WITHOUT_AUTHORIZATION",
                        capsule_digest,
                    ),
                )
            return {
                "schema_version": "hegel-phase3a-q3-not-authorized/1",
                "status": state.NOT_AUTHORIZED,
                "capsule_digest": capsule_digest,
                "authorization_consumption_count": 0,
                "logical_campaign_count": 0,
                "q3_action_audit": state.ZERO_ACTION_AUDIT,
            }
        try:
            consumption = _read_consumption(
                contract, preregistration, capsule, authorization
            )
        except ControllerTerminal as error:
            return _consumed_state_unreadable_result(
                capsule, error, authorization
            )
        if terminal_path.exists():
            if consumption is None:
                return _consumed_state_unreadable_result(
                    capsule,
                    ControllerTerminal(
                        "Q3_TERMINAL_WITHOUT_CONSUMPTION", str(terminal_path)
                    ),
                    authorization,
                )
            try:
                terminal = _read_terminal(
                    contract,
                    preregistration,
                    capsule,
                    authorization,
                    consumption,
                )
            except ControllerTerminal as error:
                try:
                    artifact = _read_regular(
                        artifact_path,
                        "Q3 candidate artifact",
                        state.MAX_Q2_ARTIFACT_BYTES,
                    )
                    state.validate_candidate_binding(artifact, capsule)
                except (ControllerPending, ValueError):
                    return _unsealed_terminal_result(
                        capsule, authorization, consumption, error
                    )
                return _partial_finalization_result(
                    capsule, authorization, consumption, error
                )
            if terminal is None:
                return _partial_finalization_result(
                    capsule,
                    authorization,
                    consumption,
                    ControllerTerminal(
                        "Q3_TERMINAL_READBACK_MISSING", str(terminal_path)
                    ),
                )
            return terminal
        if consumption is None:
            if artifact_path.exists():
                return _consumed_state_unreadable_result(
                    capsule,
                    ControllerTerminal(
                        "Q3_CANDIDATE_WITHOUT_CONSUMPTION", str(artifact_path)
                    ),
                    authorization,
                )
            try:
                _replay_capsule_source(contract, capsule)
                replay = _replay_q2(contract, preregistration)
                if replay.preflight_receipt != capsule["preflight_receipt"]:
                    raise ControllerPending(
                        "PRECONSUMPTION_REPLAY",
                        "Q3_PREFLIGHT_RECEIPT_DIFFERS",
                        capsule_digest,
                    )
                _clear_pending_before_consumption(contract)
            except ControllerPending as error:
                return _publish_pending(
                    contract, preregistration, capsule, error, authorization
                )
            consumption = state.build_consumption(
                contract,
                preregistration.value,
                capsule,
                authorization,
            )
            try:
                _publish_json(
                    consumption_path,
                    consumption,
                    "Q3 consumption",
                )
            except ControllerPending as error:
                if not consumption_path.exists():
                    return _publish_pending(
                        contract,
                        preregistration,
                        capsule,
                        error,
                        authorization,
                    )
            try:
                checked = _read_consumption(
                    contract, preregistration, capsule, authorization
                )
            except ControllerTerminal as error:
                return _consumed_state_unreadable_result(
                    capsule, error, authorization
                )
            if checked != consumption:
                return _consumed_state_unreadable_result(
                    capsule,
                    ControllerTerminal(
                        "Q3_CONSUMPTION_READBACK_DIFFERS", capsule_digest
                    ),
                    authorization,
                )
            consumption = checked
            try:
                _publish_new(
                    artifact_path,
                    replay.candidate.canonical_bytes,
                    "Q3 candidate artifact",
                )
            except ControllerPending as error:
                return _publish_invalidated(
                    contract,
                    preregistration,
                    capsule,
                    authorization,
                    consumption,
                    ControllerTerminal(
                        "Q3_CANDIDATE_PUBLICATION_FAILED_AFTER_CONSUMPTION",
                        str(error),
                    ),
                )
            try:
                _append_trajectory(
                    contract,
                    "AUTHORIZATION_CONSUMED_CANDIDATE_SEALED",
                    capsule_digest=capsule_digest,
                    authorization_consumption_count=1,
                    logical_campaign_count=1,
                    q3_action_audit=state.ZERO_ACTION_AUDIT,
                )
            except (ControllerPending, OSError):
                pass
        if not artifact_path.exists():
            return _publish_invalidated(
                contract,
                preregistration,
                capsule,
                authorization,
                consumption,
                ControllerTerminal(
                    "Q3_CANDIDATE_ARTIFACT_MISSING_AFTER_CONSUMPTION",
                    str(artifact_path),
                ),
            )
        try:
            artifact = _read_regular(
                artifact_path, "Q3 candidate artifact", state.MAX_Q2_ARTIFACT_BYTES
            )
            state.validate_candidate_binding(artifact, capsule)
        except (ControllerPending, ValueError) as error:
            return _publish_invalidated(
                contract,
                preregistration,
                capsule,
                authorization,
                consumption,
                ControllerTerminal(
                    "Q3_CANDIDATE_ARTIFACT_UNVERIFIABLE_AFTER_CONSUMPTION",
                    str(error),
                ),
            )
        try:
            terminal = state.build_complete_terminal(
                contract,
                preregistration.value,
                capsule,
                authorization,
                consumption,
                artifact,
            )
            _publish_json(terminal_path, terminal, "Q3 complete terminal")
            replayed_terminal = _read_terminal(
                contract,
                preregistration,
                capsule,
                authorization,
                consumption,
            )
            if replayed_terminal is None:
                raise ControllerTerminal(
                    "Q3_COMPLETE_TERMINAL_READBACK_MISSING", str(terminal_path)
                )
        except (ControllerPending, ControllerTerminal, ValueError) as error:
            return _partial_finalization_result(
                capsule, authorization, consumption, error
            )
        try:
            _append_trajectory(
                contract,
                "Q3_ADMISSION_COMPLETE",
                capsule_digest=capsule_digest,
                candidate_root=replayed_terminal["outcome"]["candidate_root"],
                authorization_consumption_count=1,
                logical_campaign_count=1,
                q3_action_audit=state.ZERO_ACTION_AUDIT,
            )
        except (ControllerPending, OSError):
            pass
        return replayed_terminal


def run_v1(contract_path: Path, capsule_digest: str) -> dict[str, Any]:
    contract, preregistration = _load_contract(contract_path)
    fixed = contract.value["fixed_paths"]
    execution_paths = tuple(
        Path(fixed[key])
        for key in ("consumption_path", "artifact_path", "terminal_path")
    )
    try:
        return _run_v1_locked(contract, preregistration, capsule_digest)
    except ControllerPending as error:
        if any(path.exists() for path in execution_paths):
            return _consumed_state_unreadable_result(capsule_digest, error)
        raise


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
                args.contract,
                args.source_commit,
                run_id=args.run_id,
                created_at_unix_seconds=args.created_at_unix_seconds,
            )
        elif args.command == "record-authorization":
            result = record_authorization(args.contract, args.capsule_sha256)
        else:
            result = run_v1(args.contract, args.capsule_sha256)
    except ControllerPending as error:
        result = {
            "schema_version": "hegel-phase3a-q3-controller-pending/1",
            "status": state.PENDING_NOT_CONSUMED,
            "phase": error.phase,
            "condition": error.condition,
            "detail": error.detail,
            "authorization_consumption_count": 0,
            "logical_campaign_count": 0,
            "q3_action_audit": state.ZERO_ACTION_AUDIT,
        }
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return PENDING_EXIT_CODE
    except ControllerTerminal as error:
        result = {
            "schema_version": "hegel-phase3a-q3-controller-terminal/1",
            "status": state.INVALIDATED_TERMINAL_NO_RESUME,
            "failure_code": error.code,
            "failure_detail": error.detail,
            "terminal_no_resume": True,
        }
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return TERMINAL_EXIT_CODE
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    if result.get("status") == state.NOT_AUTHORIZED:
        return NOT_AUTHORIZED_EXIT_CODE
    if result.get("status") in (
        state.PENDING_NOT_CONSUMED,
        state.PARTIAL_FINALIZATION_REQUIRED,
    ):
        return PENDING_EXIT_CODE
    if result.get("status") == state.INVALIDATED_TERMINAL_NO_RESUME:
        return TERMINAL_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
