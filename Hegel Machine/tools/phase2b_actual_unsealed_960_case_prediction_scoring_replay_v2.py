#!/usr/bin/env python3
"""Frozen replay-only controller for the Phase-2B-C1 actual-960 slice.

``prepare-capsule`` and ``preflight`` read only the checked-in public source
closure and preregistered dependency files.  They never stat or read the two
future actual inputs.  ``run`` accepts those inputs only at their contract-
fixed paths.  Before both exist and the canonical upstream admission is fully
true, it publishes same-capsule pre-consumption PENDING without consuming the
authorization.

The actual action is deliberately narrower than the historical campaign: it
imports already sealed R01--R16, recognizer, and baseline predecessor results,
then decodes and evaluates one scorer bundle in-process.  It invokes no target
producer, recognizer, baseline, subprocess, Docker API, or network API.
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

core = importlib.import_module(
    "hegel_machine.phase2b_actual_unsealed_960_prediction_scoring_replay_core_v2"
)
state = importlib.import_module(
    "hegel_machine.phase2b_actual_unsealed_960_case_prediction_scoring_"
    "completion_replay_v2"
)

DEFAULT_CONTRACT: Final = (
    PROJECT_ROOT
    / "config/phase2b_actual_unsealed_960_case_prediction_scoring_replay_actual_v2.json"
)
UPSTREAM_ADMISSION_SCHEMA_V2: Final = (
    "hegel-phase2b-c1-actual-unsealed-960-case-prediction-scoring-replay-"
    "upstream-admission-manifest/2"
)
BASELINE_IDS_IN_ORDER_V2: Final = (
    "embedding_nearest_prototype",
    "frozen_llm_semantic_only",
    "flat_learned_typed",
)
MAXIMUM_UPSTREAM_ADMISSION_BYTES_V2: Final = 1024 * 1024
MAXIMUM_SCORER_BUNDLE_BYTES_V2: Final = 512 * 1024 * 1024
PENDING_EXIT_CODE: Final = 75
NOT_AUTHORIZED_EXIT_CODE: Final = 4
TERMINAL_EXIT_CODE: Final = 2

_UPSTREAM_FIELDS_V2: Final = {
    "schema_version",
    "capsule_digest",
    "authoritative_r16_bundle_id",
    "campaign_id",
    "attempt_id",
    "recognizer_result_id",
    "baseline_result_ids",
    "prediction_archive_id",
    "answer_manifest_id",
    "scorer_bundle_id",
    "scorer_bundle_raw_sha256",
    "authoritative_schema_verifier_available",
    "authoritative_r01_r16_bundle_verified",
    "external_custodian_checkpoint_verified",
    "three_baseline_outputs_verified",
    "exact_scale_bridge_verified",
    "recognizer_executor_binding_verified",
    "scorer_executor_binding_verified",
}


class ControllerErrorV2(RuntimeError):
    """Base class for classified controller results."""


class ControllerPendingV2(ControllerErrorV2):
    def __init__(self, phase: str, condition: str, detail: str) -> None:
        super().__init__(f"{phase}/{condition}: {detail}")
        self.phase = phase
        self.condition = condition
        self.detail = detail


class ControllerTerminalV2(ControllerErrorV2):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _terminal(code: str, detail: str) -> NoReturn:
    raise ControllerTerminalV2(code, detail)


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_regular_once(path: Path, label: str, maximum_bytes: int) -> bytes:
    try:
        info = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise OSError("not one regular non-symlink file")
        if not 1 <= info.st_size <= maximum_bytes:
            raise OSError(f"size is outside 1..{maximum_bytes} bytes")
        payload = path.read_bytes()
    except OSError as error:
        raise ControllerPendingV2(
            "PRECONSUMPTION_INPUT",
            "BOUND_LOCAL_FILE_UNAVAILABLE",
            f"{label}: {error}",
        ) from error
    if len(payload) != info.st_size:
        raise ControllerPendingV2(
            "PRECONSUMPTION_INPUT",
            "BOUND_LOCAL_FILE_CHANGED_DURING_READ",
            label,
        )
    return payload


def _strict_canonical_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ControllerPendingV2(
            "UPSTREAM_ADMISSION",
            "UPSTREAM_ADMISSION_NOT_CANONICAL_JSON",
            str(error),
        ) from error
    if type(value) is not dict or state.canonical_json_bytes(value) != payload:
        raise ControllerPendingV2(
            "UPSTREAM_ADMISSION",
            "UPSTREAM_ADMISSION_NOT_CANONICAL_JSON",
            label,
        )
    return value


def _identity(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value or "\x00" in value:
        raise ControllerPendingV2(
            "UPSTREAM_ADMISSION",
            "UPSTREAM_ADMISSION_FIELD_DIFFERS",
            label,
        )
    return value


def _tagged_digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ControllerPendingV2(
            "UPSTREAM_ADMISSION",
            "UPSTREAM_ADMISSION_FIELD_DIFFERS",
            label,
        )
    return value


def _load_contract_pair(contract_path: Path) -> tuple[Any, Any]:
    try:
        contract = state.load_contract(contract_path)
        preregistration = state.load_preregistration(
            Path(contract.value["preregistration_path"])
        )
    except (KeyError, OSError, ValueError) as error:
        raise ControllerPendingV2(
            "LOCAL_CONTRACT",
            "CONTRACT_OR_PREREGISTRATION_UNAVAILABLE",
            str(error),
        ) from error
    if contract.value.get("preregistration_raw_sha256") != preregistration.raw_sha256:
        raise ControllerPendingV2(
            "LOCAL_CONTRACT",
            "CONTRACT_PREREGISTRATION_BINDING_DIFFERS",
            str(contract_path),
        )
    return contract, preregistration


def _contract_path(contract: Any, key: str) -> Path:
    try:
        value = contract.value["fixed_paths"][key]
    except (KeyError, TypeError) as error:
        raise ControllerPendingV2(
            "LOCAL_CONTRACT",
            "FIXED_PATH_UNAVAILABLE",
            key,
        ) from error
    if type(value) is not str:
        raise ControllerPendingV2(
            "LOCAL_CONTRACT",
            "FIXED_PATH_UNAVAILABLE",
            key,
        )
    path = Path(value)
    if not path.is_absolute():
        raise ControllerPendingV2(
            "LOCAL_CONTRACT",
            "FIXED_PATH_NOT_ABSOLUTE",
            key,
        )
    return path


def _dependency_rows(contract: Any, preregistration: Any) -> list[dict[str, Any]]:
    project_root = _contract_path(contract, "project_root")
    supplied = preregistration.value.get("dependency_bindings_in_exact_order")
    if type(supplied) is not list:
        raise ControllerPendingV2(
            "STATIC_PREFLIGHT",
            "DEPENDENCY_BINDINGS_DIFFER",
            "dependency list is absent",
        )
    expected_names = tuple(core.REQUIRED_STATIC_BINDING_NAMES_V2)
    if tuple(row.get("name") if type(row) is dict else None for row in supplied) != expected_names:
        raise ControllerPendingV2(
            "STATIC_PREFLIGHT",
            "DEPENDENCY_BINDINGS_DIFFER",
            "dependency names or order differ",
        )
    result: list[dict[str, Any]] = []
    for row in supplied:
        if type(row) is not dict or set(row) != {"name", "relative_path", "raw_sha256"}:
            raise ControllerPendingV2(
                "STATIC_PREFLIGHT",
                "DEPENDENCY_BINDINGS_DIFFER",
                "dependency row shape differs",
            )
        relative = row["relative_path"]
        if (
            type(relative) is not str
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ControllerPendingV2(
                "STATIC_PREFLIGHT",
                "DEPENDENCY_PATH_DIFFERS",
                str(relative),
            )
        payload = _read_regular_once(
            project_root / relative,
            f"public dependency {relative}",
            state.MAXIMUM_JSON_BYTES_V2,
        )
        observed = state.raw_sha256(payload)
        if observed != row["raw_sha256"]:
            raise ControllerPendingV2(
                "STATIC_PREFLIGHT",
                "DEPENDENCY_RAW_IDENTITY_DIFFERS",
                relative,
            )
        result.append(
            {
                "name": row["name"],
                "relative_path": relative,
                "byte_count": len(payload),
                "raw_sha256": observed,
            }
        )
    return result


def _source_closure_rows(contract: Any) -> list[dict[str, Any]]:
    project_root = _contract_path(contract, "project_root")
    rows: list[dict[str, Any]] = []
    for relative in state.SOURCE_PATHS_V2:
        payload = _read_regular_once(
            project_root / relative,
            f"source closure {relative}",
            state.MAXIMUM_JSON_BYTES_V2,
        )
        rows.append(
            {
                "relative_path": relative,
                "byte_count": len(payload),
                "raw_sha256": state.raw_sha256(payload),
            }
        )
    return state.validate_source_closure_rows(rows)


def _static_preflight(contract: Any, preregistration: Any) -> dict[str, Any]:
    dependencies = _dependency_rows(contract, preregistration)
    source_rows = _source_closure_rows(contract)
    receipt = {
        "schema_version": (
            "hegel-phase2b-c1-actual-unsealed-960-case-prediction-scoring-"
            "replay-static-dependency-receipt/2"
        ),
        "dependency_rows": dependencies,
        "actual_input_stat_or_read_count": 0,
        "logical_campaign_count": 0,
        "authorization_consumption_count": 0,
        "campaign_attempt_id_allocation_count": 0,
        "cohort_generation_or_read_count": 0,
        "recognizer_or_baseline_producer_invocation_count": 0,
        "scorer_decode_or_evaluate_count": 0,
        "subprocess_docker_network_count": 0,
        "q6_access_count": 0,
    }
    return {
        "dependency_rows": dependencies,
        "dependency_receipt_root": state.raw_sha256(
            state.canonical_json_bytes(receipt)
        ),
        "source_closure_rows": source_rows,
        "source_closure_root": state.source_closure_root(source_rows),
        "static_binding_rows": [
            {"name": row["name"], "identity": row["raw_sha256"]}
            for row in dependencies
        ],
        "action_audit": {
            key: value
            for key, value in receipt.items()
            if key.endswith("_count")
        },
    }


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_new_or_exact(path: Path, payload: bytes, label: str) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            existing = _read_regular_once(path, label, max(len(payload), 1))
            if existing != payload:
                raise OSError("existing bytes differ")
            return False
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short durable write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_parent(path)
        return True
    except (OSError, ControllerPendingV2) as error:
        raise ControllerPendingV2(
            "LOCAL_PUBLICATION",
            "DURABLE_PUBLICATION_UNAVAILABLE",
            f"{label}: {error}",
        ) from error


def _replace_pending(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".replacement")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short pending write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
        _fsync_parent(path)
    except OSError as error:
        raise ControllerPendingV2(
            "LOCAL_PUBLICATION",
            "PENDING_PUBLICATION_UNAVAILABLE",
            str(error),
        ) from error


def _append_pending_history(contract: Any, pending: Mapping[str, Any]) -> None:
    path = _contract_path(contract, "pending_history_path")
    payload = state.canonical_json_bytes(dict(pending))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short pending-history append")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_parent(path)
    except OSError as error:
        raise ControllerPendingV2(
            "LOCAL_PUBLICATION",
            "PENDING_HISTORY_PUBLICATION_UNAVAILABLE",
            str(error),
        ) from error


def _append_trajectory(
    contract: Any,
    event: str,
    *,
    capsule_digest: str,
    phase: str | None = None,
    code: str | None = None,
    detail: str | None = None,
    **fields: object,
) -> None:
    row = {
        "schema_version": (
            "hegel-phase2b-c1-actual-unsealed-960-case-prediction-"
            "scoring-replay-trajectory/2"
        ),
        "event": event,
        "observed_time_ns": time.time_ns(),
        "capsule_digest": capsule_digest,
        **fields,
    }
    if phase is not None:
        row["phase"] = phase
    if code is not None:
        row["code"] = code
    if detail is not None:
        row["detail"] = detail
    path = _contract_path(contract, "trajectory_path")
    payload = state.canonical_json_bytes(row)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short trajectory append")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_parent(path)
    except OSError as error:
        raise ControllerPendingV2(
            "TRAJECTORY_PUBLICATION",
            "TRAJECTORY_APPEND_UNAVAILABLE",
            f"{event}: {error}",
        ) from error


@contextmanager
def _campaign_lock(contract: Any) -> Iterator[None]:
    path = _contract_path(contract, "campaign_lock_path")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _load_json_record(path: Path, label: str) -> dict[str, Any]:
    try:
        return state.load_pretty_json(path, label).value
    except (OSError, ValueError) as error:
        raise ControllerPendingV2(
            "LOCAL_STATE",
            "STATE_RECORD_UNAVAILABLE",
            f"{label}: {error}",
        ) from error


def _optional_regular(path: Path, maximum_bytes: int) -> bytes | None:
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ControllerPendingV2(
            "LOCAL_STATE",
            "STATE_PATH_UNAVAILABLE",
            str(error),
        ) from error
    return _read_regular_once(path, str(path), maximum_bytes)


def _regular_file_available_without_read(
    path: Path, label: str, maximum_bytes: int
) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ControllerPendingV2(
            "PRECONSUMPTION_INPUT",
            "BOUND_LOCAL_FILE_STATUS_UNAVAILABLE",
            f"{label}: {error}",
        ) from error
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or not 1 <= info.st_size <= maximum_bytes
    ):
        raise ControllerPendingV2(
            "PRECONSUMPTION_INPUT",
            "BOUND_LOCAL_FILE_METADATA_DIFFERS",
            label,
        )
    return True


def _load_capsule(
    contract: Any,
    preregistration: Any,
    capsule_digest: str,
) -> dict[str, Any]:
    value = _load_json_record(_contract_path(contract, "capsule_path"), "capsule")
    checked = state.validate_capsule(value, contract, preregistration)
    if checked.get("capsule_digest") != capsule_digest:
        raise ControllerPendingV2(
            "CAPSULE_REPLAY",
            "CAPSULE_DIGEST_DIFFERS",
            capsule_digest,
        )
    return checked


def _replay_static_closure(
    contract: Any,
    preregistration: Any,
    capsule: Mapping[str, Any],
) -> None:
    preflight = _static_preflight(contract, preregistration)
    if (
        preflight["source_closure_rows"] != capsule.get("source_closure_rows")
        or preflight["source_closure_root"] != capsule.get("source_closure_root")
        or preflight["static_binding_rows"] != capsule.get("static_binding_rows")
        or preflight["dependency_receipt_root"]
        != capsule.get("dependency_receipt_root")
    ):
        raise ControllerPendingV2(
            "CAPSULE_REPLAY",
            "CAPSULE_STATIC_CLOSURE_DIFFERS",
            str(contract.path),
        )


def preflight(contract_path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    """Replay public engineering inputs without touching future actual paths."""

    contract, preregistration = _load_contract_pair(contract_path)
    checked = _static_preflight(contract, preregistration)
    return {
        "status": core.READY_STATUS_V2,
        "source_closure_root": checked["source_closure_root"],
        "dependency_receipt_root": checked["dependency_receipt_root"],
        "dependency_binding_count": len(checked["dependency_rows"]),
        "future_actual_input_stat_or_read_count": 0,
        "action_audit": checked["action_audit"],
    }


def prepare_capsule(
    contract_path: Path,
    *,
    source_commit: str,
    run_id: str | None = None,
    created_at_unix_seconds: int | None = None,
) -> dict[str, Any]:
    """Publish the action-free capsule; do not infer owner authorization."""

    contract, preregistration = _load_contract_pair(contract_path)
    checked = _static_preflight(contract, preregistration)
    capsule = state.build_capsule(
        contract,
        preregistration,
        source_commit=source_commit,
        source_closure_rows=checked["source_closure_rows"],
        static_binding_rows=checked["static_binding_rows"],
        dependency_receipt_root=checked["dependency_receipt_root"],
        run_id=run_id if run_id is not None else secrets.token_hex(16),
        created_at_unix_seconds=(
            created_at_unix_seconds
            if created_at_unix_seconds is not None
            else int(time.time())
        ),
    )
    state.validate_capsule(capsule, contract, preregistration)
    created = _publish_new_or_exact(
        _contract_path(contract, "capsule_path"),
        state.canonical_json_bytes(capsule),
        "capsule",
    )
    if created:
        _append_trajectory(
            contract,
            "CAPSULE_PREPARED",
            capsule_digest=capsule["capsule_digest"],
            source_closure_root=capsule["source_closure_root"],
            logical_campaign_count=0,
            authorization_consumption_count=0,
        )
    return {
        "status": core.READY_STATUS_V2,
        "capsule_digest": capsule["capsule_digest"],
        "source_closure_root": capsule["source_closure_root"],
        "dependency_receipt_root": capsule["dependency_receipt_root"],
        "logical_campaign_count": 0,
        "authorization_consumption_count": 0,
        "future_actual_input_stat_or_read_count": 0,
    }


def record_authorization(contract_path: Path, capsule_digest: str) -> dict[str, Any]:
    contract, preregistration = _load_contract_pair(contract_path)
    with _campaign_lock(contract):
        capsule = _load_capsule(contract, preregistration, capsule_digest)
        _replay_static_closure(contract, preregistration, capsule)
        authorization = state.build_authorization(
            contract, preregistration, capsule
        )
        created = _publish_new_or_exact(
            _contract_path(contract, "authorization_path"),
            state.canonical_json_bytes(authorization),
            "authorization",
        )
        if created:
            _append_trajectory(
                contract,
                "AUTHORIZATION_RECORDED",
                capsule_digest=capsule["capsule_digest"],
                authorization_digest=authorization["authorization_digest"],
                authorization_consumption_count=0,
                logical_campaign_count=0,
            )
        return authorization


def _validate_upstream_manifest(
    payload: bytes,
    capsule_digest: str,
) -> dict[str, Any]:
    value = _strict_canonical_json(payload, "upstream admission manifest")
    if set(value) != _UPSTREAM_FIELDS_V2:
        raise ControllerPendingV2(
            "UPSTREAM_ADMISSION",
            "UPSTREAM_ADMISSION_FIELDS_DIFFER",
            "field set differs",
        )
    if (
        value.get("schema_version") != UPSTREAM_ADMISSION_SCHEMA_V2
        or value.get("capsule_digest") != capsule_digest
    ):
        raise ControllerPendingV2(
            "UPSTREAM_ADMISSION",
            "UPSTREAM_ADMISSION_IDENTITY_DIFFERS",
            "schema or capsule digest differs",
        )
    for key in (
        "authoritative_r16_bundle_id",
        "campaign_id",
        "attempt_id",
        "recognizer_result_id",
        "prediction_archive_id",
        "answer_manifest_id",
        "scorer_bundle_id",
    ):
        _identity(value.get(key), key)
    _tagged_digest(value.get("scorer_bundle_raw_sha256"), "scorer bundle raw SHA-256")
    baseline_ids = value.get("baseline_result_ids")
    if (
        type(baseline_ids) is not list
        or len(baseline_ids) != 3
        or any(type(item) is not str or not item for item in baseline_ids)
        or len(set(baseline_ids)) != 3
    ):
        raise ControllerPendingV2(
            "UPSTREAM_ADMISSION",
            "BASELINE_RESULT_IDS_DIFFER",
            "three distinct ordered IDs are required",
        )
    for key in (
        "authoritative_schema_verifier_available",
        "authoritative_r01_r16_bundle_verified",
        "external_custodian_checkpoint_verified",
        "three_baseline_outputs_verified",
        "exact_scale_bridge_verified",
        "recognizer_executor_binding_verified",
        "scorer_executor_binding_verified",
    ):
        if type(value.get(key)) is not bool:
            raise ControllerPendingV2(
                "UPSTREAM_ADMISSION",
                "UPSTREAM_ADMISSION_BOOLEAN_DIFFERS",
                key,
            )
    return value


def _admission_from_manifest(
    manifest: Mapping[str, Any] | None,
    *,
    scorer_bundle_available: bool,
) -> Any:
    if manifest is None:
        return core.ActualUnsealed960PreconsumptionAdmissionV2(
            static_contract_closure_verified=True,
            authoritative_schema_verifier_available=False,
            authoritative_r01_r16_bundle_verified=False,
            external_custodian_checkpoint_verified=False,
            three_baseline_outputs_verified=False,
            exact_scale_bridge_verified=False,
            recognizer_executor_binding_verified=True,
            scorer_executor_binding_verified=scorer_bundle_available,
        )
    return core.ActualUnsealed960PreconsumptionAdmissionV2(
        static_contract_closure_verified=True,
        authoritative_schema_verifier_available=manifest[
            "authoritative_schema_verifier_available"
        ],
        authoritative_r01_r16_bundle_verified=manifest[
            "authoritative_r01_r16_bundle_verified"
        ],
        external_custodian_checkpoint_verified=manifest[
            "external_custodian_checkpoint_verified"
        ],
        three_baseline_outputs_verified=manifest["three_baseline_outputs_verified"],
        exact_scale_bridge_verified=manifest["exact_scale_bridge_verified"],
        recognizer_executor_binding_verified=manifest[
            "recognizer_executor_binding_verified"
        ],
        scorer_executor_binding_verified=(
            manifest["scorer_executor_binding_verified"]
            and scorer_bundle_available
        ),
    )


def _publish_preconsumption_pending(
    contract: Any,
    preregistration: Any,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    admission: Any,
    *,
    phase: str,
    condition: str,
    detail: str,
) -> dict[str, Any]:
    pending = state.build_preconsumption_pending(
        contract,
        preregistration,
        capsule,
        authorization,
        admission,
    )
    _replace_pending(
        _contract_path(contract, "pending_path"),
        state.canonical_json_bytes(pending),
    )
    _append_pending_history(contract, pending)
    _append_trajectory(
        contract,
        "PRECONSUMPTION_PENDING",
        capsule_digest=capsule["capsule_digest"],
        phase=phase,
        code=condition,
        detail=detail,
        pending_digest=pending["pending_digest"],
        authorization_consumption_count=0,
        logical_campaign_count=0,
    )
    return {
        **pending,
        "controller_phase": phase,
        "controller_condition": condition,
        "controller_detail": detail,
    }


def _clear_pending(contract: Any, capsule: Mapping[str, Any]) -> None:
    path = _contract_path(contract, "pending_path")
    payload = _optional_regular(path, state.MAXIMUM_JSON_BYTES_V2)
    if payload is None:
        return
    value = _strict_canonical_json(payload, "existing pre-consumption PENDING")
    if (
        value.get("status") != state.PRECONSUMPTION_PENDING_V2
        or value.get("capsule_digest") != capsule.get("capsule_digest")
    ):
        _terminal("PENDING_STATE_DIFFERS", "existing pending record is not clearable")
    try:
        path.unlink()
        _fsync_parent(path)
    except OSError as error:
        raise ControllerPendingV2(
            "PRECONSUMPTION_PENDING_CLEAR",
            "PENDING_CLEAR_UNAVAILABLE",
            str(error),
        ) from error


def _full_admission() -> Any:
    return core.ActualUnsealed960PreconsumptionAdmissionV2(
        static_contract_closure_verified=True,
        authoritative_schema_verifier_available=True,
        authoritative_r01_r16_bundle_verified=True,
        external_custodian_checkpoint_verified=True,
        three_baseline_outputs_verified=True,
        exact_scale_bridge_verified=True,
        recognizer_executor_binding_verified=True,
        scorer_executor_binding_verified=True,
    )


def _snapshot_through_completion(capsule: Mapping[str, Any]) -> Any:
    snapshot = core.initial_actual_unsealed_960_replay_snapshot_v2(
        state.static_bindings_from_capsule(capsule)
    )
    snapshot = core.reduce_actual_unsealed_960_prediction_scoring_replay_v2(
        snapshot, core.ActualUnsealed960ReplayEventV2.RECORD_AUTHORIZATION
    )
    snapshot = core.reduce_actual_unsealed_960_prediction_scoring_replay_v2(
        snapshot,
        core.ActualUnsealed960ReplayEventV2.REPLAY_PRECONSUMPTION_ADMISSION,
        admission=_full_admission(),
    )
    for event in (
        core.ActualUnsealed960ReplayEventV2.CONSUME_AUTHORIZATION,
        core.ActualUnsealed960ReplayEventV2.SEAL_RECOGNIZER_RESULT,
        core.ActualUnsealed960ReplayEventV2.ADMIT_BASELINE_RESULTS,
        core.ActualUnsealed960ReplayEventV2.ADMIT_PREDICTION_ARCHIVE,
        core.ActualUnsealed960ReplayEventV2.ADMIT_ANSWER_MANIFEST,
        core.ActualUnsealed960ReplayEventV2.SEAL_SCORER_REPLAY,
        core.ActualUnsealed960ReplayEventV2.COMPLETE_MECHANICS,
    ):
        snapshot = core.reduce_actual_unsealed_960_prediction_scoring_replay_v2(
            snapshot, event
        )
    return snapshot


def _decode_and_evaluate_once(payload: bytes) -> tuple[Any, Any]:
    scorer = importlib.import_module("hegel_machine.phase2b_scorer_bundle_v1")
    decoded = scorer.decode_phase2b_scorer_bundle_v1(payload)
    evaluated = scorer.evaluate_phase2b_scorer_bundle_v1(decoded)
    return decoded, evaluated


def _invalidate_after_consumption(
    contract: Any,
    preregistration: Any,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    *,
    failure_phase: str,
    failure_code: str,
    failure_detail: str,
) -> dict[str, Any]:
    terminal = state.build_invalidated_terminal(
        contract,
        preregistration,
        capsule,
        authorization,
        consumption,
        _full_admission(),
        failure_phase=failure_phase,
        failure_code=failure_code,
        failure_detail=failure_detail,
    )
    created = _publish_new_or_exact(
        _contract_path(contract, "terminal_path"),
        state.canonical_json_bytes(terminal),
        "invalidated terminal",
    )
    if created:
        _append_trajectory(
            contract,
            "INVALIDATED",
            capsule_digest=capsule["capsule_digest"],
            phase=failure_phase,
            code=failure_code,
            detail=failure_detail,
            terminal_record_digest=terminal["record_digest"],
        )
    return terminal


def _read_authorization(
    contract: Any,
    preregistration: Any,
    capsule: Mapping[str, Any],
) -> dict[str, Any]:
    path = _contract_path(contract, "authorization_path")
    try:
        value = _load_json_record(path, "authorization")
    except ControllerPendingV2 as error:
        raise ControllerTerminalV2("NOT_AUTHORIZED", str(error)) from error
    return state.validate_authorization(value, contract, preregistration, capsule)


def _existing_terminal_result(
    contract: Any,
    preregistration: Any,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    terminal_payload: bytes,
    artifact_payload: bytes | None,
) -> dict[str, Any]:
    terminal = _strict_canonical_json(terminal_payload, "terminal")
    return state.validate_terminal(
        terminal,
        contract,
        preregistration,
        capsule,
        authorization,
        consumption,
        _full_admission(),
        artifact_payload=artifact_payload,
    )


def _run_locked(
    contract: Any,
    preregistration: Any,
    capsule_digest: str,
) -> dict[str, Any]:
    capsule = _load_capsule(contract, preregistration, capsule_digest)
    _replay_static_closure(contract, preregistration, capsule)
    authorization = _read_authorization(contract, preregistration, capsule)

    artifact_path = _contract_path(contract, "artifact_path")
    consumption_path = _contract_path(contract, "consumption_path")
    terminal_path = _contract_path(contract, "terminal_path")
    artifact_payload = _optional_regular(
        artifact_path, state.MAXIMUM_SEALED_ARTIFACT_BYTES_V2
    )
    consumption_payload = _optional_regular(
        consumption_path, state.MAXIMUM_JSON_BYTES_V2
    )
    terminal_payload = _optional_regular(terminal_path, state.MAXIMUM_JSON_BYTES_V2)

    if terminal_payload is not None:
        if consumption_payload is None:
            _terminal("TERMINAL_STATE_DIFFERS", "terminal exists without consumption")
        consumption = _strict_canonical_json(consumption_payload, "consumption")
        return _existing_terminal_result(
            contract,
            preregistration,
            capsule,
            authorization,
            consumption,
            terminal_payload,
            artifact_payload,
        )

    if consumption_payload is not None:
        consumption = _strict_canonical_json(consumption_payload, "consumption")
        state.validate_consumption(
            consumption,
            contract,
            preregistration,
            capsule,
            authorization,
            _full_admission(),
        )
        if artifact_payload is None:
            return _invalidate_after_consumption(
                contract,
                preregistration,
                capsule,
                authorization,
                consumption,
                failure_phase="PARTIAL_FINALIZATION",
                failure_code="CONSUMED_WITHOUT_SEALED_ARTIFACT",
                failure_detail="evaluator re-entry is forbidden",
            )
        terminal = state.build_complete_terminal(
            contract,
            preregistration,
            capsule,
            authorization,
            consumption,
            _full_admission(),
            artifact_payload,
        )
        created = _publish_new_or_exact(
            terminal_path,
            state.canonical_json_bytes(terminal),
            "complete terminal",
        )
        if created:
            _append_trajectory(
                contract,
                "TERMINAL_PUBLISHED",
                capsule_digest=capsule["capsule_digest"],
                terminal_record_digest=terminal["record_digest"],
                partial_finalization=True,
            )
        return terminal

    if artifact_payload is not None:
        _terminal("ARTIFACT_STATE_DIFFERS", "artifact exists before consumption")

    manifest_path = _contract_path(contract, "upstream_admission_manifest_path")
    scorer_bundle_path = _contract_path(contract, "scorer_bundle_path")
    manifest_payload = _optional_regular(
        manifest_path, MAXIMUM_UPSTREAM_ADMISSION_BYTES_V2
    )
    if manifest_payload is None:
        admission = _admission_from_manifest(None, scorer_bundle_available=False)
        return _publish_preconsumption_pending(
            contract,
            preregistration,
            capsule,
            authorization,
            admission,
            phase="UPSTREAM_ADMISSION",
            condition="CANONICAL_UPSTREAM_ADMISSION_MANIFEST_UNAVAILABLE",
            detail=str(manifest_path),
        )
    try:
        manifest = _validate_upstream_manifest(manifest_payload, capsule_digest)
    except ControllerPendingV2 as error:
        admission = _admission_from_manifest(None, scorer_bundle_available=False)
        return _publish_preconsumption_pending(
            contract,
            preregistration,
            capsule,
            authorization,
            admission,
            phase=error.phase,
            condition=error.condition,
            detail=error.detail,
        )

    scorer_bundle_available = _regular_file_available_without_read(
        scorer_bundle_path,
        "bound scorer bundle",
        MAXIMUM_SCORER_BUNDLE_BYTES_V2,
    )
    if not scorer_bundle_available:
        admission = _admission_from_manifest(manifest, scorer_bundle_available=False)
        return _publish_preconsumption_pending(
            contract,
            preregistration,
            capsule,
            authorization,
            admission,
            phase="SCORER_BUNDLE_ADMISSION",
            condition="BOUND_SCORER_BUNDLE_UNAVAILABLE",
            detail=str(scorer_bundle_path),
        )
    admission = _admission_from_manifest(manifest, scorer_bundle_available=True)
    if not admission.admitted:
        return _publish_preconsumption_pending(
            contract,
            preregistration,
            capsule,
            authorization,
            admission,
            phase="UPSTREAM_ADMISSION",
            condition="UPSTREAM_ADMISSION_NOT_FULLY_TRUE",
            detail=",".join(admission.pending_reasons),
        )

    _clear_pending(contract, capsule)
    consumption = state.build_consumption(
        contract,
        preregistration,
        capsule,
        authorization,
        admission,
    )
    consumed_now = _publish_new_or_exact(
        consumption_path,
        state.canonical_json_bytes(consumption),
        "consumption",
    )
    if not consumed_now:
        _terminal(
            "CONSUMPTION_STATE_DIFFERS",
            "consumption appeared during the locked one-shot transition",
        )

    try:
        _append_trajectory(
            contract,
            "AUTHORIZATION_CONSUMED",
            capsule_digest=capsule["capsule_digest"],
            authorization_digest=authorization["authorization_digest"],
            consumption_record_digest=consumption["record_digest"],
            authorization_consumption_count=1,
            logical_campaign_count=1,
        )
        scorer_payload = _read_regular_once(
            scorer_bundle_path,
            "bound scorer bundle",
            MAXIMUM_SCORER_BUNDLE_BYTES_V2,
        )
        observed_scorer_raw_sha256 = state.raw_sha256(scorer_payload)
        _append_trajectory(
            contract,
            "SCORER_BUNDLE_READ",
            capsule_digest=capsule["capsule_digest"],
            scorer_bundle_raw_sha256=observed_scorer_raw_sha256,
            scorer_bundle_read_count=1,
        )
        if observed_scorer_raw_sha256 != manifest["scorer_bundle_raw_sha256"]:
            raise ValueError("scorer bundle raw SHA-256 differs from admission")
        decoded, evaluated = _decode_and_evaluate_once(scorer_payload)
        if (
            decoded.bundle_id != manifest["scorer_bundle_id"]
            or decoded.campaign_id != manifest["campaign_id"]
            or decoded.attempt_id != manifest["attempt_id"]
            or decoded.structural_receipt.receipt_id
            != manifest["recognizer_result_id"]
            or decoded.decoded_prediction_archive.archive_id
            != manifest["prediction_archive_id"]
            or decoded.answer_manifest.answer_manifest_id
            != manifest["answer_manifest_id"]
        ):
            raise ValueError("decoded scorer bundle identities differ from admission")
        scoring = evaluated.scoring_mechanics
        gates = evaluated.available_gate_mechanics
        observed_counts = (
            scoring.total_prediction_count,
            scoring.main_row_result_count,
            scoring.semantic_conflict_excluded_count,
            len(scoring.metric_results),
            gates.overall_result_count,
            gates.slice_result_count,
            gates.unavailable_result_count,
        )
        if observed_counts != (960, 720, 240, 9, 10, 24, 2):
            raise ValueError(f"bounded scorer counts differ: {observed_counts!r}")
        if (
            len(gates.available_overall_gate_mechanics_results) != 10
            or len(gates.available_slice_gate_mechanics_results) != 24
            or len(gates.unavailable_gate_mechanics) != 2
        ):
            raise ValueError("bounded gate tuple counts differ")
        _append_trajectory(
            contract,
            "SCORER_REPLAY_COMPLETED",
            capsule_digest=capsule["capsule_digest"],
            score_report_id=evaluated.score_report_id,
            available_gate_report_id=evaluated.available_gate_report_id,
            total_predictions=960,
            main_scored_rows=720,
            semantic_conflict_excluded=240,
            metric_results=9,
            available_overall_gate_mechanics=10,
            available_slice_gate_mechanics=24,
            unavailable_formal_overall_gates=2,
        )
        evidence = core.ActualUnsealed960ReplaySealedEvidenceV2(
            authorization_digest=authorization["authorization_digest"],
            consumption_record_digest=consumption["record_digest"],
            authoritative_r16_bundle_id=manifest["authoritative_r16_bundle_id"],
            campaign_id=manifest["campaign_id"],
            attempt_id=manifest["attempt_id"],
            recognizer_result_id=manifest["recognizer_result_id"],
            baseline_result_ids=tuple(manifest["baseline_result_ids"]),
            prediction_archive_id=manifest["prediction_archive_id"],
            answer_manifest_id=manifest["answer_manifest_id"],
            scorer_bundle_id=manifest["scorer_bundle_id"],
            score_report_id=evaluated.score_report_id,
            available_gate_report_id=evaluated.available_gate_report_id,
            upstream_admission_manifest_raw_sha256=state.raw_sha256(
                manifest_payload
            ),
            scorer_bundle_raw_sha256=observed_scorer_raw_sha256,
            total_predictions=960,
            main_scored_rows=720,
            semantic_conflict_excluded=240,
            metric_results=9,
            available_overall_gate_mechanics=10,
            available_slice_gate_mechanics=24,
            unavailable_formal_overall_gates=2,
        )
        snapshot = _snapshot_through_completion(capsule)
        artifact = state.build_sealed_completion_artifact_v2(snapshot, evidence)
        artifact_created = _publish_new_or_exact(
            artifact_path,
            artifact.canonical_bytes,
            "sealed completion artifact",
        )
        if not artifact_created:
            raise ValueError("artifact appeared during one-shot replay")
        _append_trajectory(
            contract,
            "ARTIFACT_PUBLISHED",
            capsule_digest=capsule["capsule_digest"],
            artifact_raw_sha256=state.raw_sha256(artifact.canonical_bytes),
            artifact_root=artifact.artifact_root_id,
        )
    except Exception as error:
        return _invalidate_after_consumption(
            contract,
            preregistration,
            capsule,
            authorization,
            consumption,
            failure_phase="SCORER_REPLAY",
            failure_code="POST_CONSUMPTION_REPLAY_FAILED",
            failure_detail=f"{type(error).__name__}: {error}",
        )

    terminal = state.build_complete_terminal(
        contract,
        preregistration,
        capsule,
        authorization,
        consumption,
        admission,
        artifact.canonical_bytes,
    )
    terminal_created = _publish_new_or_exact(
        terminal_path,
        state.canonical_json_bytes(terminal),
        "complete terminal",
    )
    if terminal_created:
        _append_trajectory(
            contract,
            "TERMINAL_PUBLISHED",
            capsule_digest=capsule["capsule_digest"],
            terminal_record_digest=terminal["record_digest"],
            partial_finalization=False,
        )
    return terminal


def run_v2(contract_path: Path, capsule_digest: str) -> dict[str, Any]:
    contract, preregistration = _load_contract_pair(contract_path)
    with _campaign_lock(contract):
        return _run_locked(contract, preregistration, capsule_digest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    prepare = subparsers.add_parser("prepare-capsule")
    prepare.add_argument("--source-commit", required=True)
    prepare.add_argument("--run-id")
    prepare.add_argument("--created-at-unix-seconds", type=int)
    authorize = subparsers.add_parser("record-authorization")
    authorize.add_argument("--capsule-digest", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--capsule-digest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight(args.contract)
        elif args.command == "prepare-capsule":
            result = prepare_capsule(
                args.contract,
                source_commit=args.source_commit,
                run_id=args.run_id,
                created_at_unix_seconds=args.created_at_unix_seconds,
            )
        elif args.command == "record-authorization":
            result = record_authorization(args.contract, args.capsule_digest)
        else:
            result = run_v2(args.contract, args.capsule_digest)
    except ControllerPendingV2 as error:
        result = {
            "status": state.PRECONSUMPTION_PENDING_V2,
            "phase": error.phase,
            "condition": error.condition,
            "detail": error.detail,
            "authorization_consumption_count": 0,
            "logical_campaign_count": 0,
        }
        print(state.canonical_json_bytes(result).decode("ascii"), end="")
        return PENDING_EXIT_CODE
    except ControllerTerminalV2 as error:
        result = {"status": error.code, "detail": error.detail}
        print(state.canonical_json_bytes(result).decode("ascii"), end="")
        return NOT_AUTHORIZED_EXIT_CODE if error.code == "NOT_AUTHORIZED" else TERMINAL_EXIT_CODE
    print(state.canonical_json_bytes(result).decode("ascii"), end="")
    if result.get("status") == state.PRECONSUMPTION_PENDING_V2:
        return PENDING_EXIT_CODE
    if result.get("status") == state.INVALIDATED_TERMINAL_NO_RESUME_V2:
        return TERMINAL_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
