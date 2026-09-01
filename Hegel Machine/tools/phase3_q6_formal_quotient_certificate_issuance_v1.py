#!/usr/bin/env python3
"""Exact controller for Phase-3A Q6 Purpose-5 3-of-3 issuance.

The controller treats each signer and each producer-free verification as a
one-shot action.  A durable verification intent is published before the sole
verification call; an intent without its verified checkpoint is therefore a
terminal unknown, never a reason to verify twice.  The controller only stats
private-key custody metadata and never opens private-key contents.
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

state = importlib.import_module("hegel_machine.phase3_q6_completion_replay_v1")
wire = importlib.import_module(
    "hegel_machine.phase3_q6_formal_quotient_certificate_issuance_v1"
)

DEFAULT_CONTRACT: Final = PROJECT_ROOT / (
    "config/phase3_q6_formal_quotient_certificate_issuance_actual_v1.json"
)
PENDING_EXIT_CODE: Final = 75
NOT_AUTHORIZED_EXIT_CODE: Final = 4
TERMINAL_EXIT_CODE: Final = 2
MAX_PRIVATE_KEY_BYTES: Final = 16 * 1024
MAX_PUBLIC_BYTES: Final = 4 * 1024 * 1024
PARTIAL_ARTIFACT_TO_TERMINAL: Final = "ARTIFACT_TO_TERMINAL"
PARTIAL_TERMINAL_TO_TRAJECTORY_SUFFIX: Final = "TERMINAL_TO_TRAJECTORY_SUFFIX"


class ControllerError(RuntimeError):
    pass


class ControllerPending(ControllerError):
    def __init__(self, phase: str, condition: str, detail: str) -> None:
        super().__init__(f"{phase}/{condition}: {detail}")
        self.phase = phase
        self.condition = condition
        self.detail = detail
        self.pending_record: dict[str, Any] | None = None


class ControllerTerminal(ControllerError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class PrefixReplayTerminal(ControllerTerminal):
    def __init__(
        self,
        code: str,
        detail: str,
        receipts: Sequence[Mapping[str, Any]],
        observed_action_audit: Mapping[str, Any],
    ) -> None:
        super().__init__(code, detail)
        self.receipts = [dict(item) for item in receipts]
        self.observed_action_audit = dict(observed_action_audit)


class ControllerPartial(ControllerError):
    def __init__(self, kind: str, detail: str) -> None:
        if kind not in {
            PARTIAL_ARTIFACT_TO_TERMINAL,
            PARTIAL_TERMINAL_TO_TRAJECTORY_SUFFIX,
        }:
            raise ValueError("Q6 partial-finalization kind differs")
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


class ControllerSealedTerminal(ControllerTerminal):
    def __init__(self, terminal_record: Mapping[str, Any]) -> None:
        outcome = terminal_record["outcome"]
        super().__init__(outcome["failure_code"], outcome["failure_detail"])
        self.terminal_record = dict(terminal_record)


def _terminal(code: str, detail: str) -> NoReturn:
    raise ControllerTerminal(code, detail)


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path_present(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ControllerPending(
            "LOCAL_STATE_REPLAY", "Q6_PATH_PRESENCE_UNPROVEN", f"{path}: {error}"
        ) from error
    return True


def _read_regular(path: Path, label: str, maximum_bytes: int) -> bytes:
    try:
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise OSError("not one regular non-symlink file")
        if metadata.st_size < 0 or metadata.st_size > maximum_bytes:
            raise OSError(f"exceeds {maximum_bytes}-byte bound")
        return path.read_bytes()
    except OSError as error:
        raise ControllerPending(
            "LOCAL_READ_ONLY_REPLAY", "Q6_BOUND_LOCAL_FILE_UNAVAILABLE",
            f"{label}: {error}",
        ) from error


def _read_json_if_present(path: Path, label: str) -> dict[str, Any] | None:
    if not _path_present(path):
        return None
    return state.strict_json_bytes(
        _read_regular(path, label, state.MAX_JSON_BYTES), label
    )


def _publish_bytes(
    path: Path, payload: bytes, label: str, *, mode: int = 0o600
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _path_present(path):
        if _read_regular(path, label, max(len(payload), 1)) == payload:
            # An earlier rename may have completed while its parent-directory
            # fsync failed.  Exact bytes are not enough to cross a one-shot
            # boundary; make the directory entry durable before accepting it.
            _fsync_parent(path)
            return
        raise OSError(f"{label} already exists with different bytes")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor = -1
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
    descriptor = -1
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
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
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _publish_json(path: Path, value: Mapping[str, Any], label: str, *, mode: int = 0o600) -> None:
    _publish_bytes(path, state.canonical_json_bytes(dict(value)), label, mode=mode)


def _confirm_exact_publication(
    path: Path, payload: bytes, label: str, *, maximum_bytes: int
) -> bool:
    """Return false only for proven absence; exact bytes require a fresh fsync."""

    if not _path_present(path):
        return False
    if _read_regular(path, label, maximum_bytes) != payload:
        raise ValueError(f"{label} differs after publication error")
    _fsync_parent(path)
    return True


def _confirm_exact_json_publication(
    path: Path, value: Mapping[str, Any], label: str
) -> bool:
    return _confirm_exact_publication(
        path, state.canonical_json_bytes(dict(value)), label,
        maximum_bytes=state.MAX_JSON_BYTES,
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
            "LOCAL_CONTRACT", "Q6_CONTRACT_OR_PREREGISTRATION_UNAVAILABLE", str(error)
        ) from error
    if contract.value.get("preregistration_raw_sha256") != preregistration.raw_digest:
        raise ControllerPending(
            "LOCAL_CONTRACT", "Q6_PREREGISTRATION_RAW_BINDING_DIFFERS",
            preregistration.raw_digest,
        )
    fixed = contract.value["fixed_paths"]
    required = {
        "project_root", "campaign_lock_path", "run_root", "ledger_root",
        "capsule_path", "authorization_path", "consumption_path", "pending_path",
        "pending_history_path", "checkpoint_path", "terminal_path", "signature_preimage_path",
        "signing_message_path", "envelope_path", "artifact_path", "trajectory_path",
    }
    for ordinal in state.SIGNER_ORDINALS:
        required.update({
            f"signer_{ordinal}_intent_path",
            f"signer_{ordinal}_verification_intent_path",
            f"signer_{ordinal}_signature_path",
            f"signer_{ordinal}_signature_temp_path",
            f"signer_{ordinal}_worker_receipt_path",
            f"signer_{ordinal}_verified_checkpoint_path",
            f"signer_{ordinal}_custody_root",
        })
    if not required.issubset(fixed):
        raise ControllerPending(
            "LOCAL_CONTRACT", "Q6_REQUIRED_FIXED_PATHS_MISSING",
            ",".join(sorted(required.difference(fixed))),
        )
    worker = contract.value["worker"]
    expected_worker_fields = {
        "python_executable": "/usr/bin/python3.10",
        "worker_path": str(Path(fixed["project_root"]) / "tools/phase3_q6_purpose5_signing_worker_v1.py"),
        "openssl_executable": "/usr/bin/openssl",
        "expected_uid": 1000,
        "expected_gid": 1000,
        "private_key_name": "ed25519-private.pem",
        "signature_temp_suffix": ".q6-tmp",
        "startup_smoke_receipt_schema": state.WORKER_SMOKE_SCHEMA,
        "startup_smoke_success_status": state.WORKER_SMOKE_STATUS,
        "signing_public_receipt_schema": state.WORKER_RECEIPT_SCHEMA,
        "signing_public_receipt_status": state.WORKER_SUCCESS_STATUS,
        "signature_byte_count": 64,
        "worker_receipt_signature_verification_count": 0,
        "worker_openssl_sign_popen_count_per_signer": 1,
        "worker_private_key_openssl_use_count_per_signer": 1,
        "signer_order": [1, 2, 3],
        "signer_roles": list(state.SIGNER_ROLE_NAMES),
    }
    if worker != expected_worker_fields:
        raise ControllerPending(
            "LOCAL_CONTRACT", "Q6_WORKER_CONFIGURATION_DIFFERS", str(path)
        )
    return contract, preregistration


def _source_closure_rows(contract: Any) -> list[dict[str, Any]]:
    project_root = Path(contract.value["fixed_paths"]["project_root"])
    rows: list[dict[str, Any]] = []
    for relative in state.Q6_SOURCE_PATHS:
        path = project_root / relative
        payload = _read_regular(path, f"Q6 source {relative}", state.MAX_JSON_BYTES)
        try:
            if path.resolve(strict=True) != path:
                raise OSError("source path resolves elsewhere")
        except OSError as error:
            raise ControllerPending(
                "SOURCE_CLOSURE", "Q6_SOURCE_PATH_IDENTITY_DIFFERS",
                f"{relative}: {error}",
            ) from error
        rows.append({
            "relative_path": relative,
            "byte_count": len(payload),
            "raw_sha256": state.raw_sha256(payload),
        })
    return state.validate_source_closure_rows(rows)


def _read_predecessor_payloads(contract: Any) -> tuple[dict[str, bytes], bytes]:
    q5b: dict[str, bytes] = {}
    for kind in ("capsule", "authorization", "consumption", "terminal", "artifact", "trajectory"):
        maximum = (
            state.MAX_ARTIFACT_BYTES if kind == "artifact"
            else state.MAX_TRAJECTORY_BYTES if kind == "trajectory"
            else state.MAX_JSON_BYTES
        )
        q5b[kind] = _read_regular(
            Path(contract.value["q5b_inputs"][kind]), f"sealed Q5b {kind}", maximum
        )
    q4 = _read_regular(
        Path(contract.value["q4_input"]["artifact"]),
        "selected Q4 capability artifact", state.MAX_ARTIFACT_BYTES,
    )
    return q5b, q4


def _replay_predecessors(contract: Any, preregistration: Any) -> tuple[Any, dict[str, bytes], bytes]:
    q5b, q4 = _read_predecessor_payloads(contract)
    try:
        replay = state.replay_predecessors_v1(
            preregistration.value, q5b_payloads=q5b, q4_artifact_payload=q4
        )
    except ValueError as error:
        raise ControllerPending(
            "SEALED_PREDECESSOR_REPLAY", "Q6_SEALED_Q5B_OR_Q4_REPLAY_DIFFERS", str(error)
        ) from error
    return replay, q5b, q4


def _worker_command(
    contract: Any,
    capsule: Mapping[str, Any],
    ordinal: int,
    *,
    startup_smoke: bool,
) -> list[str]:
    fixed = contract.value["fixed_paths"]
    worker = contract.value["worker"]
    preflight = capsule["preflight_receipt"]
    manifests = preflight["predecessor_receipt"]
    command = [worker["python_executable"], "-I", "-S", "-B", worker["worker_path"]]
    if startup_smoke:
        command.append("--startup-smoke")
    else:
        command.extend([
            "sign",
            "--custody-root", fixed[f"signer_{ordinal}_custody_root"],
            "--message-path", fixed["signing_message_path"],
            "--signature-path", fixed[f"signer_{ordinal}_signature_path"],
            "--receipt-path", fixed[f"signer_{ordinal}_worker_receipt_path"],
        ])
    command.extend([
        "--signer-ordinal", str(ordinal),
        "--signer-role", state.SIGNER_ROLE_NAMES[ordinal - 1],
        "--expected-uid", str(worker["expected_uid"]),
        "--expected-gid", str(worker["expected_gid"]),
        "--expected-public-key-hex", manifests["role_ordered_public_keys"][ordinal - 1],
        "--expected-key-id-hex", manifests["role_ordered_key_ids"][ordinal - 1],
        "--expected-message-raw-sha256", preflight["signing_message_raw_sha256"],
        "--expected-message-byte-count", str(preflight["signing_message_byte_count"]),
    ])
    return command


def _invoke_worker(command: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(command), check=False, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180,
    )


def _worker_result(command: Sequence[str], label: str, *, post_intent: bool) -> dict[str, Any]:
    try:
        result = _invoke_worker(command)
    except (OSError, subprocess.SubprocessError) as error:
        if post_intent:
            _terminal("Q6_SIGNING_WORKER_OUTCOME_UNKNOWN", f"{label}: {error}")
        raise ControllerPending(label, "Q6_STARTUP_SMOKE_OUTCOME_UNKNOWN", str(error)) from error
    detail = result.stderr.decode("ascii", errors="replace").strip()
    if result.returncode != 0 or result.stderr:
        if post_intent:
            _terminal("Q6_SIGNING_WORKER_FAILED_AFTER_INTENT", f"{label}: {detail}")
        raise ControllerPending(label, "Q6_STARTUP_SMOKE_REJECTED", detail)
    try:
        return state.strict_json_bytes(result.stdout, label)
    except ValueError as error:
        if post_intent:
            _terminal("Q6_SIGNING_WORKER_RESULT_UNKNOWN", f"{label}: {error}")
        raise ControllerPending(label, "Q6_STARTUP_SMOKE_RESULT_DIFFERS", str(error)) from error


def _startup_smokes(contract: Any, capsule_like: Mapping[str, Any]) -> list[dict[str, Any]]:
    preflight = capsule_like["preflight_receipt"]
    predecessor = preflight["predecessor_receipt"]
    receipts: list[dict[str, Any]] = []
    for ordinal in state.SIGNER_ORDINALS:
        value = _worker_result(
            _worker_command(contract, capsule_like, ordinal, startup_smoke=True),
            f"Q6_SIGNER_{ordinal}_STARTUP_SMOKE", post_intent=False,
        )
        try:
            receipts.append(state.validate_worker_smoke_receipt(
                value,
                ordinal=ordinal,
                public_key_hex=predecessor["role_ordered_public_keys"][ordinal - 1],
                key_id_hex=predecessor["role_ordered_key_ids"][ordinal - 1],
                message_raw_sha256=preflight["signing_message_raw_sha256"],
                message_byte_count=preflight["signing_message_byte_count"],
                expected_uid=contract.value["worker"]["expected_uid"],
                expected_gid=contract.value["worker"]["expected_gid"],
            ))
        except ValueError as error:
            raise ControllerPending(
                "WORKER_STARTUP_SMOKE", "Q6_WORKER_STARTUP_SMOKE_DIFFERS", str(error)
            ) from error
    return receipts


def _signer_paths(contract: Any, ordinal: int) -> dict[str, Path]:
    fixed = contract.value["fixed_paths"]
    return {
        "intent": Path(fixed[f"signer_{ordinal}_intent_path"]),
        "verification_intent": Path(fixed[f"signer_{ordinal}_verification_intent_path"]),
        "signature": Path(fixed[f"signer_{ordinal}_signature_path"]),
        "signature_temp": Path(fixed[f"signer_{ordinal}_signature_temp_path"]),
        "worker_receipt": Path(fixed[f"signer_{ordinal}_worker_receipt_path"]),
        "verified_checkpoint": Path(fixed[f"signer_{ordinal}_verified_checkpoint_path"]),
        "custody_root": Path(fixed[f"signer_{ordinal}_custody_root"]),
    }


def _future_paths(contract: Any) -> list[Path]:
    fixed = contract.value["fixed_paths"]
    result = [Path(fixed[key]) for key in (
        "consumption_path", "checkpoint_path", "terminal_path", "envelope_path", "artifact_path"
    )]
    for ordinal in state.SIGNER_ORDINALS:
        paths = _signer_paths(contract, ordinal)
        result.extend(paths[key] for key in (
            "intent", "verification_intent", "signature", "signature_temp",
            "worker_receipt", "verified_checkpoint",
        ))
    return result


def _require_absent(paths: Sequence[Path], phase: str, code: str) -> None:
    for path in paths:
        if _path_present(path):
            raise ControllerPending(phase, code, str(path))


def _preconsumption_replay(
    contract: Any, preregistration: Any, capsule: Mapping[str, Any]
) -> tuple[dict[str, bytes], bytes]:
    q5b, q4 = _replay_capsule_public_gates(
        contract, preregistration, capsule, include_startup_smokes=True
    )
    _require_absent(_future_paths(contract), "FRESH_ROOT_ADMISSION", "Q6_FUTURE_ROOT_ALREADY_EXISTS")
    return q5b, q4


def _replay_capsule_public_gates(
    contract: Any,
    preregistration: Any,
    capsule: Mapping[str, Any],
    *,
    include_startup_smokes: bool,
) -> tuple[dict[str, bytes], bytes]:
    _pending_history_rows(contract)
    rows = _source_closure_rows(contract)
    if rows != capsule["source_closure_rows"] or state.source_closure_root(rows) != capsule["source_closure_root"]:
        raise ControllerPending(
            "SOURCE_CLOSURE", "Q6_CAPSULE_SOURCE_CLOSURE_DIFFERS", capsule["source_closure_root"]
        )
    replay, q5b, q4 = _replay_predecessors(contract, preregistration)
    preimage = wire.build_q6_signature_preimage_v1(
        replay.q4_artifact, replay.q5b_artifact,
        created_at_unix_seconds=capsule["certificate_created_at_unix_seconds"],
    )
    fixed = contract.value["fixed_paths"]
    for path, payload, label, mode in (
        (Path(fixed["signature_preimage_path"]), preimage.canonical_bytes, "Q6 signature preimage", 0o600),
        (Path(fixed["signing_message_path"]), preimage.signature_message, "Q6 signing message", 0o644),
    ):
        if _path_present(path):
            if _read_regular(path, label, state.MAX_ARTIFACT_BYTES) != payload:
                raise ControllerPending(
                    "PREFLIGHT_REPLAY", "Q6_CAPSULE_PREIMAGE_OR_MESSAGE_DIFFERS",
                    str(path),
                )
            continue
        try:
            _publish_bytes(path, payload, label, mode=mode)
        except OSError as error:
            raise ControllerPending(
                "PREFLIGHT_PUBLICATION", "Q6_CAPSULE_PUBLIC_INPUT_NOT_DURABLE",
                f"{label}: {error}",
            ) from error
    if include_startup_smokes:
        smokes = _startup_smokes(contract, capsule)
        expected = state.build_preflight_receipt(replay, smokes, preimage)
        if expected != capsule["preflight_receipt"]:
            raise ControllerPending(
                "PREFLIGHT_REPLAY", "Q6_CAPSULE_PREFLIGHT_RECEIPT_DIFFERS",
                capsule["capsule_digest"],
            )
    else:
        preflight = capsule["preflight_receipt"]
        if (
            preflight["predecessor_receipt"] != replay.preflight_receipt
            or preflight["signature_preimage_raw_sha256"]
            != state.raw_sha256(preimage.canonical_bytes)
            or preflight["signature_preimage_root"]
            != "sha256:" + preimage.preimage_root.hex()
            or preflight["signing_message_raw_sha256"]
            != state.raw_sha256(preimage.signature_message)
            or preflight["signing_message_byte_count"]
            != len(preimage.signature_message)
        ):
            raise ControllerPending(
                "PUBLIC_GATE_REPLAY", "Q6_CAPSULE_PUBLIC_GATE_DIFFERS",
                capsule["capsule_digest"],
            )
    return q5b, q4


def _read_capsule(contract: Any, preregistration: Any, digest: str) -> dict[str, Any]:
    value = _read_json_if_present(Path(contract.value["fixed_paths"]["capsule_path"]), "Q6 capsule")
    if value is None:
        raise ControllerPending("CAPSULE_REPLAY", "Q6_CAPSULE_UNAVAILABLE", digest)
    try:
        checked = state.validate_capsule(value, contract, preregistration.value)
    except ValueError as error:
        raise ControllerPending("CAPSULE_REPLAY", "Q6_CAPSULE_REPLAY_DIFFERS", str(error)) from error
    if checked["capsule_digest"] != digest:
        raise ControllerPending("CAPSULE_REPLAY", "Q6_CAPSULE_DIGEST_DIFFERS", digest)
    return checked


def _read_authorization(contract: Any, preregistration: Any, capsule: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        value = _read_json_if_present(
            Path(contract.value["fixed_paths"]["authorization_path"]),
            "Q6 authorization",
        )
    except (ControllerPending, ValueError) as error:
        try:
            execution_started = _execution_state_exists(contract)
        except ControllerPending:
            execution_started = True
        if execution_started:
            _terminal("Q6_AUTHORIZATION_REPLAY_UNAVAILABLE_AFTER_EXECUTION", str(error))
        raise ControllerPending(
            "AUTHORIZATION_REPLAY", "Q6_AUTHORIZATION_REPLAY_UNAVAILABLE",
            str(error),
        ) from error
    if value is None:
        return None
    try:
        return state.validate_authorization(value, contract, preregistration.value, capsule)
    except ValueError as error:
        _terminal("Q6_AUTHORIZATION_REPLAY_DIFFERS", str(error))


def _read_consumption(
    contract: Any, preregistration: Any, capsule: Mapping[str, Any], authorization: Mapping[str, Any]
) -> dict[str, Any] | None:
    try:
        value = _read_json_if_present(
            Path(contract.value["fixed_paths"]["consumption_path"]),
            "Q6 consumption",
        )
    except (ControllerPending, ValueError) as error:
        _terminal("Q6_CONSUMPTION_REPLAY_UNAVAILABLE", str(error))
    if value is None:
        return None
    try:
        return state.validate_consumption(value, contract, preregistration.value, capsule, authorization)
    except ValueError as error:
        _terminal("Q6_CONSUMPTION_REPLAY_DIFFERS", str(error))


def _trajectory_rows(
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any] | None,
    consumption: Mapping[str, Any] | None,
    receipts: Sequence[Mapping[str, Any]],
    terminal: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    names = ["CAPSULE_PREPARED_NOT_AUTHORIZED"]
    if authorization is not None:
        names.append("AUTHORIZATION_RECORDED_NOT_CONSUMED")
    if consumption is not None:
        names.append("Q6_AUTHORIZATION_CONSUMED")
        names.extend(f"SIGNER_{ordinal}_SIGNATURE_VERIFIED" for ordinal in range(1, len(receipts) + 1))
    if terminal is not None:
        names.append(terminal["status"])
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        prefix_count = 0
        if name.startswith("SIGNER_"):
            prefix_count = int(name.split("_")[1])
        elif terminal is not None and name == terminal["status"]:
            prefix_count = len(receipts)
        is_terminal = terminal is not None and name == terminal["status"]
        outcome = terminal["outcome"] if is_terminal else {}
        rows.append({
            "schema_version": "hegel-phase3a-q6-trajectory/1",
            "event_index": index,
            "event": name,
            "capsule_digest": capsule["capsule_digest"],
            "authorization_digest": None if authorization is None or index == 0 else authorization["authorization_digest"],
            "consumption_record_digest": None if consumption is None or name in {"CAPSULE_PREPARED_NOT_AUTHORIZED", "AUTHORIZATION_RECORDED_NOT_CONSUMED"} else consumption["record_digest"],
            "completed_prefix": list(range(1, prefix_count + 1)),
            "authorization_consumption_count": 0 if consumption is None or name in {"CAPSULE_PREPARED_NOT_AUTHORIZED", "AUTHORIZATION_RECORDED_NOT_CONSUMED"} else 1,
            "logical_campaign_count": 0 if consumption is None or name in {"CAPSULE_PREPARED_NOT_AUTHORIZED", "AUTHORIZATION_RECORDED_NOT_CONSUMED"} else 1,
            "terminal_record_digest": terminal["record_digest"] if is_terminal else None,
            "artifact_raw_sha256": outcome.get("artifact_raw_sha256"),
            "envelope_raw_sha256": outcome.get("envelope_raw_sha256"),
            "complete_checkpoint_digest": outcome.get("complete_checkpoint_digest"),
            "issuance_receipt_root": outcome.get("issuance_receipt_root"),
            "pending_history_receipt": outcome.get("pending_history_receipt"),
            "failure_phase": outcome.get("failure_phase"),
            "failure_code": outcome.get("failure_code"),
            "failure_detail": outcome.get("failure_detail"),
            "q6_action_audit": (
                terminal["outcome"]["q6_action_audit"]
                if is_terminal
                else state.q6_action_audit(tuple(range(1, prefix_count + 1)))
            ),
        })
    return rows


def _reconcile_trajectory(
    contract: Any, capsule: Mapping[str, Any], *,
    authorization: Mapping[str, Any] | None = None,
    consumption: Mapping[str, Any] | None = None,
    receipts: Sequence[Mapping[str, Any]] = (),
    terminal: Mapping[str, Any] | None = None,
) -> None:
    payload = b"".join(state.canonical_json_bytes(row) for row in _trajectory_rows(
        capsule, authorization, consumption, receipts, terminal
    ))
    if len(payload) > state.MAX_TRAJECTORY_BYTES:
        raise ControllerPending("TRAJECTORY", "Q6_TRAJECTORY_BYTE_BOUND_EXCEEDED", str(len(payload)))
    path = Path(contract.value["fixed_paths"]["trajectory_path"])
    existing = b""
    if _path_present(path):
        existing = _read_regular(path, "Q6 trajectory", state.MAX_TRAJECTORY_BYTES)
        try:
            _fsync_parent(path)
        except OSError as error:
            if terminal is not None:
                raise ControllerPartial(
                    PARTIAL_TERMINAL_TO_TRAJECTORY_SUFFIX, str(error)
                ) from error
            raise ControllerPending(
                "TRAJECTORY", "Q6_TRAJECTORY_DURABILITY_UNPROVEN", str(error)
            ) from error
        if not payload.startswith(existing):
            _terminal("Q6_TRAJECTORY_PREFIX_DIFFERS", str(path))
    if existing != payload:
        try:
            _replace_bytes(path, payload, "Q6 trajectory")
        except OSError as error:
            if terminal is not None:
                raise ControllerPartial(PARTIAL_TERMINAL_TO_TRAJECTORY_SUFFIX, str(error)) from error
            raise ControllerPending("TRAJECTORY", "Q6_TRAJECTORY_SUFFIX_NOT_DURABLE", str(error)) from error


def _persist_pending(
    contract: Any, capsule: Mapping[str, Any], error: ControllerPending, *,
    authorization: Mapping[str, Any] | None = None,
    consumption: Mapping[str, Any] | None = None,
    receipts: Sequence[Mapping[str, Any]] = (),
) -> None:
    observed: Mapping[str, Any] | None = None
    inflight: Mapping[str, Any] | None = None
    if consumption is not None:
        fixed = contract.value["fixed_paths"]
        intent_count = sum(
            1 for ordinal in state.SIGNER_ORDINALS
            if _path_present(_signer_paths(contract, ordinal)["intent"])
        )
        worker_count = sum(
            1 for ordinal in state.SIGNER_ORDINALS
            if _path_present(_signer_paths(contract, ordinal)["signature"])
            and _path_present(_signer_paths(contract, ordinal)["worker_receipt"])
        )
        verification_intent_count = sum(
            1 for ordinal in state.SIGNER_ORDINALS
            if _path_present(_signer_paths(contract, ordinal)["verification_intent"])
        )
        prefix_count = len(receipts)
        conservative_worker_count = max(worker_count, verification_intent_count)
        observed = state.q6_observed_action_audit(
            tuple(range(1, prefix_count + 1)),
            metadata_admission_count=intent_count,
            signing_intent_count=intent_count,
            worker_popen_count=conservative_worker_count,
            openssl_sign_count=conservative_worker_count,
            private_key_load_count=conservative_worker_count,
            signature_output_count=conservative_worker_count,
            worker_receipt_count=conservative_worker_count,
            verification_intent_count=verification_intent_count,
            verification_count=verification_intent_count,
            valid_signature_count=prefix_count,
            verified_checkpoint_count=prefix_count,
            verified_checkpoint_publication_count=prefix_count,
            envelope_publication_count=int(_path_present(Path(fixed["envelope_path"]))),
            issuance_count=int(_path_present(Path(fixed["artifact_path"]))),
        )
        if verification_intent_count != prefix_count:
            terminal = _seal_invalidated(
                contract, capsule, authorization, consumption, receipts,
                phase="PENDING_FUNNEL",
                code="Q6_PENDING_FORBIDDEN_AFTER_VERIFICATION_INTENT",
                detail=str(verification_intent_count),
                observed_action_audit=observed,
            )
            raise ControllerSealedTerminal(terminal)
        if worker_count == prefix_count + 1:
            ordinal = prefix_count + 1
            paths = _signer_paths(contract, ordinal)
            intent = _read_json_if_present(
                paths["intent"], f"Q6 signer {ordinal} inflight intent"
            )
            worker_receipt = _read_json_if_present(
                paths["worker_receipt"], f"Q6 signer {ordinal} inflight worker receipt"
            )
            signature = _read_regular(
                paths["signature"], f"Q6 signer {ordinal} inflight signature", 64
            )
            if intent is None or worker_receipt is None:
                _terminal("Q6_PENDING_INFLIGHT_RESULT_REPLAY_DIFFERS", str(ordinal))
            inflight = {
                "status": "KNOWN_SEALED_WORKER_RESULT_AWAITING_VERIFICATION",
                "signer_ordinal": ordinal,
                "intent_digest": intent["intent_digest"],
                "worker_receipt_raw_sha256": state.raw_sha256(
                    state.canonical_json_bytes(worker_receipt)
                ),
                "signature_raw_sha256": state.raw_sha256(signature),
                "verification_intent_digest": None,
            }
        elif worker_count != prefix_count:
            _terminal("Q6_PENDING_WORKER_PREFIX_DIFFERS", str(worker_count))
    pending = state.build_pending(
        contract, capsule, failure_phase=error.phase, failure_code=error.condition,
        failure_detail=error.detail, authorization=authorization,
        consumption=consumption, receipts=receipts,
        observed_action_audit=observed,
        inflight_public_result=inflight,
    )
    error.pending_record = pending
    try:
        _append_pending_history(contract, pending)
    except ControllerPending as history_error:
        history_error.pending_record = pending
        raise
    try:
        _replace_json(Path(contract.value["fixed_paths"]["pending_path"]), pending, "Q6 pending")
    except (OSError, ControllerPending) as snapshot_error:
        error.detail = f"{error.detail}; latest snapshot not durable: {snapshot_error}"
        raise error from snapshot_error


def _pending_history_rows(contract: Any) -> tuple[list[dict[str, Any]], bytes]:
    path = Path(contract.value["fixed_paths"]["pending_history_path"])
    if not _path_present(path):
        return [], b""
    payload = _read_regular(path, "Q6 pending history", state.MAX_TRAJECTORY_BYTES)
    try:
        _fsync_parent(path)
    except OSError as error:
        raise ControllerPending(
            "PENDING_HISTORY", "Q6_PENDING_HISTORY_DURABILITY_UNPROVEN",
            str(error),
        ) from error
    if payload and not payload.endswith(b"\n"):
        _terminal("Q6_PENDING_HISTORY_FRAMING_DIFFERS", str(path))
    rows = [
        state.strict_json_bytes(line, f"Q6 pending history event {index}")
        for index, line in enumerate(payload.splitlines(keepends=True), 1)
    ]
    required = {
        "schema_version", "occurrence", "pending_digest", "failure_phase",
        "failure_code", "failure_detail", "capsule_digest",
        "authorization_digest", "consumption_record_digest", "completed_prefix",
    }
    def tagged_or_none(value: object) -> bool:
        return value is None or (
            type(value) is str
            and len(value) == 71
            and value.startswith("sha256:")
            and all(character in "0123456789abcdef" for character in value[7:])
        )

    def tagged(value: object) -> bool:
        return value is not None and tagged_or_none(value)

    for index, row in enumerate(rows, 1):
        prefix = row.get("completed_prefix")
        authorization_digest = row.get("authorization_digest")
        consumption_digest = row.get("consumption_record_digest")
        if (
            set(row) != required
            or row.get("schema_version") != "hegel-phase3a-q6-pending-history-event/1"
            or row.get("occurrence") != index
            or not tagged(row.get("pending_digest"))
            or not tagged(row.get("capsule_digest"))
            or not tagged_or_none(authorization_digest)
            or not tagged_or_none(consumption_digest)
            or (consumption_digest is not None and authorization_digest is None)
            or type(prefix) is not list
            or tuple(prefix) not in state.PENDING_PREFIXES
            or (consumption_digest is None and prefix != [])
        ):
            _terminal("Q6_PENDING_HISTORY_REPLAY_DIFFERS", str(index))
    return rows, payload


def _append_pending_history(contract: Any, pending: Mapping[str, Any]) -> None:
    rows, payload = _pending_history_rows(contract)
    row = {
        "schema_version": "hegel-phase3a-q6-pending-history-event/1",
        "occurrence": len(rows) + 1,
        "pending_digest": pending["pending_digest"],
        "failure_phase": pending["failure_phase"],
        "failure_code": pending["failure_code"],
        "failure_detail": pending["failure_detail"],
        "capsule_digest": pending["capsule_digest"],
        "authorization_digest": pending["authorization_digest"],
        "consumption_record_digest": pending["consumption_record_digest"],
        "completed_prefix": pending["completed_prefix"],
    }
    path = Path(contract.value["fixed_paths"]["pending_history_path"])
    try:
        _replace_bytes(path, payload + state.canonical_json_bytes(row), "Q6 pending history")
    except OSError as error:
        expected = payload + state.canonical_json_bytes(row)
        try:
            if _confirm_exact_publication(
                path, expected, "Q6 pending history after publication error",
                maximum_bytes=state.MAX_TRAJECTORY_BYTES,
            ):
                return
        except (OSError, ControllerPending, ValueError):
            pass
        raise ControllerPending(
            "PENDING_HISTORY", "Q6_PENDING_HISTORY_NOT_DURABLE", str(error)
        ) from error


def _pending_history_receipt(contract: Any) -> dict[str, Any]:
    rows, payload = _pending_history_rows(contract)
    return {
        "raw_sha256": state.raw_sha256(payload),
        "byte_count": len(payload),
        "event_count": len(rows),
    }


def _clear_pending(contract: Any) -> None:
    path = Path(contract.value["fixed_paths"]["pending_path"])
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as error:
        raise ControllerPending("PENDING_CLEANUP", "Q6_PENDING_CLEANUP_FAILED", str(error)) from error
    try:
        _fsync_parent(path)
    except FileNotFoundError:
        return
    except OSError as error:
        raise ControllerPending(
            "PENDING_CLEANUP", "Q6_PENDING_CLEANUP_DURABILITY_UNPROVEN",
            str(error),
        ) from error


def _clear_pending_after_terminal(contract: Any) -> None:
    try:
        _clear_pending(contract)
    except ControllerPending as error:
        raise ControllerPartial(
            PARTIAL_TERMINAL_TO_TRAJECTORY_SUFFIX, str(error)
        ) from error


def _execution_state_exists(contract: Any) -> bool:
    fixed = contract.value["fixed_paths"]
    if any(_path_present(Path(fixed[key])) for key in ("consumption_path", "checkpoint_path", "terminal_path", "envelope_path", "artifact_path")):
        return True
    return any(any(_path_present(path) for key, path in _signer_paths(contract, ordinal).items() if key != "custody_root") for ordinal in state.SIGNER_ORDINALS)


def prepare_capsule(
    contract_path: Path,
    source_commit: str,
    *,
    run_id: str | None = None,
    created_at_unix_seconds: int | None = None,
    certificate_created_at_unix_seconds: int | None = None,
) -> dict[str, Any]:
    contract, preregistration = _load_contract(contract_path)
    fixed = contract.value["fixed_paths"]
    with _campaign_lock(contract):
        existing = _read_json_if_present(Path(fixed["capsule_path"]), "Q6 capsule")
        if existing is not None:
            capsule = state.validate_capsule(existing, contract, preregistration.value)
            if _execution_state_exists(contract):
                return capsule
            try:
                _preconsumption_replay(contract, preregistration, capsule)
                authorization = _read_authorization(contract, preregistration, capsule)
                _reconcile_trajectory(contract, capsule, authorization=authorization)
                _clear_pending(contract)
            except ControllerPending as error:
                authorization = _read_authorization(contract, preregistration, capsule)
                _persist_pending(contract, capsule, error, authorization=authorization)
                raise
            return capsule

        initial = [Path(fixed[key]) for key in (
            "run_root", "ledger_root", "capsule_path", "authorization_path", "pending_path",
            "pending_history_path", "signature_preimage_path", "signing_message_path", "trajectory_path",
        )] + _future_paths(contract)
        _require_absent(initial, "FRESH_ROOT_ADMISSION", "Q6_INITIAL_ROOT_ALREADY_EXISTS")
        rows = _source_closure_rows(contract)
        replay, _, _ = _replay_predecessors(contract, preregistration)
        certificate_time = int(time.time()) if certificate_created_at_unix_seconds is None else certificate_created_at_unix_seconds
        preimage = wire.build_q6_signature_preimage_v1(
            replay.q4_artifact, replay.q5b_artifact,
            created_at_unix_seconds=certificate_time,
        )
        provisional = {
            "preflight_receipt": {
                "predecessor_receipt": replay.preflight_receipt,
                "signing_message_raw_sha256": state.raw_sha256(preimage.signature_message),
                "signing_message_byte_count": len(preimage.signature_message),
            }
        }
        smokes = _startup_smokes(contract, provisional)
        preflight = state.build_preflight_receipt(replay, smokes, preimage)
        capsule = state.build_capsule(
            contract, preregistration.value, source_commit=source_commit,
            source_closure_rows=rows, run_id=run_id or secrets.token_hex(16),
            created_at_unix_seconds=int(time.time()) if created_at_unix_seconds is None else created_at_unix_seconds,
            preflight_receipt=preflight,
        )
        try:
            _publish_json(Path(fixed["capsule_path"]), capsule, "Q6 capsule")
        except (OSError, ControllerPending) as error:
            try:
                durable = _confirm_exact_json_publication(
                    Path(fixed["capsule_path"]), capsule,
                    "Q6 capsule after publication error",
                )
            except (OSError, ControllerPending, ValueError) as replay_error:
                _terminal(
                    "Q6_CAPSULE_PUBLICATION_OUTCOME_UNKNOWN",
                    f"{error}; durability replay: {replay_error}",
                )
            if not durable:
                raise ControllerPending(
                    "CAPSULE_PUBLICATION", "Q6_CAPSULE_NOT_DURABLE",
                    f"{capsule['capsule_digest']}: {error}",
                ) from error
        try:
            _replay_capsule_public_gates(
                contract, preregistration, capsule, include_startup_smokes=False
            )
        except ControllerPending as error:
            _persist_pending(contract, capsule, error)
            raise
        _reconcile_trajectory(contract, capsule)
        return capsule


def record_authorization(contract_path: Path, capsule_digest: str) -> dict[str, Any]:
    contract, preregistration = _load_contract(contract_path)
    fixed = contract.value["fixed_paths"]
    with _campaign_lock(contract):
        capsule = _read_capsule(contract, preregistration, capsule_digest)
        try:
            existing = _read_authorization(contract, preregistration, capsule)
        except ControllerPending as error:
            _persist_pending(contract, capsule, error)
            raise
        if _execution_state_exists(contract):
            if existing is None:
                _terminal("Q6_AUTHORIZATION_MISSING_AFTER_EXECUTION_STATE", capsule_digest)
            return existing
        try:
            _preconsumption_replay(contract, preregistration, capsule)
        except ControllerPending as error:
            _persist_pending(contract, capsule, error, authorization=existing)
            raise
        if existing is None:
            existing = state.build_authorization(contract, preregistration.value, capsule)
            try:
                _publish_json(Path(fixed["authorization_path"]), existing, "Q6 authorization")
            except (OSError, ControllerPending) as error:
                try:
                    durable = _confirm_exact_json_publication(
                        Path(fixed["authorization_path"]), existing,
                        "Q6 authorization after publication error",
                    )
                except (OSError, ControllerPending, ValueError) as replay_error:
                    pending = ControllerPending(
                        "AUTHORIZATION_PUBLICATION",
                        "Q6_AUTHORIZATION_DURABILITY_UNPROVEN",
                        f"{error}; durability replay: {replay_error}",
                    )
                    _persist_pending(contract, capsule, pending)
                    raise pending
                if not durable:
                    pending = ControllerPending(
                        "AUTHORIZATION_PUBLICATION", "Q6_AUTHORIZATION_NOT_DURABLE",
                        str(error),
                    )
                    _persist_pending(contract, capsule, pending)
                    raise pending
        _reconcile_trajectory(contract, capsule, authorization=existing)
        _clear_pending(contract)
        return existing


def _admit_custody_metadata(contract: Any, ordinal: int) -> None:
    paths = _signer_paths(contract, ordinal)
    worker = contract.value["worker"]
    root = paths["custody_root"]
    private = root / worker["private_key_name"]
    try:
        root_meta = root.lstat()
        key_meta = private.lstat()
        if (
            root.is_symlink() or not stat.S_ISDIR(root_meta.st_mode)
            or stat.S_IMODE(root_meta.st_mode) != 0o700
            or root_meta.st_uid != worker["expected_uid"] or root_meta.st_gid != worker["expected_gid"]
            or private.is_symlink() or not stat.S_ISREG(key_meta.st_mode)
            or stat.S_IMODE(key_meta.st_mode) != 0o600
            or key_meta.st_uid != worker["expected_uid"] or key_meta.st_gid != worker["expected_gid"]
            or key_meta.st_nlink != 1 or not (0 < key_meta.st_size <= MAX_PRIVATE_KEY_BYTES)
        ):
            raise OSError("custody/private-key metadata differs")
    except OSError as error:
        raise ControllerPending(
            f"SIGNER_{ordinal}_CUSTODY_METADATA_ADMISSION",
            "Q6_CUSTODY_METADATA_ADMISSION_FAILED", str(error),
        ) from error
    _require_absent(
        [paths[key] for key in (
            "intent", "verification_intent", "signature", "signature_temp",
            "worker_receipt", "verified_checkpoint",
        )],
        f"SIGNER_{ordinal}_FRESH_OUTPUT_ADMISSION", "Q6_SIGNER_OUTPUT_ALREADY_EXISTS",
    )


def _read_worker_result(paths: Mapping[str, Path], intent: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    try:
        signature = _read_regular(paths["signature"], "Q6 signature", 64)
        value = _read_json_if_present(paths["worker_receipt"], "Q6 worker public receipt")
    except (ControllerPending, ValueError) as error:
        _terminal(
            "Q6_WORKER_RESULT_UNVERIFIABLE_AFTER_INTENT", str(error)
        )
    if len(signature) != 64:
        _terminal("Q6_SIGNATURE_SHAPE_DIFFERS", str(intent["signer_ordinal"]))
    if value is None:
        _terminal("Q6_WORKER_RECEIPT_MISSING_AFTER_INTENT", str(intent["signer_ordinal"]))
    expected = state.expected_worker_receipt(intent=intent, signature_payload=signature)
    if value != expected:
        _terminal("Q6_WORKER_PUBLIC_RECEIPT_DIFFERS", str(intent["signer_ordinal"]))
    try:
        transient_present = _path_present(paths["signature_temp"])
    except ControllerPending as error:
        _terminal("Q6_SIGNATURE_TRANSIENT_ABSENCE_UNPROVEN", str(error))
    if transient_present:
        _terminal("Q6_SIGNATURE_TRANSIENT_REMAINS", str(paths["signature_temp"]))
    return signature, value


def _recover_prefix(
    contract: Any, preregistration: Any, capsule: Mapping[str, Any],
    authorization: Mapping[str, Any], consumption: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[bytes], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    receipts: list[dict[str, Any]] = []
    signatures: list[bytes] = []
    intents: list[dict[str, Any]] = []
    verification_intents: list[dict[str, Any]] = []
    needs_verification: dict[str, Any] | None = None

    def fail(code: str, detail: str) -> NoReturn:
        def conservative_present(path: Path) -> bool:
            try:
                path.lstat()
            except FileNotFoundError:
                return False
            except OSError:
                return True
            return True

        intent_count = sum(
            1 for item in state.SIGNER_ORDINALS
            if conservative_present(_signer_paths(contract, item)["intent"])
        )
        verification_intent_count = sum(
            1 for item in state.SIGNER_ORDINALS
            if conservative_present(_signer_paths(contract, item)["verification_intent"])
        )
        audit = state.q6_observed_action_audit(
            tuple(range(1, len(receipts) + 1)),
            metadata_admission_count=intent_count,
            signing_intent_count=intent_count,
            worker_popen_count=intent_count,
            openssl_sign_count=intent_count,
            private_key_load_count=intent_count,
            signature_output_count=intent_count,
            worker_receipt_count=intent_count,
            verification_intent_count=verification_intent_count,
            verification_count=verification_intent_count,
            valid_signature_count=len(receipts),
            verified_checkpoint_count=len(receipts),
            verified_checkpoint_publication_count=sum(
                1 for item in state.SIGNER_ORDINALS
                if conservative_present(_signer_paths(contract, item)["verified_checkpoint"])
            ),
        )
        raise PrefixReplayTerminal(code, detail, receipts, audit)

    for ordinal in state.SIGNER_ORDINALS:
        paths = _signer_paths(contract, ordinal)
        try:
            intent_value = _read_json_if_present(paths["intent"], f"Q6 signer {ordinal} intent")
        except (ControllerPending, ValueError) as error:
            fail("Q6_SIGNING_INTENT_REPLAY_UNAVAILABLE", str(error))
        if intent_value is None:
            try:
                extra_present = any(_path_present(paths[key]) for key in (
                    "verification_intent", "signature", "signature_temp", "worker_receipt", "verified_checkpoint"
                ))
            except ControllerPending as error:
                fail("Q6_SIGNER_STATE_WITHOUT_INTENT_UNAVAILABLE", str(error))
            if extra_present:
                fail("Q6_SIGNER_STATE_WITHOUT_INTENT", str(ordinal))
            break
        try:
            intent = state.validate_signing_intent(
                intent_value, contract, preregistration.value, capsule,
                authorization, consumption, receipts,
            )
        except ValueError as error:
            fail("Q6_SIGNING_INTENT_REPLAY_DIFFERS", str(error))
        intents.append(intent)
        try:
            signature_present = _path_present(paths["signature"])
            worker_present = _path_present(paths["worker_receipt"])
            verification_present = _path_present(paths["verification_intent"])
            verified_present = _path_present(paths["verified_checkpoint"])
        except ControllerPending as error:
            fail("Q6_SIGNER_PATH_REPLAY_UNAVAILABLE", str(error))
        if not signature_present or not worker_present:
            fail("Q6_SIGNING_OUTCOME_UNKNOWN_AFTER_INTENT", str(ordinal))
        try:
            signature, worker_receipt = _read_worker_result(paths, intent)
        except ControllerTerminal as error:
            fail(error.code, error.detail)
        if not verification_present:
            if verified_present:
                fail("Q6_VERIFIED_CHECKPOINT_WITHOUT_VERIFICATION_INTENT", str(ordinal))
            needs_verification = {
                "ordinal": ordinal, "paths": paths, "intent": intent,
                "signature": signature, "worker_receipt": worker_receipt,
                "completed_receipts": list(receipts),
            }
            break
        try:
            verification_value = _read_json_if_present(paths["verification_intent"], f"Q6 signer {ordinal} verification intent")
        except (ControllerPending, ValueError) as error:
            fail("Q6_VERIFICATION_INTENT_REPLAY_UNAVAILABLE", str(error))
        try:
            verification_intent = state.validate_verification_intent(
                verification_value, contract, capsule, authorization, consumption,
                receipts, intent, worker_receipt=worker_receipt,
                signature_payload=signature,
            )
        except ValueError as error:
            fail("Q6_VERIFICATION_INTENT_REPLAY_DIFFERS", str(error))
        if not verified_present:
            fail("Q6_VERIFICATION_OUTCOME_UNKNOWN_TERMINAL_NO_RESUME", str(ordinal))
        try:
            receipt_value = _read_json_if_present(paths["verified_checkpoint"], f"Q6 signer {ordinal} verified checkpoint")
        except (ControllerPending, ValueError) as error:
            fail("Q6_VERIFIED_CHECKPOINT_REPLAY_UNAVAILABLE", str(error))
        try:
            expected = state.build_signature_receipt(
                contract, preregistration.value, capsule, authorization,
                consumption, receipts, intent, worker_receipt=worker_receipt,
                signature_payload=signature, verification_intent=verification_intent,
            )
        except ValueError as error:
            fail("Q6_VERIFIED_CHECKPOINT_CONSTRUCTION_DIFFERS", str(error))
        if receipt_value != expected:
            fail("Q6_VERIFIED_CHECKPOINT_REPLAY_DIFFERS", str(ordinal))
        receipts.append(expected)
        signatures.append(signature)
        verification_intents.append(verification_intent)
    start = len(receipts) + (2 if needs_verification is not None else 1)
    for ordinal in range(start, 4):
        paths = _signer_paths(contract, ordinal)
        try:
            if any(_path_present(path) for key, path in paths.items() if key != "custody_root"):
                fail("Q6_NONPREFIX_SIGNER_STATE", str(ordinal))
        except ControllerPending as error:
            fail("Q6_NONPREFIX_SIGNER_STATE_UNAVAILABLE", str(error))
    return receipts, signatures, intents, verification_intents, needs_verification


def _publish_global_checkpoint(
    contract: Any, capsule: Mapping[str, Any], authorization: Mapping[str, Any],
    consumption: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = state.build_checkpoint(contract, capsule, authorization, consumption, receipts)
    path = Path(contract.value["fixed_paths"]["checkpoint_path"])
    existing = _read_json_if_present(path, "Q6 global checkpoint")
    if existing is not None:
        prefix = existing.get("completed_prefix")
        if (
            type(prefix) is not list
            or tuple(prefix) not in state.PENDING_PREFIXES
            or len(prefix) > len(receipts)
        ):
            _terminal("Q6_GLOBAL_CHECKPOINT_LEADS_VERIFIED_PREFIX", str(prefix))
        prior_expected = state.build_checkpoint(
            contract, capsule, authorization, consumption,
            receipts[: len(prefix)],
        )
        if existing != prior_expected:
            _terminal("Q6_GLOBAL_CHECKPOINT_REPLAY_DIFFERS", str(prefix))
    if existing != expected:
        try:
            _replace_json(path, expected, "Q6 global checkpoint")
        except (OSError, ControllerPending) as error:
            try:
                if _confirm_exact_json_publication(
                    path, expected, "Q6 global checkpoint after publication error"
                ):
                    return expected
            except (OSError, ControllerPending, ValueError):
                pass
            raise ControllerPending(
                "CLEAN_CHECKPOINT_PUBLICATION", "Q6_GLOBAL_CHECKPOINT_NOT_DURABLE", str(error)
            ) from error
    else:
        try:
            _fsync_parent(path)
        except OSError as error:
            raise ControllerPending(
                "CLEAN_CHECKPOINT_PUBLICATION",
                "Q6_GLOBAL_CHECKPOINT_DURABILITY_UNPROVEN", str(error),
            ) from error
    return expected


def _verify_one(
    contract: Any, preregistration: Any, capsule: Mapping[str, Any],
    authorization: Mapping[str, Any], consumption: Mapping[str, Any],
    receipts: list[dict[str, Any]], signatures: list[bytes],
    pending: Mapping[str, Any],
) -> None:
    ordinal = pending["ordinal"]
    paths = pending["paths"]
    intent = pending["intent"]
    signature = pending["signature"]
    worker_receipt = pending["worker_receipt"]
    verification_intent = state.build_verification_intent(
        contract, capsule, authorization, consumption, receipts, intent,
        worker_receipt=worker_receipt, signature_payload=signature,
        created_at_unix_seconds=int(time.time()),
    )
    try:
        _publish_json(paths["verification_intent"], verification_intent, f"Q6 signer {ordinal} verification intent")
        durable = _read_json_if_present(paths["verification_intent"], f"Q6 signer {ordinal} verification intent readback")
        if durable != verification_intent:
            raise OSError("verification intent readback differs")
    except (OSError, ControllerPending) as error:
        try:
            durable = _confirm_exact_json_publication(
                paths["verification_intent"], verification_intent,
                f"Q6 signer {ordinal} verification intent after publication error",
            )
        except (OSError, ControllerPending, ValueError) as replay_error:
            _terminal(
                "Q6_VERIFICATION_INTENT_PUBLICATION_OUTCOME_UNKNOWN",
                f"{error}; readback: {replay_error}",
            )
        if not durable:
            _terminal("Q6_VERIFICATION_INTENT_PUBLICATION_OUTCOME_UNKNOWN", str(error))
    public_key = bytes.fromhex(intent["public_key_hex"])
    try:
        wire.verify_ed25519_v1(public_key, signature, _read_regular(
            Path(contract.value["fixed_paths"]["signing_message_path"]),
            "Q6 signing message", state.MAX_ARTIFACT_BYTES,
        ))
    except Exception as error:
        _terminal("Q6_SIGNATURE_VERIFICATION_FAILED", f"signer {ordinal}: {error}")
    verified = state.build_signature_receipt(
        contract, preregistration.value, capsule, authorization, consumption,
        receipts, intent, worker_receipt=worker_receipt,
        signature_payload=signature, verification_intent=verification_intent,
    )
    try:
        _publish_json(paths["verified_checkpoint"], verified, f"Q6 signer {ordinal} verified checkpoint")
    except (OSError, ControllerPending) as error:
        try:
            durable = _confirm_exact_json_publication(
                paths["verified_checkpoint"], verified,
                f"Q6 signer {ordinal} verified checkpoint after publication error",
            )
        except (OSError, ControllerPending, ValueError) as replay_error:
            _terminal(
                "Q6_VERIFIED_CHECKPOINT_PUBLICATION_FAILED_AFTER_VERIFICATION",
                f"{error}; readback: {replay_error}",
            )
        if not durable:
            _terminal(
                "Q6_VERIFIED_CHECKPOINT_PUBLICATION_FAILED_AFTER_VERIFICATION",
                str(error),
            )
    receipts.append(verified)
    signatures.append(signature)
    _publish_global_checkpoint(contract, capsule, authorization, consumption, receipts)


def _seal_invalidated(
    contract: Any, capsule: Mapping[str, Any], authorization: Mapping[str, Any],
    consumption: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]], *,
    phase: str, code: str, detail: str,
    observed_action_audit: Mapping[str, Any],
) -> dict[str, Any]:
    terminal = state.build_invalidated_terminal(
        contract, capsule, authorization, consumption, receipts,
        failure_phase=phase, failure_code=code, failure_detail=detail,
        observed_action_audit=observed_action_audit,
        pending_history_receipt=_pending_history_receipt(contract),
    )
    try:
        _publish_json(Path(contract.value["fixed_paths"]["terminal_path"]), terminal, "Q6 invalidated terminal")
    except (OSError, ControllerPending) as error:
        try:
            durable = _confirm_exact_json_publication(
                Path(contract.value["fixed_paths"]["terminal_path"]), terminal,
                "Q6 invalidated terminal after publication error",
            )
        except (OSError, ControllerPending, ValueError) as replay_error:
            _terminal(
                "Q6_INVALIDATED_TERMINAL_PUBLICATION_FAILED_NO_RESUME",
                f"{error}; durability replay: {replay_error}",
            )
        if not durable:
            _terminal("Q6_INVALIDATED_TERMINAL_PUBLICATION_FAILED_NO_RESUME", str(error))
    _reconcile_trajectory(
        contract, capsule, authorization=authorization, consumption=consumption,
        receipts=receipts, terminal=terminal,
    )
    _clear_pending_after_terminal(contract)
    return terminal


def _failure_audit(
    contract: Any,
    receipts: Sequence[Mapping[str, Any]],
    ordinal: int,
    code: str,
    *,
    worker_may_have_run: bool = True,
) -> dict[str, Any]:
    prefix_count = len(receipts)
    worker_count = ordinal if worker_may_have_run else prefix_count
    verification_intent_present = _path_present(
        _signer_paths(contract, ordinal)["verification_intent"]
    )
    verification_intent_count = ordinal if verification_intent_present else prefix_count
    verification_count = (
        prefix_count
        if code == "Q6_VERIFICATION_INTENT_PUBLICATION_OUTCOME_UNKNOWN"
        else verification_intent_count
    )
    valid_count = (
        ordinal
        if code == "Q6_VERIFIED_CHECKPOINT_PUBLICATION_FAILED_AFTER_VERIFICATION"
        else prefix_count
    )
    return state.q6_observed_action_audit(
        tuple(range(1, prefix_count + 1)),
        metadata_admission_count=ordinal,
        signing_intent_count=(
            ordinal
            if _path_present(_signer_paths(contract, ordinal)["intent"])
            else prefix_count
        ),
        worker_popen_count=worker_count,
        openssl_sign_count=worker_count,
        private_key_load_count=worker_count,
        signature_output_count=worker_count,
        worker_receipt_count=worker_count,
        verification_intent_count=verification_intent_count,
        verification_count=verification_count,
        valid_signature_count=valid_count,
        verified_checkpoint_count=prefix_count,
        verified_checkpoint_publication_count=(
            ordinal
            if code == "Q6_VERIFIED_CHECKPOINT_PUBLICATION_FAILED_AFTER_VERIFICATION"
            else prefix_count
        ),
    )


def _finalization_failure_audit(
    receipts: Sequence[Mapping[str, Any]], *, envelope: int = 0, issuance: int = 0
) -> dict[str, Any]:
    count = len(receipts)
    return state.q6_observed_action_audit(
        tuple(range(1, count + 1)),
        metadata_admission_count=count,
        signing_intent_count=count,
        worker_popen_count=count,
        openssl_sign_count=count,
        private_key_load_count=count,
        signature_output_count=count,
        worker_receipt_count=count,
        verification_intent_count=count,
        verification_count=count,
        valid_signature_count=count,
        verified_checkpoint_count=count,
        verified_checkpoint_publication_count=count,
        envelope_publication_count=envelope,
        issuance_count=issuance,
    )


def _not_authorized(capsule: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "hegel-phase3a-q6-controller-not-authorized/1",
        "status": state.NOT_AUTHORIZED,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_consumption_count": 0,
        "logical_campaign_count": 0,
        "q6_action_audit": state.ZERO_ACTION_AUDIT,
    }


def _run_v1_inner(contract_path: Path, capsule_digest: str) -> dict[str, Any]:
    contract, preregistration = _load_contract(contract_path)
    fixed = contract.value["fixed_paths"]
    with _campaign_lock(contract):
        capsule = _read_capsule(contract, preregistration, capsule_digest)
        try:
            authorization = _read_authorization(
                contract, preregistration, capsule
            )
        except ControllerPending as error:
            _persist_pending(contract, capsule, error)
            raise
        if authorization is None:
            if _execution_state_exists(contract):
                _terminal("Q6_AUTHORIZATION_MISSING_AFTER_EXECUTION_STATE", capsule_digest)
            return _not_authorized(capsule)
        consumption = _read_consumption(contract, preregistration, capsule, authorization)
        if consumption is None:
            if _execution_state_exists(contract):
                _terminal("Q6_CONSUMPTION_MISSING_AFTER_EXECUTION_STATE", capsule_digest)
            try:
                _preconsumption_replay(contract, preregistration, capsule)
            except ControllerPending as error:
                _persist_pending(contract, capsule, error, authorization=authorization)
                raise
            consumption = state.build_consumption(contract, preregistration.value, capsule, authorization)
            try:
                _publish_json(Path(fixed["consumption_path"]), consumption, "Q6 consumption")
            except (OSError, ControllerPending) as error:
                try:
                    durable = _confirm_exact_json_publication(
                        Path(fixed["consumption_path"]), consumption,
                        "Q6 consumption after publication error",
                    )
                except (OSError, ControllerPending, ValueError) as replay_error:
                    pending = ControllerPending(
                        "CONSUMPTION_PUBLICATION",
                        "Q6_CONSUMPTION_DURABILITY_UNPROVEN",
                        f"{error}; durability replay: {replay_error}",
                    )
                    _persist_pending(
                        contract, capsule, pending, authorization=authorization
                    )
                    raise pending
                if not durable:
                    pending = ControllerPending("CONSUMPTION_PUBLICATION", "Q6_CONSUMPTION_NOT_DURABLE", str(error))
                    _persist_pending(contract, capsule, pending, authorization=authorization)
                    raise pending
            try:
                _clear_pending(contract)
            except ControllerPending as error:
                _persist_pending(
                    contract, capsule, error, authorization=authorization,
                    consumption=consumption, receipts=(),
                )
                raise

        try:
            terminal_value = _read_json_if_present(
                Path(fixed["terminal_path"]), "Q6 terminal"
            )
        except (ControllerPending, ValueError) as error:
            raise ControllerPartial(
                PARTIAL_TERMINAL_TO_TRAJECTORY_SUFFIX, str(error)
            ) from error

        # A durable terminal freezes the pending-history identity.  Replay it
        # before inspecting any actor or artifact state so no later local
        # unavailability can append history or publish a replacement terminal.
        if terminal_value is not None:
            try:
                history = _pending_history_receipt(contract)
                terminal_status = terminal_value.get("status")
                if terminal_status == state.INVALIDATED_TERMINAL_NO_RESUME:
                    terminal = state.validate_terminal(
                        terminal_value, contract, capsule, authorization,
                        consumption, pending_history_receipt=history,
                    )
                    trajectory_receipts = [
                        {"signer_ordinal": ordinal}
                        for ordinal in terminal["outcome"]["completed_prefix"]
                    ]
                elif terminal_status == wire.Q6_SUCCESS_STATUS:
                    try:
                        receipts, _, _, _, _ = _recover_prefix(
                            contract, preregistration, capsule, authorization,
                            consumption,
                        )
                    except PrefixReplayTerminal as error:
                        raise ValueError(
                            f"terminal signer replay differs: {error.code}: {error.detail}"
                        ) from error
                    if len(receipts) != 3:
                        raise ValueError("successful terminal signer prefix differs")
                    q5b, q4 = _read_predecessor_payloads(contract)
                    artifact = _read_regular(
                        Path(fixed["artifact_path"]), "Q6 issuance artifact",
                        state.MAX_ARTIFACT_BYTES,
                    )
                    envelope = _read_regular(
                        Path(fixed["envelope_path"]), "Q6 certificate envelope",
                        state.MAX_ARTIFACT_BYTES,
                    )
                    terminal_checkpoint = state.build_checkpoint(
                        contract, capsule, authorization, consumption, receipts
                    )
                    terminal = state.validate_terminal(
                        terminal_value, contract, capsule, authorization,
                        consumption, artifact_payload=artifact,
                        envelope_payload=envelope, q4_artifact_payload=q4,
                        q5b_artifact_payload=q5b["artifact"],
                        verified_checkpoints=receipts,
                        complete_checkpoint_digest=terminal_checkpoint[
                            "checkpoint_digest"
                        ],
                        pending_history_receipt=history,
                    )
                    trajectory_receipts = receipts
                else:
                    raise ValueError("Q6 terminal status differs")
                _reconcile_trajectory(
                    contract, capsule, authorization=authorization,
                    consumption=consumption, receipts=trajectory_receipts,
                    terminal=terminal,
                )
                _clear_pending_after_terminal(contract)
                return terminal
            except ControllerPartial:
                raise
            except (ControllerPending, ControllerTerminal, ValueError, TypeError, OSError) as error:
                raise ControllerPartial(
                    PARTIAL_TERMINAL_TO_TRAJECTORY_SUFFIX, str(error)
                ) from error

        artifact_path = Path(fixed["artifact_path"])
        envelope_path = Path(fixed["envelope_path"])
        try:
            finalization_started = _path_present(artifact_path) or _path_present(
                envelope_path
            )
        except ControllerPending as error:
            raise ControllerPartial(
                PARTIAL_ARTIFACT_TO_TERMINAL, str(error)
            ) from error

        try:
            receipts, signatures, _, _, needs_verification = _recover_prefix(
                contract, preregistration, capsule, authorization, consumption
            )
        except PrefixReplayTerminal as error:
            if finalization_started:
                raise ControllerPartial(
                    PARTIAL_ARTIFACT_TO_TERMINAL,
                    f"{error.code}: {error.detail}",
                ) from error
            return _seal_invalidated(
                contract, capsule, authorization, consumption, error.receipts,
                phase="POSTCONSUMPTION_REPLAY", code=error.code, detail=error.detail,
                observed_action_audit=error.observed_action_audit,
            )
        try:
            q5b, q4 = _replay_capsule_public_gates(
                contract, preregistration, capsule, include_startup_smokes=False
            )
        except ControllerPending as error:
            if finalization_started:
                raise ControllerPartial(
                    PARTIAL_ARTIFACT_TO_TERMINAL, str(error)
                ) from error
            _persist_pending(
                contract, capsule, error, authorization=authorization,
                consumption=consumption, receipts=receipts,
            )
            raise

        if needs_verification is not None:
            try:
                _verify_one(
                    contract, preregistration, capsule, authorization, consumption,
                    receipts, signatures, needs_verification,
                )
            except ControllerTerminal as error:
                return _seal_invalidated(
                    contract, capsule, authorization, consumption, receipts,
                    phase=f"SIGNER_{needs_verification['ordinal']}_VERIFICATION",
                    code=error.code, detail=error.detail,
                    observed_action_audit=_failure_audit(
                        contract, receipts, needs_verification["ordinal"], error.code
                    ),
                )
            except ControllerPending as error:
                _persist_pending(
                    contract, capsule, error, authorization=authorization,
                    consumption=consumption, receipts=receipts,
                )
                raise

        try:
            _publish_global_checkpoint(contract, capsule, authorization, consumption, receipts)
            _reconcile_trajectory(contract, capsule, authorization=authorization, consumption=consumption, receipts=receipts)
        except ControllerTerminal as error:
            return _seal_invalidated(
                contract, capsule, authorization, consumption, receipts,
                phase="CLEAN_CHECKPOINT_REPLAY", code=error.code,
                detail=error.detail,
                observed_action_audit=_finalization_failure_audit(receipts),
            )
        except ControllerPending as error:
            _persist_pending(contract, capsule, error, authorization=authorization, consumption=consumption, receipts=receipts)
            raise

        while len(receipts) < 3:
            ordinal = len(receipts) + 1
            try:
                _replay_capsule_public_gates(
                    contract, preregistration, capsule,
                    include_startup_smokes=False,
                )
                _admit_custody_metadata(contract, ordinal)
            except ControllerPending as error:
                _persist_pending(contract, capsule, error, authorization=authorization, consumption=consumption, receipts=receipts)
                raise
            intent = state.build_signing_intent(
                contract, preregistration.value, capsule, authorization, consumption,
                receipts, created_at_unix_seconds=int(time.time()),
            )
            paths = _signer_paths(contract, ordinal)
            try:
                _publish_json(paths["intent"], intent, f"Q6 signer {ordinal} intent")
            except (OSError, ControllerPending) as error:
                try:
                    durable = _confirm_exact_json_publication(
                        paths["intent"], intent,
                        f"Q6 signer {ordinal} intent after publication error",
                    )
                except (OSError, ControllerPending, ValueError) as replay_error:
                    return _seal_invalidated(
                        contract, capsule, authorization, consumption, receipts,
                        phase=f"SIGNER_{ordinal}_INTENT",
                        code="Q6_SIGNING_INTENT_DURABILITY_UNPROVEN",
                        detail=f"{error}; durability replay: {replay_error}",
                        observed_action_audit=_failure_audit(
                            contract, receipts, ordinal,
                            "Q6_SIGNING_INTENT_PUBLICATION_OUTCOME_UNKNOWN",
                            worker_may_have_run=False,
                        ),
                    )
                if not durable:
                    pending_error = ControllerPending(
                        f"SIGNER_{ordinal}_INTENT_PUBLICATION",
                        "Q6_SIGNING_INTENT_NOT_DURABLE", str(error),
                    )
                    _persist_pending(
                        contract, capsule, pending_error,
                        authorization=authorization, consumption=consumption,
                        receipts=receipts,
                    )
                    raise pending_error
            try:
                returned = _worker_result(
                    _worker_command(contract, capsule, ordinal, startup_smoke=False),
                    f"Q6_SIGNER_{ordinal}_WORKER", post_intent=True,
                )
                signature, worker_receipt = _read_worker_result(paths, intent)
                if returned != worker_receipt:
                    _terminal("Q6_WORKER_STDOUT_DIFFERS_FROM_DURABLE_RECEIPT", str(ordinal))
                pending = {
                    "ordinal": ordinal, "paths": paths, "intent": intent,
                    "signature": signature, "worker_receipt": worker_receipt,
                }
                _verify_one(
                    contract, preregistration, capsule, authorization, consumption,
                    receipts, signatures, pending,
                )
            except ControllerTerminal as error:
                return _seal_invalidated(
                    contract, capsule, authorization, consumption, receipts,
                    phase=f"SIGNER_{ordinal}", code=error.code, detail=error.detail,
                    observed_action_audit=_failure_audit(
                        contract, receipts, ordinal, error.code
                    ),
                )
            except ControllerPending as error:
                _persist_pending(
                    contract, capsule, error, authorization=authorization,
                    consumption=consumption, receipts=receipts,
                )
                raise
            try:
                _reconcile_trajectory(
                    contract, capsule, authorization=authorization,
                    consumption=consumption, receipts=receipts,
                )
            except ControllerTerminal as error:
                return _seal_invalidated(
                    contract, capsule, authorization, consumption, receipts,
                    phase=f"SIGNER_{ordinal}_TRAJECTORY",
                    code=error.code, detail=error.detail,
                    observed_action_audit=_finalization_failure_audit(receipts),
                )
            except ControllerPending as error:
                _persist_pending(
                    contract, capsule, error, authorization=authorization,
                    consumption=consumption, receipts=receipts,
                )
                raise

        try:
            q5b, q4 = _replay_capsule_public_gates(
                contract, preregistration, capsule, include_startup_smokes=False
            )
        except ControllerPending as error:
            if finalization_started:
                raise ControllerPartial(
                    PARTIAL_ARTIFACT_TO_TERMINAL, str(error)
                ) from error
            _persist_pending(
                contract, capsule, error, authorization=authorization,
                consumption=consumption, receipts=receipts,
            )
            raise
        try:
            checkpoint = _publish_global_checkpoint(
                contract, capsule, authorization, consumption, receipts
            )
        except ControllerPending as error:
            if finalization_started:
                raise ControllerPartial(
                    PARTIAL_ARTIFACT_TO_TERMINAL, str(error)
                ) from error
            _persist_pending(
                contract, capsule, error, authorization=authorization,
                consumption=consumption, receipts=receipts,
            )
            raise
        except ControllerTerminal as error:
            return _seal_invalidated(
                contract, capsule, authorization, consumption, receipts,
                phase="COMPLETE_CHECKPOINT_REPLAY", code=error.code,
                detail=error.detail,
                observed_action_audit=_finalization_failure_audit(receipts),
            )
        try:
            receipt = state.build_issuance_artifact(
                capsule, receipts, signatures, q4_artifact_payload=q4,
                q5b_artifact_payload=q5b["artifact"],
                complete_checkpoint_digest=checkpoint["checkpoint_digest"],
            )
        except ControllerPending as error:
            raise ControllerPartial(
                PARTIAL_ARTIFACT_TO_TERMINAL, str(error)
            ) from error
        except ValueError as error:
            return _seal_invalidated(
                contract, capsule, authorization, consumption, receipts,
                phase="ISSUANCE_ARTIFACT_CONSTRUCTION",
                code="Q6_ISSUANCE_ARTIFACT_CONSTRUCTION_REJECTED",
                detail=str(error),
                observed_action_audit=_finalization_failure_audit(receipts),
            )
        artifact = receipt.canonical_bytes
        envelope = receipt.envelope.canonical_bytes
        for path, payload, label in (
            (envelope_path, envelope, "Q6 certificate envelope"),
            (artifact_path, artifact, "Q6 issuance artifact"),
        ):
            if _path_present(path):
                try:
                    if _read_regular(path, label, state.MAX_ARTIFACT_BYTES) != payload:
                        raise ValueError(f"{label} differs from deterministic replay")
                    _fsync_parent(path)
                except (ControllerPending, ValueError, OSError) as error:
                    return _seal_invalidated(
                        contract, capsule, authorization, consumption, receipts,
                        phase="FINAL_ARTIFACT_REPLAY",
                        code="Q6_FINAL_ARTIFACT_REPLAY_DIFFERS", detail=str(error),
                        observed_action_audit=_finalization_failure_audit(
                            receipts,
                            envelope=int(_path_present(envelope_path)),
                            issuance=int(_path_present(artifact_path)),
                        ),
                    )
                continue
            try:
                _publish_bytes(path, payload, label, mode=0o644)
            except (OSError, ControllerPending) as error:
                try:
                    if _confirm_exact_publication(
                        path, payload, f"{label} after publication error",
                        maximum_bytes=state.MAX_ARTIFACT_BYTES,
                    ):
                        continue
                except (OSError, ControllerPending, ValueError):
                    pass
                try:
                    final_output_present = _path_present(envelope_path) or _path_present(
                        artifact_path
                    )
                except ControllerPending:
                    final_output_present = True
                if final_output_present:
                    raise ControllerPartial(
                        PARTIAL_ARTIFACT_TO_TERMINAL, str(error)
                    ) from error
                pending = ControllerPending(
                    "FINALIZATION", "Q6_FINAL_ARTIFACT_PUBLICATION_NOT_DURABLE",
                    str(error),
                )
                _persist_pending(
                    contract, capsule, pending, authorization=authorization,
                    consumption=consumption, receipts=receipts,
                )
                raise pending
        try:
            receipt = state.validate_artifact_binding(
                artifact, envelope, capsule, q4_artifact_payload=q4,
                q5b_artifact_payload=q5b["artifact"],
                complete_checkpoint_digest=checkpoint["checkpoint_digest"],
                verified_checkpoints=receipts,
            )
        except (TypeError, ValueError) as error:
            return _seal_invalidated(
                contract, capsule, authorization, consumption, receipts,
                phase="FINAL_ARTIFACT_REPLAY",
                code="Q6_FINAL_ARTIFACT_REPLAY_DIFFERS", detail=str(error),
                observed_action_audit=_finalization_failure_audit(
                    receipts, envelope=1, issuance=1
                ),
            )
        try:
            terminal = state.build_complete_terminal(
                contract, capsule, authorization, consumption, artifact, envelope,
                receipt, receipts,
                complete_checkpoint_digest=checkpoint["checkpoint_digest"],
                pending_history_receipt=_pending_history_receipt(contract),
            )
        except ControllerPending as error:
            raise ControllerPartial(
                PARTIAL_ARTIFACT_TO_TERMINAL, str(error)
            ) from error
        except ValueError as error:
            return _seal_invalidated(
                contract, capsule, authorization, consumption, receipts,
                phase="TERMINAL_CONSTRUCTION",
                code="Q6_TERMINAL_CONSTRUCTION_REJECTED", detail=str(error),
                observed_action_audit=_finalization_failure_audit(
                    receipts, envelope=1, issuance=1
                ),
            )
        try:
            _publish_json(Path(fixed["terminal_path"]), terminal, "Q6 terminal")
        except (OSError, ControllerPending) as error:
            try:
                durable = _confirm_exact_json_publication(
                    Path(fixed["terminal_path"]), terminal,
                    "Q6 terminal after publication error",
                )
            except (OSError, ControllerPending, ValueError) as replay_error:
                raise ControllerPartial(
                    PARTIAL_ARTIFACT_TO_TERMINAL,
                    f"{error}; durability replay: {replay_error}",
                ) from error
            if not durable:
                raise ControllerPartial(
                    PARTIAL_ARTIFACT_TO_TERMINAL, str(error)
                ) from error
        try:
            _reconcile_trajectory(
                contract, capsule, authorization=authorization,
                consumption=consumption, receipts=receipts, terminal=terminal,
            )
            _clear_pending_after_terminal(contract)
        except ControllerPartial:
            raise
        except (ControllerPending, ControllerTerminal, OSError, ValueError) as error:
            raise ControllerPartial(
                PARTIAL_TERMINAL_TO_TRAJECTORY_SUFFIX, str(error)
            ) from error
        return terminal


def run_v1(contract_path: Path, capsule_digest: str) -> dict[str, Any]:
    try:
        return _run_v1_inner(contract_path, capsule_digest)
    except ControllerSealedTerminal as error:
        return error.terminal_record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-capsule")
    prepare.add_argument("--source-commit", required=True)
    prepare.add_argument("--run-id")
    prepare.add_argument("--created-at-unix-seconds", type=int)
    prepare.add_argument("--certificate-created-at-unix-seconds", type=int, required=True)
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
                certificate_created_at_unix_seconds=arguments.certificate_created_at_unix_seconds,
            )
        elif arguments.command == "record-authorization":
            result = record_authorization(arguments.contract, arguments.capsule_digest)
        else:
            result = run_v1(arguments.contract, arguments.capsule_digest)
    except ControllerPending as error:
        output = error.pending_record or {
            "status": state.PENDING_NOT_CONSUMED,
            "failure_phase": error.phase,
            "failure_code": error.condition,
            "failure_detail": error.detail,
            "authorization_consumption_count": 0,
            "logical_campaign_count": 0,
            "completed_prefix": [],
        }
        print(state.canonical_json_bytes(output).decode("ascii"), end="")
        return PENDING_EXIT_CODE
    except ControllerTerminal as error:
        print(state.canonical_json_bytes({
            "status": state.INVALIDATED_TERMINAL_NO_RESUME,
            "failure_code": error.code, "failure_detail": error.detail,
        }).decode("ascii"), end="")
        return TERMINAL_EXIT_CODE
    except ControllerPartial as error:
        print(state.canonical_json_bytes({
            "status": state.PARTIAL_FINALIZATION_REQUIRED,
            "partial_finalization_kind": error.kind,
            "detail": error.detail,
            "authorization_consumption_count": 1,
            "logical_campaign_count": 1,
        }).decode("ascii"), end="")
        return TERMINAL_EXIT_CODE
    print(state.canonical_json_bytes(result).decode("ascii"), end="")
    return NOT_AUTHORIZED_EXIT_CODE if result.get("status") == state.NOT_AUTHORIZED else 0


if __name__ == "__main__":
    raise SystemExit(main())
