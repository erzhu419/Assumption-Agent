#!/usr/bin/env python3
"""Target-blind preflight and one-shot Phase-3A-Q2 membership campaign."""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import fcntl
import importlib
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import types
from typing import Any, Iterator, Mapping, NoReturn, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src/hegel_machine"
PACKAGE_NAME = "hegel_machine"
if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(PACKAGE_ROOT)]
    package.__package__ = PACKAGE_NAME
    sys.modules[PACKAGE_NAME] = package

admission = importlib.import_module("hegel_machine.phase3_q2_admission_v1")
archive_reader = importlib.import_module("hegel_machine.phase3_q2_archive_reader_v1")
membership = importlib.import_module("hegel_machine.phase3_q2_role_membership_v1")
state = importlib.import_module("hegel_machine.phase3_q2_campaign_v1")


DEFAULT_CONTRACT = PROJECT_ROOT / "config/phase3_q2_role_membership_actual_v1.json"
PENDING_EXIT_CODE = 75
NOT_AUTHORIZED = "Q2_ROLE_MEMBERSHIP_NOT_AUTHORIZED_NOT_CONSUMED"
PARTIAL_FINALIZATION = "Q2_ROLE_MEMBERSHIP_PARTIAL_FINALIZATION_REQUIRED"
UNSEALED_TERMINAL = "Q2_ROLE_MEMBERSHIP_UNSEALED_TERMINAL_NO_RESUME"
PENDING_PUBLICATION_UNAVAILABLE = (
    "Q2_ROLE_MEMBERSHIP_PENDING_PUBLICATION_UNAVAILABLE_NOT_CONSUMED"
)

_SOURCE_ROOTS = (
    "src/hegel_machine/phase3_m25_rows_v1.py",
    "src/hegel_machine/phase3_q2_admission_v1.py",
    "src/hegel_machine/phase3_q2_archive_reader_v1.py",
    "src/hegel_machine/phase3_q2_campaign_v1.py",
    "src/hegel_machine/phase3_q2_role_membership_v1.py",
    "tools/phase3_q2_role_membership_v1.py",
)


class ControllerPending(RuntimeError):
    def __init__(self, phase: str, condition: str, detail: str) -> None:
        super().__init__(f"{phase}/{condition}: {detail}")
        self.phase = phase
        self.condition = condition
        self.detail = detail


class ControllerTerminal(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _pending(phase: str, condition: str, detail: str) -> NoReturn:
    raise ControllerPending(phase, condition, detail)


def _terminal(code: str, detail: str) -> NoReturn:
    raise ControllerTerminal(code, detail)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not path.is_dir():
            raise OSError("path is not a directory")
    except OSError as error:
        _pending("LOCAL_STATE", "Q2_LOCAL_PUBLICATION_UNAVAILABLE", f"{path}: {error}")


def _ensure_roots(contract: Any) -> None:
    fixed = contract.value["fixed_paths"]
    for path in (
        Path(fixed["run_root"]),
        Path(fixed["ledger_root"]),
        Path(fixed["artifact_path"]).parent,
        Path(fixed["campaign_lock_path"]).parent,
    ):
        _ensure_directory(path)


@contextmanager
def _campaign_lock(contract: Any) -> Iterator[None]:
    path = Path(contract.value["fixed_paths"]["campaign_lock_path"])
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        _pending("LOCAL_STATE", "Q2_LOCAL_PUBLICATION_UNAVAILABLE", f"{path}: {error}")
    try:
        yield
    finally:
        os.close(descriptor)


def _stage_path(path: Path, marker: str) -> Path:
    return path.with_name(path.name + ".stage-" + marker[-16:])


def _publish_bytes(path: Path, payload: bytes, marker: str) -> None:
    if path.exists():
        try:
            if path.read_bytes() == payload:
                return
        except OSError as error:
            _pending("LOCAL_STATE", "Q2_LOCAL_PUBLICATION_UNAVAILABLE", str(error))
        _terminal("Q2_NO_REPLACE_PATH_OCCUPIED", str(path))
    stage = _stage_path(path, marker)
    try:
        if stage.exists():
            stage.unlink()
            _fsync_directory(path.parent)
        descriptor = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(stage, path)
        _fsync_directory(path.parent)
        stage.unlink()
        _fsync_directory(path.parent)
    except OSError as error:
        try:
            if stage.exists():
                stage.unlink()
                _fsync_directory(path.parent)
        except OSError:
            pass
        _pending("LOCAL_STATE", "Q2_LOCAL_PUBLICATION_UNAVAILABLE", f"{path}: {error}")


def _publish_json(path: Path, value: Mapping[str, Any], marker: str) -> None:
    _publish_bytes(path, state.canonical_json_bytes(dict(value)), marker)


def _replace_json(path: Path, value: Mapping[str, Any], marker: str) -> None:
    stage = _stage_path(path, marker)
    try:
        if stage.exists():
            stage.unlink()
        descriptor = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(state.canonical_json_bytes(dict(value)))
            output.flush()
            os.fsync(output.fileno())
        os.replace(stage, path)
        _fsync_directory(path.parent)
    except OSError as error:
        try:
            if stage.exists():
                stage.unlink()
        except OSError:
            pass
        _pending("LOCAL_STATE", "Q2_LOCAL_PUBLICATION_UNAVAILABLE", f"{path}: {error}")


def _append_trajectory(
    contract: Any,
    event: str,
    phase: str,
    *,
    capsule_digest: str | None = None,
    detail: Mapping[str, Any] | None = None,
) -> None:
    path = Path(contract.value["fixed_paths"]["trajectory_path"])
    existed = path.exists()
    row = {
        "schema_version": "hegel-phase3a-q2-trajectory-event/1",
        "transition_authority": False,
        "action_audit_evidence": True,
        "at_unix_time_ns": time.time_ns(),
        "event": event,
        "phase": phase,
        "capsule_digest": capsule_digest,
        "detail": dict(detail or {}),
    }
    try:
        with path.open("ab") as output:
            output.write(state.canonical_json_bytes(row))
            output.flush()
            os.fsync(output.fileno())
        if not existed:
            _fsync_directory(path.parent)
    except OSError as error:
        _pending("TRAJECTORY", "Q2_LOCAL_PUBLICATION_UNAVAILABLE", str(error))


def _append_trajectory_after_state(
    contract: Any,
    event: str,
    phase: str,
    *,
    capsule_digest: str | None = None,
    detail: Mapping[str, Any] | None = None,
) -> None:
    """Best-effort audit after an authoritative state is already durable."""
    try:
        _append_trajectory(
            contract,
            event,
            phase,
            capsule_digest=capsule_digest,
            detail=detail,
        )
    except ControllerPending:
        pass


def _read_state(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        payload = path.read_bytes()
        return payload, state.strict_json_bytes(payload, label)
    except (OSError, ValueError) as error:
        _pending("LOCAL_STATE", "Q2_LOCAL_PUBLICATION_UNAVAILABLE", f"{label}: {error}")


def _load_contract(path: Path) -> Any:
    try:
        return state.load_contract(path)
    except (OSError, ValueError) as error:
        _pending("CONTRACT", "Q2_LOCAL_IMPLEMENTATION_UNAVAILABLE", str(error))


def _load_preregistration(contract: Any) -> tuple[bytes, dict[str, Any]]:
    path = Path(contract.value["preregistration_path"])
    try:
        raw = path.read_bytes()
        value = state.validate_preregistration_payload(
            raw,
            contract.value["preregistration_raw_sha256"],
        )
    except (OSError, ValueError) as error:
        _pending("PREREGISTRATION", "Q2_LOCAL_IMPLEMENTATION_UNAVAILABLE", str(error))
    return raw, value


def _imported_project_modules(path: Path) -> tuple[str, ...]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError) as error:
        _pending("SOURCE_CLOSURE", "Q2_LOCAL_IMPLEMENTATION_UNAVAILABLE", str(error))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level and node.module:
                names.add(node.module.split(".")[0])
            elif node.level and node.module is None:
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif node.module and node.module.startswith("hegel_machine."):
                names.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            names.update(
                alias.name.split(".")[1]
                for alias in node.names
                if alias.name.startswith("hegel_machine.")
            )
    return tuple(sorted(names))


def _source_paths() -> tuple[str, ...]:
    pending = list(_SOURCE_ROOTS[:-1])
    discovered = {_SOURCE_ROOTS[-1]}
    while pending:
        relative = pending.pop()
        if relative in discovered:
            continue
        discovered.add(relative)
        for module_name in _imported_project_modules(PROJECT_ROOT / relative):
            candidate = f"src/hegel_machine/{module_name}.py"
            if (PROJECT_ROOT / candidate).is_file() and candidate not in discovered:
                pending.append(candidate)
    return tuple(sorted(discovered))


def _source_closure_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative in _source_paths():
        try:
            payload = (PROJECT_ROOT / relative).read_bytes()
        except OSError as error:
            _pending("SOURCE_CLOSURE", "Q2_LOCAL_IMPLEMENTATION_UNAVAILABLE", str(error))
        rows.append(
            {
                "relative_path": relative,
                "byte_count": len(payload),
                "raw_sha256": state.raw_sha256(payload),
            }
        )
    return rows


def _replay_source_closure(capsule: Mapping[str, Any]) -> None:
    rows = capsule["source_closure_rows"]
    if tuple(row["relative_path"] for row in rows) != _source_paths():
        _pending(
            "SOURCE_CLOSURE",
            "Q2_LOCAL_IMPLEMENTATION_UNAVAILABLE",
            "source path closure differs",
        )
    for row in rows:
        path = PROJECT_ROOT / row["relative_path"]
        try:
            payload = path.read_bytes()
        except OSError as error:
            _pending("SOURCE_CLOSURE", "Q2_LOCAL_IMPLEMENTATION_UNAVAILABLE", str(error))
        if len(payload) != row["byte_count"] or state.raw_sha256(payload) != row["raw_sha256"]:
            _pending(
                "SOURCE_CLOSURE",
                "Q2_LOCAL_IMPLEMENTATION_UNAVAILABLE",
                f"source row differs: {row['relative_path']}",
            )


def _selected_reader_total_bytes(root: Path, streams: Sequence[Any]) -> int:
    paths = [root / "endpoint-result.json", root / "closure-bundle.framed"]
    for stream in streams:
        signature_root = root / f"signature-{stream.input_signature_id:02d}"
        paths.extend(
            (
                signature_root / "signature-manifest.framed",
                signature_root / "chunk-manifests.framed",
            )
        )
        paths.extend(root / relative for relative in stream.framed_relative_paths)
    try:
        return sum(path.stat().st_size for path in paths)
    except OSError as error:
        _pending("Q1_ARCHIVE", "Q2_Q1_ARCHIVE_UNAVAILABLE", str(error))


def _target_blind_replay(
    contract: Any,
    preregistration: Mapping[str, Any],
) -> tuple[Any, tuple[Any, Any], dict[str, object]]:
    q1 = contract.value["q1_inputs"]
    try:
        terminal_payload = Path(q1["terminal_path"]).read_bytes()
        artifact_payload = Path(q1["completion_artifact_path"]).read_bytes()
        admitted = admission.admit_q1_completion_v1(terminal_payload, artifact_payload)
    except (OSError, ValueError) as error:
        _pending("Q1_PREDECESSOR", "Q2_Q1_PREDECESSOR_UNAVAILABLE", str(error))
    try:
        streams = archive_reader.read_q1_class_streams_v1(
            Path(q1["archive_root"]),
            admitted,
            expected_endpoint_kind=q1["selected_endpoint_kind"],
        )
    except ValueError as error:
        _pending("Q1_ARCHIVE", "Q2_Q1_ARCHIVE_UNAVAILABLE", str(error))
    odd, sink = streams
    reader_total = _selected_reader_total_bytes(Path(q1["archive_root"]), streams)
    envelope = preregistration["resource_envelope"]
    if (
        odd.framed_bytes + sink.framed_bytes
        > envelope["maximum_class_stream_framed_bytes"]
        or reader_total > envelope["maximum_selected_reader_total_bytes"]
        or len(odd.records) + len(sink.records) > envelope["maximum_role_rows"]
    ):
        _pending(
            "RESOURCE_ADMISSION",
            "Q2_RESOURCE_ADMISSION_UNAVAILABLE",
            "target-blind reader envelope exhausted",
        )
    receipt = {
        "endpoint_kind": "python",
        "odd_class_count": len(odd.records),
        "sink_class_count": len(sink.records),
        "odd_universe_root": "sha256:" + odd.universe_root.hex(),
        "sink_universe_root": "sha256:" + sink.universe_root.hex(),
        "odd_signature_manifest_root": "sha256:" + odd.signature_manifest_root.hex(),
        "sink_signature_manifest_root": "sha256:" + sink.signature_manifest_root.hex(),
        "odd_saturation_state_root": "sha256:" + odd.saturation_state_root.hex(),
        "sink_saturation_state_root": "sha256:" + sink.saturation_state_root.hex(),
        "odd_class_archive_root": "sha256:" + odd.class_archive_root.hex(),
        "sink_class_archive_root": "sha256:" + sink.class_archive_root.hex(),
        "odd_class_stream_framed_bytes": odd.framed_bytes,
        "sink_class_stream_framed_bytes": sink.framed_bytes,
        "total_class_stream_framed_bytes": odd.framed_bytes + sink.framed_bytes,
        "selected_reader_total_bytes": reader_total,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
    }
    state.validate_preflight_receipt(receipt)
    return admitted, streams, receipt


def _resource_gate(contract: Any, started_ns: int) -> None:
    envelope = contract.value["resource_envelope"]
    elapsed_ns = time.monotonic_ns() - started_ns
    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    artifact_parent = Path(contract.value["fixed_paths"]["artifact_path"]).parent
    try:
        free_bytes = shutil.disk_usage(artifact_parent).free
    except OSError as error:
        _pending("RESOURCE_ADMISSION", "Q2_RESOURCE_ADMISSION_UNAVAILABLE", str(error))
    if (
        elapsed_ns > envelope["maximum_wall_time_seconds"] * 1_000_000_000
        or peak_rss_bytes > envelope["maximum_peak_rss_bytes"]
        or free_bytes < envelope["minimum_fresh_persistent_storage_bytes"]
    ):
        _pending(
            "RESOURCE_ADMISSION",
            "Q2_RESOURCE_ADMISSION_UNAVAILABLE",
            f"elapsed_ns={elapsed_ns}; peak_rss_bytes={peak_rss_bytes}; free_bytes={free_bytes}",
        )


def _fresh_absence(contract: Any, *, allow_capsule: bool, allow_authorization: bool) -> None:
    fixed = contract.value["fixed_paths"]
    allowed = set()
    if allow_capsule:
        allowed.add("capsule_path")
    if allow_authorization:
        allowed.add("authorization_path")
    for key in ("capsule_path", "authorization_path", "consumption_path", "terminal_path", "artifact_path"):
        if key not in allowed and Path(fixed[key]).exists():
            _pending(
                "LOCAL_STATE",
                "Q2_LOCAL_PUBLICATION_UNAVAILABLE",
                f"fresh path is occupied: {fixed[key]}",
            )


def _read_capsule(
    contract: Any,
    preregistration: Mapping[str, Any],
    expected_digest: str | None = None,
) -> dict[str, Any] | None:
    path = Path(contract.value["fixed_paths"]["capsule_path"])
    if not path.exists():
        return None
    raw, value = _read_state(path, "Q2 capsule")
    try:
        checked = state.validate_capsule(value, contract, preregistration)
    except ValueError as error:
        _pending("CAPSULE", "Q2_LOCAL_STATE_UNAVAILABLE", str(error))
    if raw != state.canonical_json_bytes(checked) or (
        expected_digest is not None and checked["capsule_digest"] != expected_digest
    ):
        _pending("CAPSULE", "Q2_LOCAL_STATE_UNAVAILABLE", "capsule replay differs")
    return checked


def _read_authorization(
    contract: Any,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = Path(contract.value["fixed_paths"]["authorization_path"])
    if not path.exists():
        return None
    raw, value = _read_state(path, "Q2 authorization")
    try:
        checked = state.validate_authorization(value, contract, preregistration, capsule)
    except ValueError as error:
        _pending("AUTHORIZATION", "Q2_LOCAL_STATE_UNAVAILABLE", str(error))
    if raw != state.canonical_json_bytes(checked):
        _pending("AUTHORIZATION", "Q2_LOCAL_STATE_UNAVAILABLE", "authorization replay differs")
    return checked


def _read_consumption(
    contract: Any,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = Path(contract.value["fixed_paths"]["consumption_path"])
    if not path.exists():
        return None
    try:
        raw, value = _read_state(path, "Q2 consumption")
    except ControllerPending as error:
        _terminal("Q2_CONSUMPTION_REPLAY_DIFFERS", str(error))
    try:
        checked = state.validate_consumption(
            value, contract, preregistration, capsule, authorization
        )
    except ValueError as error:
        _terminal("Q2_CONSUMPTION_REPLAY_DIFFERS", str(error))
    if raw != state.canonical_json_bytes(checked):
        _terminal("Q2_CONSUMPTION_REPLAY_DIFFERS", "consumption bytes differ")
    return checked


def _read_artifact(
    contract: Any,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
) -> tuple[bytes, Any] | None:
    path = Path(contract.value["fixed_paths"]["artifact_path"])
    if not path.exists():
        return None
    try:
        payload = path.read_bytes()
        artifact = membership.decode_q2_membership_artifact_v1(payload)
        state.validate_artifact_binding(
            artifact,
            contract,
            preregistration,
            capsule,
            authorization,
            consumption,
        )
    except (OSError, ValueError) as error:
        _terminal("Q2_SEALED_ARTIFACT_REPLAY_DIFFERS", str(error))
    return payload, artifact


def _read_terminal(
    contract: Any,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = Path(contract.value["fixed_paths"]["terminal_path"])
    if not path.exists():
        return None
    try:
        raw, value = _read_state(path, "Q2 terminal")
    except ControllerPending as error:
        _terminal("Q2_TERMINAL_REPLAY_DIFFERS", str(error))
    artifact_payload = None
    if value.get("status") in (membership.PASS_STATUS, membership.COMPLETE_FAIL_STATUS):
        artifact_path = Path(contract.value["fixed_paths"]["artifact_path"])
        try:
            artifact_payload = artifact_path.read_bytes()
        except OSError as error:
            _terminal("Q2_TERMINAL_ARTIFACT_UNAVAILABLE", str(error))
    else:
        artifact_path = Path(contract.value["fixed_paths"]["artifact_path"])
        if artifact_path.exists():
            try:
                artifact_pair = _read_artifact(
                    contract,
                    preregistration,
                    capsule,
                    authorization,
                    consumption,
                )
            except ControllerTerminal:
                artifact_pair = None
            if artifact_pair is not None:
                _terminal(
                    "Q2_INVALIDATED_TERMINAL_VALID_ARTIFACT_COEXISTS",
                    contract.value["fixed_paths"]["artifact_path"],
                )
    try:
        checked = state.validate_terminal(
            value,
            contract,
            preregistration,
            capsule,
            authorization,
            consumption,
            artifact_payload=artifact_payload,
        )
    except ValueError as error:
        _terminal("Q2_TERMINAL_REPLAY_DIFFERS", str(error))
    if raw != state.canonical_json_bytes(checked):
        _terminal("Q2_TERMINAL_REPLAY_DIFFERS", "terminal bytes differ")
    return checked


def _publish_pending(
    contract: Any,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    error: ControllerPending,
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pending = state.build_pending(
        contract,
        preregistration,
        capsule,
        phase=error.phase,
        condition=error.condition,
        detail=error.detail,
        authorization=authorization,
    )
    pending_path = Path(contract.value["fixed_paths"]["pending_path"])
    try:
        _replace_json(pending_path, pending, pending["pending_digest"])
    except ControllerPending as publication_error:
        return {
            "schema_version": "hegel-phase3a-q2-pending-publication-unavailable/1",
            "status": PENDING_PUBLICATION_UNAVAILABLE,
            "capsule_digest": capsule["capsule_digest"],
            "authorization_digest": (
                authorization["authorization_digest"]
                if authorization is not None
                else None
            ),
            "authorization_consumption_count": 0,
            "logical_campaign_count": 0,
            "target_truth_accessed": False,
            "role_evaluation_performed": False,
            "original_phase": error.phase,
            "original_condition": error.condition,
            "original_detail": error.detail,
            "publication_detail": publication_error.detail,
            "same_capsule_digest_reentry_allowed": True,
        }
    try:
        _append_trajectory(
            contract,
            "PENDING",
            error.phase,
            capsule_digest=capsule["capsule_digest"],
            detail={"condition": error.condition, "detail": error.detail},
        )
    except ControllerPending as trajectory_error:
        pending = state.build_pending(
            contract,
            preregistration,
            capsule,
            phase="TRAJECTORY",
            condition=trajectory_error.condition,
            detail=(
                f"{trajectory_error.detail}; while recording "
                f"{error.phase}/{error.condition}: {error.detail}"
            ),
            authorization=authorization,
        )
        try:
            _replace_json(pending_path, pending, pending["pending_digest"])
        except ControllerPending as publication_error:
            return {
                "schema_version": (
                    "hegel-phase3a-q2-pending-publication-unavailable/1"
                ),
                "status": PENDING_PUBLICATION_UNAVAILABLE,
                "capsule_digest": capsule["capsule_digest"],
                "authorization_digest": (
                    authorization["authorization_digest"]
                    if authorization is not None
                    else None
                ),
                "authorization_consumption_count": 0,
                "logical_campaign_count": 0,
                "target_truth_accessed": False,
                "role_evaluation_performed": False,
                "original_phase": "TRAJECTORY",
                "original_condition": trajectory_error.condition,
                "original_detail": trajectory_error.detail,
                "publication_detail": publication_error.detail,
                "same_capsule_digest_reentry_allowed": True,
            }
    return pending


def _unbound_pending(contract_path: Path, error: Exception) -> dict[str, Any]:
    return {
        "schema_version": "hegel-phase3a-q2-unbound-pending/1",
        "status": "Q2_ROLE_MEMBERSHIP_UNBOUND_PENDING_NOT_CONSUMED",
        "contract_path": str(contract_path),
        "authorization_consumption_count": 0,
        "logical_campaign_count": 0,
        "target_truth_accessed": False,
        "role_evaluation_performed": False,
        "detail": str(error),
    }


def _not_authorized(capsule: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "hegel-phase3a-q2-not-authorized/1",
        "status": NOT_AUTHORIZED,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_consumption_count": 0,
        "logical_campaign_count": 0,
        "q1_state": "COMPLETE",
        "q2_state": "NOT_RUN",
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
        "internal_popen_count": 0,
        "actor_start_count": 0,
        "docker_invocation_count": 0,
    }


def _partial_finalization_result(
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    error: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": "hegel-phase3a-q2-partial-finalization/1",
        "status": PARTIAL_FINALIZATION,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "consumption_record_digest": consumption["record_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "exact_artifact_to_terminal_reentry_allowed": True,
        "target_or_role_reentry_allowed": False,
        "detail": str(error),
    }


def _empty_action_audit() -> dict[str, dict[str, int | None]]:
    return {
        "target_open_count_by_role": {
            "IN_LANGUAGE_NULL": 0,
            "OUTSIDE_TARGET": 0,
        },
        "role_evaluation_count_by_role": {
            "IN_LANGUAGE_NULL": 0,
            "OUTSIDE_TARGET": 0,
        },
    }


def _trajectory_action_audit(
    contract: Any,
    capsule_digest: str,
) -> dict[str, dict[str, int | None]]:
    audit = _empty_action_audit()
    path = Path(contract.value["fixed_paths"]["trajectory_path"])
    try:
        lines = path.read_bytes().splitlines(keepends=True)
        for payload in lines:
            row = state.strict_json_bytes(payload, "Q2 trajectory event")
            if (
                row.get("schema_version")
                != "hegel-phase3a-q2-trajectory-event/1"
                or row.get("capsule_digest") != capsule_digest
            ):
                continue
            role = row.get("phase")
            if role not in ("IN_LANGUAGE_NULL", "OUTSIDE_TARGET"):
                continue
            if row.get("event") == "TARGET_OPEN_INTENT":
                audit["target_open_count_by_role"][role] = 1
            elif row.get("event") == "ROLE_EVALUATION_INTENT":
                audit["role_evaluation_count_by_role"][role] = 1
    except (OSError, ValueError):
        return {
            "target_open_count_by_role": {
                "IN_LANGUAGE_NULL": None,
                "OUTSIDE_TARGET": None,
            },
            "role_evaluation_count_by_role": {
                "IN_LANGUAGE_NULL": None,
                "OUTSIDE_TARGET": None,
            },
        }
    return audit


def _unsealed_terminal_result(
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    error: Exception,
) -> dict[str, Any]:
    return {
        "schema_version": "hegel-phase3a-q2-unsealed-terminal/1",
        "status": UNSEALED_TERMINAL,
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "consumption_record_digest": consumption["record_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "target_or_role_reentry_allowed": False,
        "terminal_no_resume": True,
        "detail": str(error),
    }


def _consumed_state_unreadable_result(
    capsule_digest: str,
    error: Exception,
    *,
    capsule: Mapping[str, Any] | None = None,
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "hegel-phase3a-q2-consumed-state-unreadable/1",
        "status": UNSEALED_TERMINAL,
        "capsule_digest": (
            capsule["capsule_digest"] if capsule is not None else capsule_digest
        ),
        "authorization_digest": (
            authorization["authorization_digest"]
            if authorization is not None
            else None
        ),
        "consumption_record_digest": None,
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "target_or_role_reentry_allowed": False,
        "terminal_no_resume": True,
        "detail": str(error),
    }


def preflight_v1(
    contract_path: Path,
    run_id: str,
    created_at_unix_seconds: int,
) -> dict[str, Any]:
    contract = None
    capsule = None
    started_ns = time.monotonic_ns()
    try:
        contract = _load_contract(contract_path)
        _ensure_roots(contract)
        with _campaign_lock(contract):
            _, preregistration = _load_preregistration(contract)
            existing = _read_capsule(contract, preregistration)
            _fresh_absence(
                contract,
                allow_capsule=existing is not None,
                allow_authorization=False,
            )
            source_rows = _source_closure_rows()
            admitted, streams, receipt = _target_blind_replay(contract, preregistration)
            del admitted, streams
            _resource_gate(contract, started_ns)
            inputs = {
                "schema_version": state.CAPSULE_INPUTS_SCHEMA,
                "source_closure_rows": source_rows,
                "source_closure_root": state.source_closure_root(source_rows),
                "run_id": run_id,
                "created_at_unix_seconds": created_at_unix_seconds,
                "preflight_receipt": receipt,
            }
            capsule = state.build_capsule(contract, preregistration, inputs)
            if existing is not None and existing != capsule:
                _pending("CAPSULE", "Q2_LOCAL_STATE_UNAVAILABLE", "published capsule inputs differ")
            _publish_json(
                Path(contract.value["fixed_paths"]["capsule_path"]),
                capsule,
                capsule["capsule_digest"],
            )
            _append_trajectory(
                contract,
                "CAPSULE_PUBLISHED",
                "PREFLIGHT",
                capsule_digest=capsule["capsule_digest"],
                detail={
                    "target_truth_accessed": False,
                    "role_evaluation_performed": False,
                    "selected_reader_total_bytes": receipt["selected_reader_total_bytes"],
                },
            )
            replayed = _read_capsule(
                contract, preregistration, capsule["capsule_digest"]
            )
            assert replayed is not None
            return replayed
    except ControllerPending as error:
        if contract is not None and capsule is not None:
            return _publish_pending(contract, preregistration, capsule, error)
        return _unbound_pending(contract_path, error)
    except ControllerTerminal as error:
        return _unbound_pending(contract_path, error)


def record_authorization_v1(
    contract_path: Path,
    capsule_digest: str,
) -> dict[str, Any]:
    contract = None
    capsule = None
    authorization = None
    started_ns = time.monotonic_ns()
    try:
        contract = _load_contract(contract_path)
        _ensure_roots(contract)
        with _campaign_lock(contract):
            _, preregistration = _load_preregistration(contract)
            capsule = _read_capsule(contract, preregistration, capsule_digest)
            if capsule is None:
                _pending("AUTHORIZATION", "Q2_LOCAL_STATE_UNAVAILABLE", "capsule is absent")
            existing = _read_authorization(contract, preregistration, capsule)
            if existing is not None:
                return existing
            _fresh_absence(contract, allow_capsule=True, allow_authorization=False)
            _replay_source_closure(capsule)
            _, streams, receipt = _target_blind_replay(contract, preregistration)
            del streams
            if receipt != capsule["preflight_receipt"]:
                _pending("Q1_ARCHIVE", "Q2_Q1_ARCHIVE_UNAVAILABLE", "preflight receipt differs")
            _resource_gate(contract, started_ns)
            authorization = state.build_authorization(contract, preregistration, capsule)
            _publish_json(
                Path(contract.value["fixed_paths"]["authorization_path"]),
                authorization,
                authorization["authorization_digest"],
            )
            _append_trajectory(
                contract,
                "AUTHORIZATION_RECORDED",
                "RECORD_AUTHORIZATION",
                capsule_digest=capsule_digest,
            )
            replayed = _read_authorization(contract, preregistration, capsule)
            assert replayed is not None
            return replayed
    except ControllerPending as error:
        if contract is not None and capsule is not None:
            return _publish_pending(
                contract, preregistration, capsule, error, authorization
            )
        return _unbound_pending(contract_path, error)
    except ControllerTerminal as error:
        return _unbound_pending(contract_path, error)


def _execute_measurement(
    contract: Any,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    admitted: Any,
    streams: tuple[Any, Any],
    campaign_started_ns: int,
    action_audit: dict[str, dict[str, int | None]],
) -> tuple[bytes, Any]:
    rows_module = importlib.import_module("hegel_machine.phase3_m25_rows_v1")
    results = []
    for role_name, generator_name, stream, commitment in (
        (
            "OUTSIDE_TARGET",
            "generate_odd_role_rows_v1",
            streams[0],
            admission.Q2_ROLE_COMMITMENTS[0],
        ),
        (
            "IN_LANGUAGE_NULL",
            "generate_sink_role_rows_v1",
            streams[1],
            admission.Q2_ROLE_COMMITMENTS[1],
        ),
    ):
        _append_trajectory(
            contract,
            "TARGET_OPEN_INTENT",
            role_name,
            capsule_digest=capsule["capsule_digest"],
            detail={"conservative_action_count": 1},
        )
        action_audit["target_open_count_by_role"][role_name] = 1
        typed_rows = getattr(rows_module, generator_name)()
        target_bits = membership.verified_role_target_bits_v1(
            typed_rows, commitment
        )
        _append_trajectory(
            contract,
            "TARGET_OPEN_RETURN",
            role_name,
            capsule_digest=capsule["capsule_digest"],
            detail={"row_count": len(target_bits)},
        )
        _append_trajectory(
            contract,
            "ROLE_EVALUATION_INTENT",
            role_name,
            capsule_digest=capsule["capsule_digest"],
            detail={"conservative_action_count": 1},
        )
        action_audit["role_evaluation_count_by_role"][role_name] = 1
        result = membership.evaluate_role_membership_v1(
            stream, commitment, target_bits
        )
        results.append(result)
        _append_trajectory(
            contract,
            "ROLE_EVALUATION_RETURN",
            role_name,
            capsule_digest=capsule["capsule_digest"],
            detail={
                "evaluated_class_count": result.evaluated_class_count,
                "match_count": result.match_count,
                "membership_verdict": result.membership_verdict,
            },
        )
    elapsed_ns = time.monotonic_ns() - campaign_started_ns
    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    artifact = membership.build_q2_membership_artifact_v1(
        bytes.fromhex(capsule["capsule_digest"][7:]),
        bytes.fromhex(authorization["authorization_digest"][7:]),
        bytes.fromhex(consumption["record_digest"][7:]),
        admitted,
        results[0],
        results[1],
        observed_wall_time_ns=elapsed_ns,
        observed_peak_rss_bytes=peak_rss_bytes,
    )
    state.validate_artifact_binding(
        artifact,
        contract,
        preregistration,
        capsule,
        authorization,
        consumption,
    )
    return artifact.canonical_bytes, artifact


def _publish_complete(
    contract: Any,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    artifact_payload: bytes,
    artifact: Any,
) -> dict[str, Any]:
    fixed = contract.value["fixed_paths"]
    _publish_bytes(
        Path(fixed["artifact_path"]),
        artifact_payload,
        capsule["capsule_digest"],
    )
    _append_trajectory_after_state(
        contract,
        "ARTIFACT_PUBLISHED",
        "FINALIZATION",
        capsule_digest=capsule["capsule_digest"],
        detail={"status": artifact.status, "artifact_bytes": len(artifact_payload)},
    )
    terminal = state.build_complete_terminal(
        contract,
        preregistration,
        capsule,
        authorization,
        consumption,
        artifact,
    )
    _publish_json(
        Path(fixed["terminal_path"]),
        terminal,
        terminal["record_digest"],
    )
    _append_trajectory_after_state(
        contract,
        "TERMINAL_PUBLISHED",
        "FINALIZATION",
        capsule_digest=capsule["capsule_digest"],
        detail={"status": terminal["status"]},
    )
    replayed = _read_terminal(
        contract, preregistration, capsule, authorization, consumption
    )
    assert replayed is not None
    return replayed


def _publish_invalidated(
    contract: Any,
    preregistration: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    error: Exception,
    *,
    action_audit: Mapping[str, Mapping[str, int | None]] | None = None,
) -> dict[str, Any]:
    if isinstance(error, ControllerTerminal):
        code, detail = error.code, error.detail
    else:
        code, detail = "Q2_POSTCONSUMPTION_OUTCOME_UNKNOWN", str(error)
    phase = (
        error.phase
        if isinstance(error, ControllerPending)
        else "POSTCONSUMPTION"
    )
    if action_audit is None:
        action_audit = _trajectory_action_audit(
            contract, capsule["capsule_digest"]
        )
    artifact_path = Path(contract.value["fixed_paths"]["artifact_path"])
    if artifact_path.exists():
        try:
            artifact_pair = _read_artifact(
                contract,
                preregistration,
                capsule,
                authorization,
                consumption,
            )
        except ControllerTerminal:
            artifact_pair = None
        if artifact_pair is not None:
            return _partial_finalization_result(
                capsule, authorization, consumption, error
            )
    terminal = state.build_invalidated_terminal(
        contract,
        preregistration,
        capsule,
        authorization,
        consumption,
        failure_phase=phase,
        failure_code=code,
        failure_detail=detail,
        target_open_count_by_role=action_audit["target_open_count_by_role"],
        role_evaluation_count_by_role=action_audit[
            "role_evaluation_count_by_role"
        ],
    )
    try:
        _publish_json(
            Path(contract.value["fixed_paths"]["terminal_path"]),
            terminal,
            terminal["record_digest"],
        )
        _append_trajectory_after_state(
            contract,
            "INVALIDATED_TERMINAL_PUBLISHED",
            "FINALIZATION",
            capsule_digest=capsule["capsule_digest"],
            detail={"failure_code": code, "failure_detail": detail},
        )
        replayed = _read_terminal(
            contract, preregistration, capsule, authorization, consumption
        )
        assert replayed is not None
        return replayed
    except Exception as publication_error:
        return _unsealed_terminal_result(
            capsule, authorization, consumption, publication_error
        )


def run_v1(contract_path: Path, capsule_digest: str) -> dict[str, Any]:
    contract = None
    capsule = None
    authorization = None
    consumption = None
    consumption_path_observed = False
    campaign_started_ns = time.monotonic_ns()
    try:
        contract = _load_contract(contract_path)
        consumption_path_observed = Path(
            contract.value["fixed_paths"]["consumption_path"]
        ).exists()
        _ensure_roots(contract)
        with _campaign_lock(contract):
            _, preregistration = _load_preregistration(contract)
            capsule = _read_capsule(contract, preregistration, capsule_digest)
            if capsule is None:
                _pending("RUN", "Q2_LOCAL_STATE_UNAVAILABLE", "capsule is absent")
            _replay_source_closure(capsule)
            authorization = _read_authorization(contract, preregistration, capsule)
            if authorization is None:
                if consumption_path_observed:
                    return _consumed_state_unreadable_result(
                        capsule_digest,
                        ControllerTerminal(
                            "Q2_CONSUMED_STATE_AUTHORIZATION_ABSENT",
                            contract.value["fixed_paths"]["authorization_path"],
                        ),
                        capsule=capsule,
                    )
                return _not_authorized(capsule)
            consumption = _read_consumption(
                contract, preregistration, capsule, authorization
            )
            if consumption is not None:
                terminal = _read_terminal(
                    contract, preregistration, capsule, authorization, consumption
                )
                if terminal is not None:
                    return terminal
                try:
                    artifact_pair = _read_artifact(
                        contract,
                        preregistration,
                        capsule,
                        authorization,
                        consumption,
                    )
                    if artifact_pair is None:
                        _terminal(
                            "Q2_POSTCONSUMPTION_ARTIFACT_ABSENT",
                            "consumption exists without a sealed artifact",
                        )
                    return _publish_complete(
                        contract,
                        preregistration,
                        capsule,
                        authorization,
                        consumption,
                        artifact_pair[0],
                        artifact_pair[1],
                    )
                except ControllerPending as error:
                    return _partial_finalization_result(
                        capsule, authorization, consumption, error
                    )
                except Exception as error:
                    return _publish_invalidated(
                        contract,
                        preregistration,
                        capsule,
                        authorization,
                        consumption,
                        error,
                    )

            _fresh_absence(contract, allow_capsule=True, allow_authorization=True)
            admitted, streams, receipt = _target_blind_replay(
                contract, preregistration
            )
            if receipt != capsule["preflight_receipt"]:
                _pending("Q1_ARCHIVE", "Q2_Q1_ARCHIVE_UNAVAILABLE", "preflight receipt differs")
            _resource_gate(contract, campaign_started_ns)
            consumption_candidate = state.build_consumption(
                contract, preregistration, capsule, authorization
            )
            consumption_path = Path(
                contract.value["fixed_paths"]["consumption_path"]
            )
            try:
                _publish_json(
                    consumption_path,
                    consumption_candidate,
                    consumption_candidate["record_digest"],
                )
                replayed_consumption = _read_consumption(
                    contract, preregistration, capsule, authorization
                )
                if replayed_consumption is None:
                    _terminal(
                        "Q2_CONSUMPTION_REPLAY_ABSENT_AFTER_PUBLICATION",
                        str(consumption_path),
                    )
            except (ControllerPending, ControllerTerminal) as error:
                if not consumption_path.exists():
                    raise
                try:
                    replayed_consumption = _read_consumption(
                        contract, preregistration, capsule, authorization
                    )
                except (ControllerPending, ControllerTerminal) as replay_error:
                    return _consumed_state_unreadable_result(
                        capsule_digest,
                        replay_error,
                        capsule=capsule,
                        authorization=authorization,
                    )
                if replayed_consumption is None:
                    return _consumed_state_unreadable_result(
                        capsule_digest,
                        error,
                        capsule=capsule,
                        authorization=authorization,
                    )
                consumption = replayed_consumption
                return _publish_invalidated(
                    contract,
                    preregistration,
                    capsule,
                    authorization,
                    consumption,
                    error,
                    action_audit=_empty_action_audit(),
                )
            consumption = replayed_consumption
            action_audit = _empty_action_audit()
            try:
                _append_trajectory(
                    contract,
                    "AUTHORIZATION_CONSUMED",
                    "CONSUMPTION",
                    capsule_digest=capsule["capsule_digest"],
                    detail={
                        "target_truth_accessed_before_consumption": False,
                        "role_evaluation_performed_before_consumption": False,
                    },
                )
                artifact_payload, artifact = _execute_measurement(
                    contract,
                    preregistration,
                    capsule,
                    authorization,
                    consumption,
                    admitted,
                    streams,
                    campaign_started_ns,
                    action_audit,
                )
            except Exception as error:
                return _publish_invalidated(
                    contract,
                    preregistration,
                    capsule,
                    authorization,
                    consumption,
                    error,
                    action_audit=action_audit,
                )
            try:
                return _publish_complete(
                    contract,
                    preregistration,
                    capsule,
                    authorization,
                    consumption,
                    artifact_payload,
                    artifact,
                )
            except ControllerPending as error:
                return _publish_invalidated(
                    contract,
                    preregistration,
                    capsule,
                    authorization,
                    consumption,
                    error,
                    action_audit=action_audit,
                )
            except Exception as error:
                return _publish_invalidated(
                    contract,
                    preregistration,
                    capsule,
                    authorization,
                    consumption,
                    error,
                    action_audit=action_audit,
                )
    except ControllerPending as error:
        if consumption_path_observed and consumption is None:
            return _consumed_state_unreadable_result(
                capsule_digest,
                error,
                capsule=capsule,
                authorization=authorization,
            )
        if contract is not None and capsule is not None and consumption is None:
            return _publish_pending(
                contract, preregistration, capsule, error, authorization
            )
        if capsule is not None and authorization is not None and consumption is not None:
            return _partial_finalization_result(
                capsule, authorization, consumption, error
            )
        return _unbound_pending(contract_path, error)
    except ControllerTerminal as error:
        if consumption_path_observed and consumption is None:
            return _consumed_state_unreadable_result(
                capsule_digest,
                error,
                capsule=capsule,
                authorization=authorization,
            )
        if (
            contract is not None
            and capsule is not None
            and authorization is not None
            and consumption is not None
        ):
            return _publish_invalidated(
                contract,
                preregistration,
                capsule,
                authorization,
                consumption,
                error,
            )
        return _unbound_pending(contract_path, error)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--record-authorization", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--run-id")
    parser.add_argument("--created-at-unix-seconds", type=int)
    parser.add_argument("--capsule-digest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.preflight:
        if (
            args.run_id is None
            or args.created_at_unix_seconds is None
            or args.capsule_digest is not None
        ):
            parser.error(
                "--preflight requires --run-id and --created-at-unix-seconds "
                "and forbids --capsule-digest"
            )
        result = preflight_v1(
            args.contract, args.run_id, args.created_at_unix_seconds
        )
    elif args.record_authorization:
        if (
            args.capsule_digest is None
            or args.run_id is not None
            or args.created_at_unix_seconds is not None
        ):
            parser.error(
                "--record-authorization requires --capsule-digest and forbids "
                "preflight identity arguments"
            )
        result = record_authorization_v1(args.contract, args.capsule_digest)
    else:
        if (
            args.capsule_digest is None
            or args.run_id is not None
            or args.created_at_unix_seconds is not None
        ):
            parser.error(
                "--run requires --capsule-digest and forbids preflight identity arguments"
            )
        result = run_v1(args.contract, args.capsule_digest)
    sys.stdout.buffer.write(state.canonical_json_bytes(result))
    status = result["status"]
    if status in (
        state.PENDING_NOT_CONSUMED,
        "Q2_ROLE_MEMBERSHIP_UNBOUND_PENDING_NOT_CONSUMED",
        PENDING_PUBLICATION_UNAVAILABLE,
        PARTIAL_FINALIZATION,
    ):
        return PENDING_EXIT_CODE
    if status in (state.INVALIDATED_TERMINAL_NO_RESUME, UNSEALED_TERMINAL):
        return 2
    if status == NOT_AUTHORIZED:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
