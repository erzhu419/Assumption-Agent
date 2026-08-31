#!/usr/bin/env python3
"""Zero-Popen controller for the Q1 completion-only v8c successor.

The controller never resumes or launches a v7r1 actor.  It verifies the exact
terminal v7r1 lineage, replays its sealed outputs read-only, and publishes a
new outer authority chain for the corrected completion receipt.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import importlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
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

state = importlib.import_module(
    "hegel_machine.phase3_q1_completion_replay_successor_v1"
)
adapter = importlib.import_module(
    "hegel_machine.phase3_q1_start_completion_adapter_v1"
)
completion = importlib.import_module(
    "hegel_machine.phase3_q1_start_completion_wire_v1"
)
v7_campaign = importlib.import_module("hegel_machine.phase3_q1_start_campaign_v7")
v7_successor = importlib.import_module(
    "hegel_machine.phase3_q1_prestart_successor_candidate_v7"
)

DEFAULT_CONTRACT: Final = PROJECT_ROOT / "config/phase3_q1_completion_successor_v8c.json"
V7_SUPERVISOR_RELATIVE: Final = "tools/phase3_q1_start_supervisor_v7.py"
PENDING_EXIT_CODE: Final = 75
NOT_AUTHORIZED_EXIT_CODE: Final = 4
TERMINAL_EXIT_CODE: Final = 2
UNSEALED_TERMINAL_NO_RESUME: Final = (
    "Q1_COMPLETION_REPLAY_V8C_UNSEALED_TERMINAL_NO_RESUME"
)
_ACTORS: Final = ("python", "rust", "host")
_EXTRA_SOURCE_PATHS: Final = (
    "config/phase3_q1_completion_successor_v8c.json",
    "docs/Hegel_Machine_Phase3A_Q1_Completion_Successor_v8c.md",
    "src/hegel_machine/phase3_q1_completion_replay_successor_v1.py",
    "src/hegel_machine/phase3_q1_start_completion_adapter_v1.py",
    "tools/phase3_q1_completion_successor_v8c.py",
)


class CompletionReplayControllerError(RuntimeError):
    """Base class for classified controller failures."""


class ControllerPending(CompletionReplayControllerError):
    def __init__(self, phase: str, condition: str, detail: str) -> None:
        super().__init__(f"{phase}/{condition}: {detail}")
        self.phase = phase
        self.condition = condition
        self.detail = detail


class ControllerTerminal(CompletionReplayControllerError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _terminal(code: str, detail: str) -> NoReturn:
    raise ControllerTerminal(code, detail)


@dataclass(frozen=True, slots=True)
class ReplayMaterialV8c:
    v7_contract: Any
    v7_capsule: Mapping[str, Any]
    v7_authorization: Mapping[str, Any]
    v7_consumption: Mapping[str, Any]
    v7_terminal: Mapping[str, Any]
    run_spec: Mapping[str, Any]
    supervisor_outcome: Mapping[str, Any]
    completion_artifact: bytes
    completion_receipt: Any
    q1_output_slots: tuple[str, ...]


def _load_supervisor_module() -> Any:
    path = PROJECT_ROOT / V7_SUPERVISOR_RELATIVE
    spec = importlib.util.spec_from_file_location("hegel_q1_start_supervisor_v7c", path)
    if spec is None or spec.loader is None:
        raise ControllerPending(
            "LOCAL_DEPENDENCY",
            "V7_SUPERVISOR_HELPERS_UNAVAILABLE",
            str(path),
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(spec.name, None)
        raise ControllerPending(
            "LOCAL_DEPENDENCY",
            "V7_SUPERVISOR_HELPERS_UNAVAILABLE",
            str(error),
        ) from error
    return module


def _read_bytes(path: Path, label: str, maximum: int = state.MAX_JSON_BYTES) -> bytes:
    try:
        info = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise OSError("not one regular non-symlink file")
        if info.st_size > maximum:
            raise OSError(f"exceeds {maximum}-byte bound")
        payload = path.read_bytes()
    except OSError as error:
        raise ControllerPending(
            "V7R1_READ_ONLY_REPLAY",
            "BOUND_EVIDENCE_UNAVAILABLE",
            f"{label}: {error}",
        ) from error
    return payload


def _read_bound(
    path: Path,
    expected_raw_sha256: str,
    label: str,
    maximum: int = state.MAX_JSON_BYTES,
) -> bytes:
    payload = _read_bytes(path, label, maximum)
    if state.raw_sha256(payload) != expected_raw_sha256:
        raise ControllerPending(
            "V7R1_READ_ONLY_REPLAY",
            "BOUND_EVIDENCE_RAW_IDENTITY_DIFFERS",
            label,
        )
    return payload


def _strict_v7(payload: bytes, label: str) -> dict[str, Any]:
    try:
        return v7_campaign.strict_json_bytes_v7(payload, label)
    except Exception as error:
        raise ControllerPending(
            "V7R1_READ_ONLY_REPLAY",
            "BOUND_EVIDENCE_FORMAT_DIFFERS",
            f"{label}: {error}",
        ) from error


def _source_rows(contract: Any) -> list[dict[str, Any]]:
    project_root = Path(contract.value["fixed_paths"]["project_root"])
    predecessor = contract.value["predecessor_v7r1"]
    v7_contract_raw = _read_bound(
        Path(predecessor["contract_path"]),
        predecessor["contract_raw_sha256"],
        "v7r1 contract",
    )
    v7_contract_value = _strict_v7(v7_contract_raw, "v7r1 contract")
    paths = v7_contract_value.get("controller_source_closure_relative_paths")
    if type(paths) is not list or any(type(item) is not str for item in paths):
        raise ControllerPending(
            "SOURCE_CLOSURE",
            "V7R1_SOURCE_REGISTRY_DIFFERS",
            str(predecessor["contract_path"]),
        )
    relative_paths = sorted(set((*paths, *_EXTRA_SOURCE_PATHS)))
    rows: list[dict[str, Any]] = []
    for relative in relative_paths:
        path = project_root / relative
        payload = _read_bytes(path, f"source closure {relative}")
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ControllerPending(
                "SOURCE_CLOSURE",
                "SOURCE_PATH_UNAVAILABLE",
                f"{relative}: {error}",
            ) from error
        if resolved != path:
            raise ControllerPending(
                "SOURCE_CLOSURE",
                "SOURCE_PATH_IDENTITY_DIFFERS",
                relative,
            )
        rows.append(
            {
                "relative_path": relative,
                "byte_count": len(payload),
                "raw_sha256": state.raw_sha256(payload),
            }
        )
    state.source_closure_root(rows)
    return rows


def _replay_source_closure(contract: Any, capsule: Mapping[str, Any]) -> None:
    observed = _source_rows(contract)
    if (
        observed != capsule["source_closure_rows"]
        or state.source_closure_root(observed) != capsule["source_closure_root"]
    ):
        raise ControllerPending(
            "SOURCE_CLOSURE",
            "CAPSULE_SOURCE_CLOSURE_DIFFERS",
            str(contract.path),
        )


def _validate_intent_and_complete(
    predecessor: Mapping[str, Any],
    stage: str,
    intent: Mapping[str, Any],
    complete: Mapping[str, Any],
) -> None:
    expected_output = str(Path(predecessor["run_root"]) / f"{stage}-output")
    expected_scratch = str(Path(predecessor["run_root"]) / f"{stage}-scratch")
    if (
        intent.get("status") != "DURABLE_INTENT_FSYNCED_BEFORE_ACTOR_POPEN"
        or intent.get("stage") != stage
        or intent.get("capsule_digest") != predecessor["capsule_digest"]
        or intent.get("start_authorization_digest")
        != predecessor["authorization_digest"]
        or complete.get("status")
        != "CLEAN_CHECKPOINT_COMPLETE_READONLY_REPLAY_ONLY"
        or complete.get("stage") != stage
        or complete.get("capsule_digest") != predecessor["capsule_digest"]
        or complete.get("start_authorization_digest")
        != predecessor["authorization_digest"]
        or complete.get("actor_popen_count") != 1
        or complete.get("intent_path") != predecessor[f"{stage}_intent_path"]
        or complete.get("output_path") != expected_output
        or complete.get("scratch_path") != expected_scratch
        or complete.get("scratch_absent_after_cleanup") is not True
        or type(complete.get("output_tree_bytes")) is not int
        or complete.get("output_tree_bytes", -1) < 0
    ):
        raise ControllerPending(
            "V7R1_READ_ONLY_REPLAY",
            "ACTOR_CHECKPOINT_BINDING_DIFFERS",
            stage,
        )
    if Path(expected_scratch).exists():
        raise ControllerPending(
            "V7R1_READ_ONLY_REPLAY",
            "ACTOR_SCRATCH_ABSENCE_DIFFERS",
            expected_scratch,
        )


def _receipt_slots(receipt: Any) -> tuple[str, ...]:
    slots = tuple("sha256:" + row[2].hex() for row in receipt.output_slots)
    if len(slots) != 8 or slots[-1] != "sha256:" + receipt.receipt_root.hex():
        _terminal(
            "Q1_COMPLETION_REPLAY_RECEIPT_PROJECTION_DIFFERS",
            "completion receipt output slots differ",
        )
    return slots


def _replay_v7r1(contract: Any) -> ReplayMaterialV8c:
    predecessor = contract.value["predecessor_v7r1"]
    try:
        v7_contract = v7_campaign.load_contract_v7(
            Path(predecessor["contract_path"])
        )
    except Exception as error:
        raise ControllerPending(
            "V7R1_READ_ONLY_REPLAY",
            "V7R1_CONTRACT_REPLAY_DIFFERS",
            str(error),
        ) from error
    if v7_contract.raw_sha256 != predecessor["contract_raw_sha256"]:
        raise ControllerPending(
            "V7R1_READ_ONLY_REPLAY",
            "V7R1_CONTRACT_REPLAY_DIFFERS",
            str(v7_contract.path),
        )

    capsule_raw = _read_bound(
        Path(predecessor["capsule_path"]),
        predecessor["capsule_raw_sha256"],
        "v7r1 capsule",
    )
    authorization_raw = _read_bound(
        Path(predecessor["authorization_path"]),
        predecessor["authorization_raw_sha256"],
        "v7r1 authorization",
    )
    consumption_raw = _read_bound(
        Path(predecessor["consumption_path"]),
        predecessor["consumption_raw_sha256"],
        "v7r1 consumption",
    )
    terminal_raw = _read_bound(
        Path(predecessor["terminal_path"]),
        predecessor["terminal_raw_sha256"],
        "v7r1 terminal",
    )
    try:
        v7_capsule = v7_campaign.validate_capsule_v7(
            _strict_v7(capsule_raw, "v7r1 capsule"), v7_contract
        )
        v7_authorization = v7_campaign.validate_authorization_v7(
            _strict_v7(authorization_raw, "v7r1 authorization"),
            v7_contract,
            v7_capsule,
        )
        v7_consumption = v7_campaign.validate_consumption_v7(
            _strict_v7(consumption_raw, "v7r1 consumption"),
            v7_contract,
            v7_capsule,
            v7_authorization,
        )
        v7_terminal = v7_campaign.validate_terminal_record_v7(
            _strict_v7(terminal_raw, "v7r1 terminal"),
            v7_contract,
            v7_capsule,
            v7_authorization,
        )
    except ControllerPending:
        raise
    except Exception as error:
        raise ControllerPending(
            "V7R1_READ_ONLY_REPLAY",
            "V7R1_LEDGER_REPLAY_DIFFERS",
            str(error),
        ) from error
    if (
        v7_capsule["capsule_digest"] != predecessor["capsule_digest"]
        or v7_capsule["run_id"] != predecessor["run_id"]
        or v7_authorization["authorization_digest"]
        != predecessor["authorization_digest"]
        or v7_consumption["record_digest"]
        != predecessor["consumption_record_digest"]
        or v7_terminal["record_digest"] != predecessor["terminal_record_digest"]
        or v7_terminal["status"] != predecessor["status"]
        or v7_terminal["outcome"].get("failure_code")
        != predecessor["failure_code"]
        or v7_terminal["outcome"].get("failure_detail")
        != predecessor["failure_detail"]
        or v7_terminal["outcome"].get("actor_start_count_by_actor")
        != predecessor["actor_start_count_by_actor"]
        or v7_terminal["outcome"].get("q1_state") != "NOT_RUN"
        or v7_terminal["outcome"].get("q1_gate_count") != 14
        or v7_terminal["outcome"].get("q1_gate_mask") != 16_383
        or v7_terminal["outcome"].get("q1_output_slots") != [None] * 8
        or v7_terminal.get("artifact_raw_sha256") is not None
    ):
        raise ControllerPending(
            "V7R1_READ_ONLY_REPLAY",
            "V7R1_TERMINAL_BOUNDARY_DIFFERS",
            str(predecessor["terminal_path"]),
        )

    controller_raw = _read_bound(
        Path(predecessor["controller_outcome_path"]),
        predecessor["controller_outcome_raw_sha256"],
        "v7r1 controller outcome",
    )
    controller_outcome = _strict_v7(controller_raw, "v7r1 controller outcome")
    if controller_outcome != v7_terminal["outcome"]:
        raise ControllerPending(
            "V7R1_READ_ONLY_REPLAY",
            "V7R1_CONTROLLER_TERMINAL_DIFFERS",
            str(predecessor["controller_outcome_path"]),
        )

    supervisor_module = _load_supervisor_module()
    run_spec_raw = _read_bound(
        Path(predecessor["run_spec_path"]),
        predecessor["run_spec_raw_sha256"],
        "v7r1 run spec",
    )
    run_spec = _strict_v7(run_spec_raw, "v7r1 run spec")
    try:
        loaded_spec = supervisor_module.load_run_spec_v7(run_spec_raw)
    except Exception as error:
        raise ControllerPending(
            "V7R1_READ_ONLY_REPLAY",
            "V7R1_RUN_SPEC_REPLAY_DIFFERS",
            str(error),
        ) from error
    if (
        loaded_spec.start_capsule_digest != predecessor["capsule_digest"]
        or loaded_spec.start_authorization_digest
        != predecessor["authorization_digest"]
    ):
        raise ControllerPending(
            "V7R1_READ_ONLY_REPLAY",
            "V7R1_RUN_SPEC_AUTHORITY_DIFFERS",
            str(predecessor["run_spec_path"]),
        )

    for stage in _ACTORS:
        intent_raw = _read_bound(
            Path(predecessor[f"{stage}_intent_path"]),
            predecessor[f"{stage}_intent_raw_sha256"],
            f"v7r1 {stage} intent",
        )
        complete_raw = _read_bound(
            Path(predecessor[f"{stage}_complete_path"]),
            predecessor[f"{stage}_complete_raw_sha256"],
            f"v7r1 {stage} complete",
        )
        _validate_intent_and_complete(
            predecessor,
            stage,
            _strict_v7(intent_raw, f"v7r1 {stage} intent"),
            _strict_v7(complete_raw, f"v7r1 {stage} complete"),
        )

    supervisor_raw = _read_bound(
        Path(predecessor["supervisor_outcome_path"]),
        predecessor["supervisor_outcome_raw_sha256"],
        "v7r1 supervisor outcome",
    )
    supervisor_outcome = _strict_v7(supervisor_raw, "v7r1 supervisor outcome")
    if (
        supervisor_outcome.get("schema_version")
        != supervisor_module.OUTCOME_SCHEMA
        or supervisor_outcome.get("status") != supervisor_module.SUCCESS_STATUS
        or supervisor_outcome.get("completed_stages")
        != predecessor["completed_stages"]
        or supervisor_outcome.get("q1_completion_receipt_issued") is not False
        or supervisor_outcome.get("state_published") is not False
    ):
        raise ControllerPending(
            "V7R1_READ_ONLY_REPLAY",
            "V7R1_SUPERVISOR_OUTCOME_DIFFERS",
            str(predecessor["supervisor_outcome_path"]),
        )

    host_raw = _read_bound(
        Path(predecessor["host_result_path"]),
        predecessor["host_result_raw_sha256"],
        "v7r1 host result",
    )
    host_result = _strict_v7(host_raw, "v7r1 host result")
    if (
        supervisor_outcome.get("host_result") != host_result
        or host_result.get("status") != "DUAL_FORMAL_ARCHIVE_EXACT_AGREEMENT"
        or host_result.get("python_rust_formal_bytes_exact") is not True
        or host_result.get("python_rust_replayed_roots_exact") is not True
        or host_result.get("target_split_role_absence_attested") is not True
        or host_result.get("target_truth_accessed") is not False
        or host_result.get("split_accessed") is not False
        or host_result.get("role_evaluation_performed") is not False
    ):
        raise ControllerPending(
            "V7R1_READ_ONLY_REPLAY",
            "V7R1_HOST_RESULT_DIFFERS",
            str(predecessor["host_result_path"]),
        )

    try:
        artifact = adapter.build_completion_artifact_v1(
            supervisor_outcome,
            v7_capsule,
            run_spec,
            supervisor_module=supervisor_module,
            strict_json_bytes=v7_campaign.strict_json_bytes_v7,
            validate_successor_bundle=(
                v7_successor.validate_q1_prestart_successor_candidate_v7
            ),
        )
        receipt = completion.decode_completion_receipt_v1(artifact)
    except adapter.CompletionAdapterError as error:
        if error.pending_phase is not None:
            raise ControllerPending(
                error.pending_phase,
                error.failure_code,
                error.detail,
            ) from error
        _terminal(error.failure_code, error.detail)
    except Exception as error:
        _terminal("Q1_COMPLETION_REPLAY_ARTIFACT_INVALID", str(error))
    slots = _receipt_slots(receipt)
    bundle = v7_capsule["successor_prestart_bundle"]
    if (
        receipt.core.run_genesis_root.hex()
        != str(bundle["run_genesis_root"]).removeprefix("sha256:")
        or receipt.core.prestart_admission_receipt_root.hex()
        != str(bundle["receipt_root"]).removeprefix("sha256:")
    ):
        _terminal(
            "Q1_COMPLETION_REPLAY_RECEIPT_AUTHORITY_DIFFERS",
            "v7r1 predecessor roots differ",
        )
    return ReplayMaterialV8c(
        v7_contract,
        v7_capsule,
        v7_authorization,
        v7_consumption,
        v7_terminal,
        run_spec,
        supervisor_outcome,
        artifact,
        receipt,
        slots,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    try:
        if not path.exists():
            path.mkdir(mode=0o700)
            _fsync_directory(path.parent)
        info = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISDIR(info.st_mode)
            or path.resolve(strict=True) != path
        ):
            raise OSError("directory identity differs")
    except OSError as error:
        raise ControllerPending(
            "PATH_AND_FORMAT",
            "PROTOCOL_DIRECTORY_UNAVAILABLE",
            f"{path}: {error}",
        ) from error


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
        descriptor = os.open(
            path,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as error:
        raise ControllerPending(
            "PATH_AND_FORMAT", "CAMPAIGN_LOCK_UNAVAILABLE", str(error)
        ) from error
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ControllerPending(
                "PATH_AND_FORMAT", "CAMPAIGN_LOCK_BUSY", str(error)
            ) from error
        yield
    finally:
        os.close(descriptor)


def _stage_path(path: Path, digest: str) -> Path:
    return path.with_name(path.name + ".stage-" + digest[-16:])


def _publish_bytes(
    path: Path,
    payload: bytes,
    digest: str,
    *,
    mode: int = 0o400,
) -> None:
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as error:
            _terminal("Q1_COMPLETION_REPLAY_PUBLISHED_FILE_UNAVAILABLE", str(error))
        if existing != payload:
            _terminal("Q1_COMPLETION_REPLAY_NO_REPLACE_PATH_OCCUPIED", str(path))
        return
    stage_path = _stage_path(path, digest)
    try:
        if stage_path.exists():
            stage_path.unlink()
            _fsync_directory(path.parent)
        descriptor = os.open(
            stage_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            mode,
        )
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(stage_path, path, follow_symlinks=False)
        _fsync_directory(path.parent)
        stage_path.unlink()
        _fsync_directory(path.parent)
    except OSError as error:
        try:
            if stage_path.exists():
                stage_path.unlink()
                _fsync_directory(path.parent)
        except OSError:
            pass
        raise ControllerPending(
            "PATH_AND_FORMAT",
            "DURABLE_NO_REPLACE_PUBLICATION_FAILED",
            f"{path}: {error}",
        ) from error


def _publish_json(path: Path, value: Mapping[str, Any], digest: str) -> None:
    _publish_bytes(path, state.canonical_json_bytes(dict(value)), digest)


def _load_state_json(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    payload = _read_bytes(path, label)
    try:
        return payload, state.strict_json_bytes(payload, label)
    except Exception as error:
        raise ControllerPending(
            "PATH_AND_FORMAT",
            "STATE_RECORD_FORMAT_DIFFERS",
            f"{label}: {error}",
        ) from error


def _read_capsule(
    contract: Any,
    expected_digest: str | None = None,
) -> dict[str, Any] | None:
    path = Path(contract.value["fixed_paths"]["capsule_path"])
    if not path.exists():
        return None
    raw, value = _load_state_json(path, "v8c capsule")
    try:
        capsule = state.validate_capsule(value, contract)
    except Exception as error:
        raise ControllerPending(
            "PATH_AND_FORMAT", "CAPSULE_REPLAY_DIFFERS", str(error)
        ) from error
    if (
        raw != state.canonical_json_bytes(capsule)
        or (
            expected_digest is not None
            and capsule["capsule_digest"] != expected_digest
        )
    ):
        raise ControllerPending(
            "PATH_AND_FORMAT",
            "CAPSULE_REPLAY_DIFFERS",
            expected_digest or str(path),
        )
    return capsule


def _read_authorization(
    contract: Any,
    capsule: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = Path(contract.value["fixed_paths"]["authorization_path"])
    if not path.exists():
        return None
    raw, value = _load_state_json(path, "v8c authorization")
    try:
        authorization = state.validate_authorization(value, contract, capsule)
    except Exception as error:
        raise ControllerPending(
            "PATH_AND_FORMAT", "AUTHORIZATION_REPLAY_DIFFERS", str(error)
        ) from error
    if raw != state.canonical_json_bytes(authorization):
        raise ControllerPending(
            "PATH_AND_FORMAT", "AUTHORIZATION_REPLAY_DIFFERS", str(path)
        )
    return authorization


def _read_consumption(
    contract: Any,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = Path(contract.value["fixed_paths"]["consumption_path"])
    if not path.exists():
        return None
    raw, value = _load_state_json(path, "v8c consumption")
    try:
        consumption = state.validate_consumption(
            value, contract, capsule, authorization
        )
    except Exception as error:
        _terminal("Q1_COMPLETION_REPLAY_CONSUMPTION_DIFFERS", str(error))
    if raw != state.canonical_json_bytes(consumption):
        _terminal("Q1_COMPLETION_REPLAY_CONSUMPTION_DIFFERS", str(path))
    return consumption


def _read_terminal(
    contract: Any,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
) -> dict[str, Any] | None:
    fixed = contract.value["fixed_paths"]
    path = Path(fixed["terminal_path"])
    if not path.exists():
        return None
    raw, value = _load_state_json(path, "v8c terminal")
    try:
        terminal = state.validate_terminal(
            value, contract, capsule, authorization, consumption
        )
    except Exception as error:
        _terminal("Q1_COMPLETION_REPLAY_TERMINAL_DIFFERS", str(error))
    if raw != state.canonical_json_bytes(terminal):
        _terminal("Q1_COMPLETION_REPLAY_TERMINAL_DIFFERS", str(path))
    artifact_digest = terminal["artifact_raw_sha256"]
    artifact_path = Path(fixed["artifact_path"])
    if artifact_digest is not None:
        try:
            artifact = artifact_path.read_bytes()
            receipt = completion.decode_completion_receipt_v1(artifact)
        except Exception as error:
            _terminal("Q1_COMPLETION_REPLAY_TERMINAL_ARTIFACT_DIFFERS", str(error))
        if (
            state.raw_sha256(artifact) != artifact_digest
            or list(_receipt_slots(receipt))
            != terminal["outcome"]["q1_output_slots"]
        ):
            _terminal(
                "Q1_COMPLETION_REPLAY_TERMINAL_ARTIFACT_DIFFERS",
                str(artifact_path),
            )
    elif artifact_path.exists():
        _terminal(
            "Q1_COMPLETION_REPLAY_FAILED_TERMINAL_HAS_ARTIFACT",
            str(artifact_path),
        )
    return terminal


def _fresh_absence(contract: Any, *, allow_capsule: bool, allow_authorization: bool) -> None:
    fixed = contract.value["fixed_paths"]
    allowed: set[str] = set()
    if allow_capsule:
        allowed.add("capsule_path")
    if allow_authorization:
        allowed.add("authorization_path")
    for key in (
        "capsule_path",
        "authorization_path",
        "consumption_path",
        "outcome_path",
        "terminal_path",
        "artifact_path",
    ):
        if key not in allowed and Path(fixed[key]).exists():
            raise ControllerPending(
                "PATH_AND_FORMAT",
                "FRESH_COMPLETION_PATH_OCCUPIED",
                str(fixed[key]),
            )


def _write_probe(parent: Path, run_id: str) -> None:
    path = parent / f".q1-v8c-probe-{run_id}"
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as output:
            output.write(b"Q1-V8C-PROBE\n")
            output.flush()
            os.fsync(output.fileno())
        path.unlink()
        _fsync_directory(parent)
    except OSError as error:
        try:
            if path.exists():
                path.unlink()
                _fsync_directory(parent)
        except OSError:
            pass
        raise ControllerPending(
            "PATH_AND_FORMAT",
            "REVERSIBLE_PUBLICATION_PROBE_FAILED",
            f"{parent}: {error}",
        ) from error


def _preconsumption_path_gate(contract: Any, capsule: Mapping[str, Any]) -> None:
    fixed = contract.value["fixed_paths"]
    parents = {
        Path(fixed["capsule_path"]).parent,
        Path(fixed["authorization_path"]).parent,
        Path(fixed["artifact_path"]).parent,
    }
    for parent in sorted(parents):
        _write_probe(parent, capsule["run_id"])


def _validate_expected_completion(
    capsule: Mapping[str, Any],
    material: ReplayMaterialV8c,
) -> None:
    artifact_digest = state.raw_sha256(material.completion_artifact)
    receipt_root = "sha256:" + material.completion_receipt.receipt_root.hex()
    if (
        capsule["expected_completion_artifact_raw_sha256"] != artifact_digest
        or capsule["expected_completion_receipt_root"] != receipt_root
        or capsule["expected_q1_output_slots"] != list(material.q1_output_slots)
    ):
        raise ControllerPending(
            "V7R1_READ_ONLY_REPLAY",
            "CAPSULE_EXPECTED_COMPLETION_DIFFERS",
            artifact_digest,
        )


def _preflight_result(capsule: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "hegel-phase3a-q1-completion-replay-preflight/v8c",
        "status": capsule["status"],
        "capsule_digest": capsule["capsule_digest"],
        "capsule_raw_sha256": state.raw_sha256(
            state.canonical_json_bytes(dict(capsule))
        ),
        "source_closure_root": capsule["source_closure_root"],
        "expected_completion_artifact_raw_sha256": capsule[
            "expected_completion_artifact_raw_sha256"
        ],
        "expected_completion_receipt_root": capsule[
            "expected_completion_receipt_root"
        ],
        "expected_q1_output_slots": list(capsule["expected_q1_output_slots"]),
        "authorization_consumption_count": 0,
        "logical_campaign_count": 0,
        "popen_count": 0,
        "actor_start_count": 0,
        "docker_invocation_count": 0,
        "network_accessed": False,
        "v7r1_resumed": False,
        "v7r1_outputs_mutated": False,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "q2_state": "NOT_RUN",
    }


def _unbound_pending(
    contract_path: Path,
    phase: str,
    condition: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "schema_version": "hegel-phase3a-q1-completion-replay-unbound-pending/v8c",
        "status": state.PENDING_NOT_CONSUMED,
        "contract_path": str(contract_path),
        "capsule_digest": None,
        "authorization_consumption_count": 0,
        "logical_campaign_count": 0,
        "popen_count": 0,
        "actor_start_count": 0,
        "docker_invocation_count": 0,
        "phase": phase,
        "condition": condition,
        "detail": detail,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "q2_state": "NOT_RUN",
    }


def _pending_result(
    contract: Any,
    capsule: Mapping[str, Any],
    error: ControllerPending | ControllerTerminal,
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(error, ControllerPending):
        phase, condition, detail = error.phase, error.condition, error.detail
    else:
        phase, condition, detail = (
            "PRECONSUMPTION_COMPLETION_REPLAY",
            error.code,
            error.detail,
        )
    return state.build_pending(
        contract,
        capsule,
        phase=phase,
        condition=condition,
        detail=detail,
        authorization=authorization,
    )


def preflight_v8c(
    contract_path: Path,
    *,
    run_id: str,
    created_at_unix_seconds: int,
) -> dict[str, Any]:
    try:
        contract = state.load_contract(contract_path)
    except Exception as error:
        return _unbound_pending(
            contract_path,
            "CONTRACT",
            "CONTRACT_UNAVAILABLE_OR_INVALID",
            str(error),
        )
    try:
        _ensure_roots(contract)
        with _campaign_lock(contract):
            existing = _read_capsule(contract)
            if existing is not None:
                if (
                    existing["run_id"] != run_id
                    or existing["created_at_unix_seconds"]
                    != created_at_unix_seconds
                ):
                    raise ControllerPending(
                        "PATH_AND_FORMAT",
                        "CAPSULE_ALREADY_PUBLISHED_FOR_DIFFERENT_INPUTS",
                        str(contract.value["fixed_paths"]["capsule_path"]),
                    )
                _replay_source_closure(contract, existing)
                material = _replay_v7r1(contract)
                _validate_expected_completion(existing, material)
                return _preflight_result(existing)

            _fresh_absence(
                contract, allow_capsule=False, allow_authorization=False
            )
            source_rows = _source_rows(contract)
            material = _replay_v7r1(contract)
            receipt_root = (
                "sha256:" + material.completion_receipt.receipt_root.hex()
            )
            capsule_inputs = {
                "schema_version": state.CAPSULE_INPUTS_SCHEMA,
                "source_closure_rows": source_rows,
                "source_closure_root": state.source_closure_root(source_rows),
                "run_id": run_id,
                "created_at_unix_seconds": created_at_unix_seconds,
                "expected_completion_artifact_raw_sha256": state.raw_sha256(
                    material.completion_artifact
                ),
                "expected_completion_receipt_root": receipt_root,
                "expected_q1_output_slots": list(material.q1_output_slots),
            }
            capsule = state.build_capsule(contract, capsule_inputs)
            _preconsumption_path_gate(contract, capsule)
            _publish_json(
                Path(contract.value["fixed_paths"]["capsule_path"]),
                capsule,
                capsule["capsule_digest"],
            )
            replayed = _read_capsule(contract, capsule["capsule_digest"])
            assert replayed is not None
            return _preflight_result(replayed)
    except (ControllerPending, ControllerTerminal) as error:
        return _unbound_pending(
            contract_path,
            "PREFLIGHT",
            getattr(error, "condition", getattr(error, "code", "PREFLIGHT_FAILED")),
            getattr(error, "detail", str(error)),
        )


def record_authorization_v8c(
    contract_path: Path,
    capsule_digest: str,
) -> dict[str, Any]:
    try:
        contract = state.load_contract(contract_path)
    except Exception as error:
        return _unbound_pending(
            contract_path,
            "CONTRACT",
            "CONTRACT_UNAVAILABLE_OR_INVALID",
            str(error),
        )
    capsule: dict[str, Any] | None = None
    try:
        _ensure_roots(contract)
        with _campaign_lock(contract):
            capsule = _read_capsule(contract, capsule_digest)
            if capsule is None:
                raise ControllerPending(
                    "PATH_AND_FORMAT", "CAPSULE_NOT_PUBLISHED", capsule_digest
                )
            existing = _read_authorization(contract, capsule)
            if existing is not None:
                return existing
            _fresh_absence(contract, allow_capsule=True, allow_authorization=False)
            _replay_source_closure(contract, capsule)
            material = _replay_v7r1(contract)
            _validate_expected_completion(capsule, material)
            _preconsumption_path_gate(contract, capsule)
            authorization = state.build_authorization(contract, capsule)
            _publish_json(
                Path(contract.value["fixed_paths"]["authorization_path"]),
                authorization,
                capsule_digest,
            )
            replayed = _read_authorization(contract, capsule)
            assert replayed is not None
            return replayed
    except (ControllerPending, ControllerTerminal) as error:
        if capsule is None:
            return _unbound_pending(
                contract_path,
                "RECORD_AUTHORIZATION",
                getattr(error, "condition", getattr(error, "code", "FAILED")),
                getattr(error, "detail", str(error)),
            )
        return _pending_result(contract, capsule, error)


def _not_authorized(capsule: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "hegel-phase3a-q1-completion-replay-not-authorized/v8c",
        "status": "Q1_COMPLETION_REPLAY_V8C_NOT_AUTHORIZED_NOT_CONSUMED",
        "capsule_digest": capsule["capsule_digest"],
        "authorization_consumption_count": 0,
        "logical_campaign_count": 0,
        "popen_count": 0,
        "actor_start_count": 0,
        "docker_invocation_count": 0,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "q2_state": "NOT_RUN",
    }


def _publish_success(
    contract: Any,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    material: ReplayMaterialV8c,
) -> dict[str, Any]:
    artifact_digest = state.raw_sha256(material.completion_artifact)
    terminal = state.build_terminal(
        contract,
        capsule,
        authorization,
        consumption,
        success=True,
        q1_output_slots=list(material.q1_output_slots),
        artifact_raw_sha256=artifact_digest,
        read_only_replay_completed=True,
    )
    fixed = contract.value["fixed_paths"]
    _publish_bytes(
        Path(fixed["artifact_path"]),
        material.completion_artifact,
        capsule["capsule_digest"],
    )
    _publish_json(
        Path(fixed["outcome_path"]),
        terminal["outcome"],
        capsule["capsule_digest"],
    )
    _publish_json(
        Path(fixed["terminal_path"]),
        terminal,
        capsule["capsule_digest"],
    )
    replayed = _read_terminal(
        contract, capsule, authorization, consumption
    )
    assert replayed is not None
    return replayed


def _publish_failure(
    contract: Any,
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    error: ControllerPending | ControllerTerminal | Exception,
) -> dict[str, Any]:
    if isinstance(error, ControllerTerminal):
        code, detail = error.code, error.detail
    elif isinstance(error, ControllerPending):
        code, detail = error.condition, error.detail
    else:
        code, detail = "Q1_COMPLETION_REPLAY_POSTCONSUMPTION_EXCEPTION", str(error)
    terminal = state.build_terminal(
        contract,
        capsule,
        authorization,
        consumption,
        success=False,
        failure_code=code,
        failure_detail=detail,
        read_only_replay_completed=False,
    )
    fixed = contract.value["fixed_paths"]
    try:
        _publish_json(
            Path(fixed["outcome_path"]),
            terminal["outcome"],
            capsule["capsule_digest"],
        )
        _publish_json(
            Path(fixed["terminal_path"]),
            terminal,
            capsule["capsule_digest"],
        )
    except ControllerPending as publication_error:
        return _partial_finalization_result(
            capsule,
            authorization,
            consumption,
            publication_error,
            intended_failure_code=code,
            intended_failure_detail=detail,
        )
    except ControllerTerminal as publication_error:
        return _unsealed_terminal_result(
            capsule["capsule_digest"],
            capsule,
            authorization,
            consumption,
            publication_error,
            intended_failure_code=code,
            intended_failure_detail=detail,
        )
    try:
        replayed = _read_terminal(
            contract, capsule, authorization, consumption
        )
    except ControllerPending as replay_error:
        return _partial_finalization_result(
            capsule,
            authorization,
            consumption,
            replay_error,
            intended_failure_code=code,
            intended_failure_detail=detail,
        )
    except ControllerTerminal as replay_error:
        return _unsealed_terminal_result(
            capsule["capsule_digest"],
            capsule,
            authorization,
            consumption,
            replay_error,
            intended_failure_code=code,
            intended_failure_detail=detail,
        )
    if replayed is None:
        return _partial_finalization_result(
            capsule,
            authorization,
            consumption,
            ControllerPending(
                "PARTIAL_FINALIZATION",
                "TERMINAL_NOT_DURABLY_REPLAYABLE",
                str(fixed["terminal_path"]),
            ),
            intended_failure_code=code,
            intended_failure_detail=detail,
        )
    return replayed


def _partial_finalization_result(
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    consumption: Mapping[str, Any],
    error: ControllerPending,
    *,
    intended_failure_code: str | None = None,
    intended_failure_detail: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": (
            "hegel-phase3a-q1-completion-replay-partial-finalization/v8c"
        ),
        "status": (
            "Q1_COMPLETION_REPLAY_V8C_PARTIAL_FINALIZATION_RECOVERY_REQUIRED"
        ),
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "consumption_record_digest": consumption["record_digest"],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "popen_count": 0,
        "actor_start_count": 0,
        "docker_invocation_count": 0,
        "exact_partial_finalization_reentry_allowed": True,
        "phase": error.phase,
        "condition": error.condition,
        "detail": error.detail,
        "intended_failure_code": intended_failure_code,
        "intended_failure_detail": intended_failure_detail,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "q2_state": "NOT_RUN",
    }


def _unsealed_terminal_result(
    requested_capsule_digest: str,
    capsule: Mapping[str, Any] | None,
    authorization: Mapping[str, Any] | None,
    consumption: Mapping[str, Any] | None,
    error: ControllerTerminal,
    *,
    intended_failure_code: str | None = None,
    intended_failure_detail: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": (
            "hegel-phase3a-q1-completion-replay-unsealed-terminal/v8c"
        ),
        "status": UNSEALED_TERMINAL_NO_RESUME,
        "capsule_digest": (
            capsule["capsule_digest"]
            if capsule is not None
            else requested_capsule_digest
        ),
        "authorization_digest": (
            authorization["authorization_digest"]
            if authorization is not None
            else None
        ),
        "consumption_record_digest": (
            consumption["record_digest"] if consumption is not None else None
        ),
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "popen_count": 0,
        "actor_start_count": 0,
        "docker_invocation_count": 0,
        "failure_code": error.code,
        "failure_detail": error.detail,
        "intended_failure_code": intended_failure_code,
        "intended_failure_detail": intended_failure_detail,
        "durable_terminal_published": False,
        "terminal_no_resume": True,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "q2_state": "NOT_RUN",
    }


def run_v8c(contract_path: Path, capsule_digest: str) -> dict[str, Any]:
    try:
        contract = state.load_contract(contract_path)
    except Exception as error:
        return _unbound_pending(
            contract_path,
            "CONTRACT",
            "CONTRACT_UNAVAILABLE_OR_INVALID",
            str(error),
        )
    capsule: dict[str, Any] | None = None
    authorization: dict[str, Any] | None = None
    consumption: dict[str, Any] | None = None
    consumption_path_observed = False
    try:
        _ensure_roots(contract)
        with _campaign_lock(contract):
            consumption_path_observed = Path(
                contract.value["fixed_paths"]["consumption_path"]
            ).exists()
            capsule = _read_capsule(contract, capsule_digest)
            if capsule is None:
                raise ControllerPending(
                    "PATH_AND_FORMAT", "CAPSULE_NOT_PUBLISHED", capsule_digest
                )
            authorization = _read_authorization(contract, capsule)
            if authorization is None:
                if consumption_path_observed:
                    raise ControllerTerminal(
                        "Q1_COMPLETION_REPLAY_AUTHORIZATION_MISSING_AFTER_CONSUMPTION",
                        contract.value["fixed_paths"]["authorization_path"],
                    )
                return _not_authorized(capsule)
            consumption = _read_consumption(
                contract, capsule, authorization
            )
            if consumption is not None:
                terminal = _read_terminal(
                    contract, capsule, authorization, consumption
                )
                if terminal is not None:
                    return terminal
            if consumption is None:
                _fresh_absence(
                    contract, allow_capsule=True, allow_authorization=True
                )
            _replay_source_closure(contract, capsule)
            material = _replay_v7r1(contract)
            _validate_expected_completion(capsule, material)
            if consumption is None:
                _preconsumption_path_gate(contract, capsule)
                candidate = state.build_consumption(
                    contract, capsule, authorization
                )
                _publish_json(
                    Path(contract.value["fixed_paths"]["consumption_path"]),
                    candidate,
                    capsule_digest,
                )
                consumption = _read_consumption(
                    contract, capsule, authorization
                )
                assert consumption is not None
            return _publish_success(
                contract, capsule, authorization, consumption, material
            )
    except (ControllerPending, ControllerTerminal) as error:
        if consumption_path_observed and consumption is None:
            terminal_error = (
                error
                if isinstance(error, ControllerTerminal)
                else ControllerTerminal(error.condition, error.detail)
            )
            return _unsealed_terminal_result(
                capsule_digest,
                capsule,
                authorization,
                consumption,
                terminal_error,
            )
        if capsule is None:
            return _unbound_pending(
                contract_path,
                "RUN",
                getattr(error, "condition", getattr(error, "code", "FAILED")),
                getattr(error, "detail", str(error)),
            )
        if consumption is None and not consumption_path_observed:
            return _pending_result(contract, capsule, error, authorization)
        assert authorization is not None
        if (
            isinstance(error, ControllerPending)
            and error.phase == "PATH_AND_FORMAT"
            and error.condition == "DURABLE_NO_REPLACE_PUBLICATION_FAILED"
        ):
            return _partial_finalization_result(
                capsule, authorization, consumption, error
            )
        return _publish_failure(
            contract, capsule, authorization, consumption, error
        )
    except Exception as error:
        if consumption_path_observed and consumption is None:
            return _unsealed_terminal_result(
                capsule_digest,
                capsule,
                authorization,
                consumption,
                ControllerTerminal(
                    "Q1_COMPLETION_REPLAY_POSTCONSUMPTION_EXCEPTION",
                    str(error),
                ),
            )
        if capsule is None or authorization is None or consumption is None:
            return _unbound_pending(
                contract_path,
                "RUN",
                "CONTROLLER_EXCEPTION_BEFORE_CONSUMPTION",
                str(error),
            )
        return _publish_failure(
            contract, capsule, authorization, consumption, error
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--record-authorization", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--capsule-digest")
    parser.add_argument("--run-id")
    parser.add_argument("--created-at-unix-seconds", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.preflight:
        if args.run_id is None or args.created_at_unix_seconds is None:
            raise SystemExit(
                "--preflight requires --run-id and --created-at-unix-seconds"
            )
        result = preflight_v8c(
            args.contract,
            run_id=args.run_id,
            created_at_unix_seconds=args.created_at_unix_seconds,
        )
    else:
        if args.capsule_digest is None:
            raise SystemExit(
                "--record-authorization/--run requires --capsule-digest"
            )
        if args.record_authorization:
            result = record_authorization_v8c(
                args.contract, args.capsule_digest
            )
        else:
            result = run_v8c(args.contract, args.capsule_digest)
    sys.stdout.buffer.write(state.canonical_json_bytes(result))
    status = result.get("status")
    if status == state.PENDING_NOT_CONSUMED:
        return PENDING_EXIT_CODE
    if status == "Q1_COMPLETION_REPLAY_V8C_NOT_AUTHORIZED_NOT_CONSUMED":
        return NOT_AUTHORIZED_EXIT_CODE
    if status == state.TERMINAL_NO_RESUME:
        return TERMINAL_EXIT_CODE
    if status == UNSEALED_TERMINAL_NO_RESUME:
        return TERMINAL_EXIT_CODE
    if status == "Q1_COMPLETION_REPLAY_V8C_PARTIAL_FINALIZATION_RECOVERY_REQUIRED":
        return PENDING_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
