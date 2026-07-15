from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any, Mapping, Sequence

from ..events import Event, JsonlEventSink
from ..models import stable_hash
from ..splits import SplitManifest
from .execution_contract_integration_v2 import (
    ExecutionContractCompileBundleV2,
    ExecutionContractSubprocessBackendV2,
)
from .paper_protocol import PaperProtocol
from .train_execution_contract_actual_v2 import (
    MODEL_INFERENCE_SLOTS,
    V320_PROTOCOL_RELATIVE_PATH,
    _configure_environment,
    _prepare_scoped_runtime_assets,
)
from .train_execution_contract_crossfit_v2 import (
    ASSET_PREFLIGHT_POLICY,
    ORGANIZE_TRACE_REFINED_ITEM_OUT_FOLDS,
    TRACE_REFINED_CONTRACT_VARIANT,
    TrainItemOutCompileV2,
    compile_v320_train_item_out_crossfit_v2,
)
from .train_execution_contract_development_v2 import (
    SKILLLEARN_BENCHMARK_RELATIVE_ROOT,
    V320_MANIFEST_RELATIVE_PATH,
)
from .train_outcome_production_runner_v2 import (
    ProductionTrainCandidateRunnerV2,
)
from .train_outcome_ranker_v2 import (
    TrainCandidateSpecV2,
    TrainCandidateWorkUnitV2,
    TrainOutcomeRankingResultV2,
    TrainOutcomeRankerV2,
)
from .v320_train_candidate_material_v2 import (
    V320_EVALUATOR_EPOCH,
    V320_MANIFEST_HASH,
    V320_MODEL,
)


TYPED_ASSIGNMENT_CROSSFIT_VERSION = (
    "historically_informed_typed_assignment_train_crossfit_v3"
)
TYPED_ASSIGNMENT_COMPILE_POLICY = (
    "v320_typed_assignment_train_crossfit_compile_v3"
)
TYPED_ASSIGNMENT_ACTUAL_POLICY = (
    "v320_typed_assignment_train_crossfit_actual_offline_v3"
)
TYPED_ASSIGNMENT_PREREGISTRATION_POLICY = (
    "v320_typed_assignment_train_crossfit_preregistration_v3"
)
TYPED_ASSIGNMENT_PROVIDER_POLICY = (
    "pretask_plus_model_response_then_transport_only_pro_single_batch_v2"
)
PROVIDER_TRANSPORT_FAILURE_RECEIPT_VERSION = (
    "provider_transport_no_model_response_receipt_v1"
)
TYPED_ASSIGNMENT_CLASSIFICATION_OBJECTIVE = (
    "content_evidence_to_typed_file_assignment_to_one_to_one_reconciliation"
)
TYPED_ASSIGNMENT_STATIC_COMPLEXITY_DELTA = 3
TYPED_ASSIGNMENT_STATIC_COMPLEXITY = 11
REGISTERED_HELDOUT_ITEM_IDS = tuple(
    sorted(ORGANIZE_TRACE_REFINED_ITEM_OUT_FOLDS)
)
REGISTERED_CELL_COUNT = 3
MAXIMUM_CONCURRENT_RUNNER_CALLS = 3
EXPECTED_ACTIVE_EXECUTION_COUNT = 3
EXPECTED_INACTIVE_RAW_REPLAY_COUNT = 111
MINIMUM_VALID_ACTIVE_EXECUTIONS = 3
MINIMUM_DISTINCT_FOLD_RECOVERIES = 2
PREREGISTRATION_RELATIVE_PATH = (
    "manifests/train_typed_assignment_organize_crossfit_v3.json"
)
COMPILE_REPORT_FILENAME = "typed_assignment_crossfit.compile.report.json"
ACTUAL_REPORT_FILENAME = "typed_assignment_crossfit.report.json"
FAILURE_REPORT_FILENAME = "typed_assignment_crossfit.failure.json"
EXECUTION_EVENTS_FILENAME = "typed_assignment_crossfit.execution.events.jsonl"

IMPLEMENTATION_RELATIVE_PATHS = (
    "assumption_agent/typed_assignment_contract_v3.py",
    "assumption_agent/benchmarks/typed_assignment_runtime_tool_v3.py",
    "assumption_agent/benchmarks/typed_assignment_integration_v3.py",
)
EXECUTION_IMPLEMENTATION_RELATIVE_PATHS = (
    "assumption_agent/benchmarks/train_typed_assignment_crossfit_v3.py",
    "assumption_agent/benchmarks/train_outcome_ranker_v2.py",
    "assumption_agent/benchmarks/train_outcome_production_runner_v2.py",
    "assumption_agent/benchmarks/train_execution_contract_actual_v2.py",
    "assumption_agent/benchmarks/train_execution_contract_crossfit_v2.py",
    "assumption_agent/benchmarks/execution_contract_integration_v2.py",
    "assumption_agent/benchmarks/execution_contract_prompt_v2.py",
    "assumption_agent/benchmarks/skilllearn_lifecycle.py",
)


class TrainTypedAssignmentCrossfitError(PermissionError):
    """The typed-assignment TRAIN search crossed its preregistration."""


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainTypedAssignmentCrossfitError(f"{label} is not sha256")
    return value


def _receipt_body_hash_is_bound(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    without_hash = dict(payload)
    embedded = without_hash.pop("receipt_hash", None)
    return (
        isinstance(embedded, str)
        and len(embedded) == 64
        and all(character in "0123456789abcdef" for character in embedded)
        and embedded == stable_hash(without_hash)
    )


def _runtime_receipt_bodies_are_bound(row: Mapping[str, Any]) -> bool:
    prepare = row.get("prepare_receipt_body")
    reconciliation = row.get("reconciliation_receipt_body")
    delivery = row.get("post_agent_runtime_delivery")
    if (
        not isinstance(prepare, Mapping)
        or not isinstance(reconciliation, Mapping)
        or not isinstance(delivery, Mapping)
        or not _receipt_body_hash_is_bound(prepare)
        or not _receipt_body_hash_is_bound(reconciliation)
    ):
        return False
    runtime_tool_sha256 = prepare.get("runtime_tool_sha256")
    return (
        row.get("prepare_receipt_hash") == prepare.get("receipt_hash")
        and row.get("reconciliation_receipt_hash")
        == reconciliation.get("receipt_hash")
        and row.get("contract_hash") == prepare.get("contract_hash")
        and row.get("contract_hash") == reconciliation.get("contract_hash")
        and row.get("evidence_set_hash") == prepare.get("evidence_set_hash")
        and row.get("evidence_set_hash")
        == reconciliation.get("evidence_set_hash")
        and prepare.get("host_safe_receipt") is True
        and reconciliation.get("host_safe_receipt") is True
        and prepare.get("raw_public_instruction_in_receipt") is False
        and prepare.get("raw_content_evidence_in_receipt") is False
        and prepare.get("source_filenames_in_receipt") is False
        and reconciliation.get("raw_public_instruction_in_receipt") is False
        and reconciliation.get("raw_content_evidence_in_receipt") is False
        and reconciliation.get("source_filenames_in_receipt") is False
        and runtime_tool_sha256 == reconciliation.get("runtime_tool_sha256")
        and runtime_tool_sha256 == delivery.get("runtime_tool_sha256")
        and runtime_tool_sha256 == delivery.get("container_readback_sha256")
        and delivery.get(
            "fresh_unpredictable_path_selected_after_agent_exit"
        )
        is True
        and delivery.get("pre_agent_prepare_tool_removed_before_agent_start")
        is True
    )


def _require_git_object_id(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainTypedAssignmentCrossfitError(
            f"{label} is not a Git object id"
        )
    return value


def _sha256_regular_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise TrainTypedAssignmentCrossfitError(
            "typed-assignment implementation file is unavailable"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_complete_model_response_canary_v3(
    path: Path,
    *,
    provider_label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a completed canary without using its semantic acceptance."""

    if provider_label not in {"plus", "pro"}:
        raise TrainTypedAssignmentCrossfitError(
            "provider canary label is not registered"
        )
    if path.is_symlink() or not path.is_file():
        raise TrainTypedAssignmentCrossfitError(
            "provider canary report is unavailable"
        )
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainTypedAssignmentCrossfitError(
            "provider canary report is unreadable"
        ) from exc
    nodes = payload.get("nodes") if isinstance(payload, Mapping) else None
    node_count = (
        payload.get("recursive_node_count")
        if isinstance(payload, Mapping)
        else None
    )
    if (
        not isinstance(payload, dict)
        or payload.get("canary_version") != "proposal_canary_v1"
        or payload.get("model") != V320_MODEL
        or payload.get("provider_chain") != ["openai_compatible"]
        or not _is_sha256(payload.get("provider_chain_hash"))
        or not isinstance(payload.get("root_hypothesis_id"), str)
        or not payload.get("root_hypothesis_id")
        or not _is_sha256(payload.get("root_hypothesis_hash"))
        or isinstance(node_count, bool)
        or not isinstance(node_count, int)
        or node_count <= 0
        or not isinstance(payload.get("recursive_depth"), int)
        or isinstance(payload.get("recursive_depth"), bool)
        or payload.get("recursive_depth") < 0
        or not isinstance(nodes, list)
        or len(nodes) != node_count
        or any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("hypothesis_id"), str)
            or not row.get("hypothesis_id")
            or not _is_sha256(row.get("hypothesis_hash"))
            or isinstance(row.get("depth"), bool)
            or not isinstance(row.get("depth"), int)
            or not isinstance(row.get("passed"), bool)
            or not isinstance(row.get("checks"), list)
            for row in nodes
        )
        or not isinstance(payload.get("accepted"), bool)
        or (
            payload.get("accepted") is True
            and not isinstance(payload.get("accepted_program"), Mapping)
        )
        or (
            payload.get("accepted") is False
            and payload.get("accepted_program") is not None
        )
        or payload.get("api_key_present") is not True
        or payload.get("secret_value_persisted") is not False
        or payload.get("raw_content_persisted") is not False
    ):
        raise TrainTypedAssignmentCrossfitError(
            "provider canary lacks a complete model response"
        )
    receipt = {
        "probe_kind": "complete_model_response_canary",
        "provider_label_hash": stable_hash(
            {"provider_label": provider_label}
        ),
        "canary_file_sha256": hashlib.sha256(raw).hexdigest(),
        "canary_payload_hash": stable_hash(payload),
        "model_response_received": True,
        "semantic_acceptance_used_for_provider_selection": False,
        "canary_semantic_accepted": payload["accepted"],
        "raw_canary_content_persisted": False,
    }
    return payload, receipt


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_verified_provider_event_ledger_v3(
    path: Path,
) -> tuple[tuple[dict[str, Any], ...], str]:
    if path.is_symlink() or not path.is_file():
        raise TrainTypedAssignmentCrossfitError(
            "provider failure event ledger is unavailable"
        )
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise TrainTypedAssignmentCrossfitError(
            "provider failure event ledger is unreadable"
        ) from exc
    if not raw or len(raw) > 32 * 1024 * 1024:
        raise TrainTypedAssignmentCrossfitError(
            "provider failure event ledger is outside its byte bound"
        )
    rows: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    for line in text.splitlines():
        if not line:
            raise TrainTypedAssignmentCrossfitError(
                "provider failure event ledger contains a blank row"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainTypedAssignmentCrossfitError(
                "provider failure event ledger contains malformed JSON"
            ) from exc
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("payload"), Mapping)
            or set(row)
            != {
                "event",
                "stage",
                "trace_id",
                "payload",
                "payload_hash",
                "event_id",
                "raw_content_persisted",
            }
        ):
            raise TrainTypedAssignmentCrossfitError(
                "provider failure event envelope drifted"
            )
        reconstructed = Event(
            event=str(row["event"]),
            stage=str(row["stage"]),
            trace_id=str(row["trace_id"]),
            payload=dict(row["payload"]),
        ).to_dict()
        if row != reconstructed or row["event_id"] in event_ids:
            raise TrainTypedAssignmentCrossfitError(
                "provider failure event ledger integrity failed"
            )
        event_ids.add(str(row["event_id"]))
        rows.append(row)
    if not rows:
        raise TrainTypedAssignmentCrossfitError(
            "provider failure event ledger is empty"
        )
    return tuple(rows), hashlib.sha256(raw).hexdigest()


_TRANSPORT_FAILURE_ERROR_TYPES = frozenset(
    {
        "BrokenPipeError",
        "ConnectionAbortedError",
        "ConnectionError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "OSError",
        "RuntimeError",
        "TimeoutError",
        "URLError",
    }
)
_TRANSPORT_FAILURE_HTTP_STATUSES = frozenset(
    {408, 425, 429, 500, 502, 503, 504}
)


def _transport_failure_summary_v3(
    event_ledger_path: Path,
) -> dict[str, Any]:
    events, ledger_sha256 = _read_verified_provider_event_ledger_v3(
        event_ledger_path
    )
    started = [
        row for row in events if row["event"] == "model_attempt_started"
    ]
    failed = [
        row for row in events if row["event"] == "model_attempt_failed"
    ]
    succeeded = [
        row for row in events if row["event"] == "model_attempt_succeeded"
    ]
    selected = [
        row for row in events if row["event"] == "model_provider_selected"
    ]
    terminal = [
        row
        for row in events
        if row["event"]
        in {"model_provider_failed", "model_provider_chain_exhausted"}
    ]
    request_hashes = {
        str(row["payload"].get("request_hash") or "")
        for row in (*started, *failed)
    }
    failure_rows = [
        {
            "error_type": row["payload"].get("error_type"),
            "http_status": row["payload"].get("http_status"),
            "retryable": row["payload"].get("retryable"),
        }
        for row in failed
    ]
    if (
        not started
        or not failed
        or not terminal
        or succeeded
        or selected
        or len(request_hashes) != 1
        or not _is_sha256(next(iter(request_hashes), ""))
        or any(
            row["payload"].get("model") != V320_MODEL
            for row in (*started, *failed)
        )
        or any(
            (
                row["error_type"] not in _TRANSPORT_FAILURE_ERROR_TYPES
                and row["http_status"]
                not in _TRANSPORT_FAILURE_HTTP_STATUSES
            )
            for row in failure_rows
        )
    ):
        raise TrainTypedAssignmentCrossfitError(
            "Plus canary failure is not a verified transport/no-response failure"
        )
    return {
        "event_ledger_sha256": ledger_sha256,
        "request_hash": next(iter(request_hashes)),
        "attempt_started_count": len(started),
        "attempt_failed_count": len(failed),
        "terminal_failure_event_count": len(terminal),
        "failure_rows": failure_rows,
        "failure_row_set_hash": stable_hash(
            {"failure_rows": failure_rows}
        ),
        "model_response_received": False,
        "transport_or_no_response_failure_verified": True,
        "raw_failure_content_persisted": False,
    }


def write_plus_transport_failure_receipt_v3(
    *,
    event_ledger_path: Path,
    expected_canary_report_path: Path,
    process_exit_code: int,
    output_path: Path,
) -> Path:
    """Create a safe Plus failure receipt from a canonical event ledger."""

    if (
        event_ledger_path.is_symlink()
        or expected_canary_report_path.exists()
        or expected_canary_report_path.is_symlink()
        or isinstance(process_exit_code, bool)
        or not isinstance(process_exit_code, int)
        or process_exit_code == 0
    ):
        raise TrainTypedAssignmentCrossfitError(
            "Plus failure receipt requires an absent report and nonzero exit"
        )
    summary = _transport_failure_summary_v3(
        event_ledger_path.resolve(strict=True)
    )
    without_hash = {
        "receipt_version": PROVIDER_TRANSPORT_FAILURE_RECEIPT_VERSION,
        "provider_label_hash": stable_hash({"provider_label": "plus"}),
        "model_hash": stable_hash({"model": V320_MODEL}),
        "expected_canary_report_path_hash": stable_hash(
            {"path": str(expected_canary_report_path)}
        ),
        "expected_canary_report_absent": True,
        "process_exit_code": process_exit_code,
        "failure_summary": summary,
        "semantic_acceptance_observed": False,
        "crossfit_task_calls": 0,
        "crossfit_model_calls": 0,
        "raw_error_persisted": False,
        "secret_value_persisted": False,
    }
    receipt = {**without_hash, "receipt_hash": stable_hash(without_hash)}
    destination = output_path.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Plus transport failure receipt already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _verify_plus_transport_failure_receipt_v3(
    *,
    receipt_path: Path,
    event_ledger_path: Path,
    expected_canary_report_path: Path,
) -> dict[str, Any]:
    if (
        event_ledger_path.is_symlink()
        or expected_canary_report_path.exists()
        or expected_canary_report_path.is_symlink()
    ):
        raise TrainTypedAssignmentCrossfitError(
            "Plus failure receipt conflicts with an existing canary report"
        )
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise TrainTypedAssignmentCrossfitError(
            "Plus transport failure receipt is unavailable"
        )
    try:
        raw = receipt_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainTypedAssignmentCrossfitError(
            "Plus transport failure receipt is unreadable"
        ) from exc
    if not isinstance(payload, dict):
        raise TrainTypedAssignmentCrossfitError(
            "Plus transport failure receipt is malformed"
        )
    without_hash = dict(payload)
    receipt_hash = without_hash.pop("receipt_hash", None)
    summary = _transport_failure_summary_v3(
        event_ledger_path.resolve(strict=True)
    )
    if (
        receipt_hash != stable_hash(without_hash)
        or payload.get("receipt_version")
        != PROVIDER_TRANSPORT_FAILURE_RECEIPT_VERSION
        or payload.get("provider_label_hash")
        != stable_hash({"provider_label": "plus"})
        or payload.get("model_hash") != stable_hash({"model": V320_MODEL})
        or payload.get("expected_canary_report_path_hash")
        != stable_hash({"path": str(expected_canary_report_path)})
        or payload.get("expected_canary_report_absent") is not True
        or isinstance(payload.get("process_exit_code"), bool)
        or not isinstance(payload.get("process_exit_code"), int)
        or payload.get("process_exit_code") == 0
        or payload.get("failure_summary") != summary
        or payload.get("semantic_acceptance_observed") is not False
        or payload.get("crossfit_task_calls") != 0
        or payload.get("crossfit_model_calls") != 0
        or payload.get("raw_error_persisted") is not False
        or payload.get("secret_value_persisted") is not False
    ):
        raise TrainTypedAssignmentCrossfitError(
            "Plus transport failure receipt drifted"
        )
    return {
        "probe_kind": "transport_or_no_model_response_failure",
        "receipt_file_sha256": hashlib.sha256(raw).hexdigest(),
        "receipt_hash": receipt_hash,
        "failure_summary": summary,
        "model_response_received": False,
        "raw_failure_content_persisted": False,
    }


def write_provider_selection_receipt_v3(
    *,
    output_path: Path,
    plus_canary_report_path: Path | None = None,
    plus_transport_failure_receipt_path: Path | None = None,
    plus_failure_event_ledger_path: Path | None = None,
    plus_expected_canary_report_path: Path | None = None,
    pro_canary_report_path: Path | None = None,
) -> Path:
    """Select Plus on any complete response; otherwise Pro before the batch."""

    complete_plus = plus_canary_report_path is not None
    transport_failed_plus = plus_transport_failure_receipt_path is not None
    if complete_plus == transport_failed_plus:
        raise TrainTypedAssignmentCrossfitError(
            "provider selection requires exactly one Plus probe outcome"
        )
    pro_receipt: dict[str, Any] | None = None
    if complete_plus:
        assert plus_canary_report_path is not None
        _, plus_receipt = _read_complete_model_response_canary_v3(
            plus_canary_report_path,
            provider_label="plus",
        )
        if pro_canary_report_path is not None:
            raise TrainTypedAssignmentCrossfitError(
                "Pro is not authorized after any complete Plus response"
            )
        selected_provider_label = "plus"
        selected_response_receipt = plus_receipt
        probe_order = ["plus_complete_model_response"]
    else:
        if (
            plus_transport_failure_receipt_path is None
            or plus_failure_event_ledger_path is None
            or plus_expected_canary_report_path is None
            or pro_canary_report_path is None
        ):
            raise TrainTypedAssignmentCrossfitError(
                "Pro selection lacks the complete Plus failure evidence"
            )
        plus_receipt = _verify_plus_transport_failure_receipt_v3(
            receipt_path=plus_transport_failure_receipt_path,
            event_ledger_path=plus_failure_event_ledger_path,
            expected_canary_report_path=plus_expected_canary_report_path,
        )
        _, pro_receipt = _read_complete_model_response_canary_v3(
            pro_canary_report_path,
            provider_label="pro",
        )
        selected_provider_label = "pro"
        selected_response_receipt = pro_receipt
        probe_order = [
            "plus_transport_or_no_model_response_failure",
            "pro_complete_model_response",
        ]
    without_hash = {
        "selection_policy": TYPED_ASSIGNMENT_PROVIDER_POLICY,
        "selected_provider_label": selected_provider_label,
        "selected_provider_label_hash": stable_hash(
            {"provider_label": selected_provider_label}
        ),
        "probe_order": probe_order,
        "plus_probe_receipt": plus_receipt,
        "pro_model_response_receipt": pro_receipt,
        "selected_model_response_receipt": selected_response_receipt,
        "plus_semantic_acceptance_used_for_selection": False,
        "selection_completed_before_crossfit_task_calls": True,
        "crossfit_task_calls_before_selection": 0,
        "crossfit_model_calls_before_selection": 0,
        "selected_provider_fixed_for_complete_three_cell_batch": True,
        "mid_batch_provider_switch_authorized": False,
        "mid_batch_retry_authorized": False,
        "valid_failure_retry_authorized": False,
        "resampling_authorized": False,
        "secret_value_persisted": False,
        "raw_canary_content_persisted": False,
    }
    receipt = {**without_hash, "receipt_hash": stable_hash(without_hash)}
    destination = output_path.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("provider selection receipt already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _verify_provider_selection_receipt_v3(
    *,
    selection_receipt_path: Path,
    selected_canary_report_path: Path,
    provider_label: str,
    plus_canary_report_path: Path | None = None,
    plus_transport_failure_receipt_path: Path | None = None,
    plus_failure_event_ledger_path: Path | None = None,
    plus_expected_canary_report_path: Path | None = None,
) -> dict[str, Any]:
    if selection_receipt_path.is_symlink() or not selection_receipt_path.is_file():
        raise TrainTypedAssignmentCrossfitError(
            "provider selection receipt is unavailable"
        )
    try:
        raw = selection_receipt_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainTypedAssignmentCrossfitError(
            "provider selection receipt is unreadable"
        ) from exc
    if not isinstance(payload, dict):
        raise TrainTypedAssignmentCrossfitError(
            "provider selection receipt is malformed"
        )
    without_hash = dict(payload)
    receipt_hash = without_hash.pop("receipt_hash", None)
    _, selected_response_receipt = _read_complete_model_response_canary_v3(
        selected_canary_report_path,
        provider_label=provider_label,
    )
    plus_route_valid = False
    pro_route_valid = False
    plus_probe_receipt: dict[str, Any]
    if provider_label == "plus" and plus_canary_report_path is not None:
        _, plus_probe_receipt = _read_complete_model_response_canary_v3(
            plus_canary_report_path,
            provider_label="plus",
        )
        plus_route_valid = (
            plus_canary_report_path.resolve(strict=True)
            == selected_canary_report_path.resolve(strict=True)
            and payload.get("probe_order")
            == ["plus_complete_model_response"]
            and payload.get("pro_model_response_receipt") is None
        )
    elif provider_label == "pro" and all(
        value is not None
        for value in (
            plus_transport_failure_receipt_path,
            plus_failure_event_ledger_path,
            plus_expected_canary_report_path,
        )
    ):
        assert plus_transport_failure_receipt_path is not None
        assert plus_failure_event_ledger_path is not None
        assert plus_expected_canary_report_path is not None
        plus_probe_receipt = _verify_plus_transport_failure_receipt_v3(
            receipt_path=plus_transport_failure_receipt_path,
            event_ledger_path=plus_failure_event_ledger_path,
            expected_canary_report_path=plus_expected_canary_report_path,
        )
        pro_route_valid = payload.get("probe_order") == [
            "plus_transport_or_no_model_response_failure",
            "pro_complete_model_response",
        ]
    else:
        raise TrainTypedAssignmentCrossfitError(
            "provider selection source evidence is incomplete"
        )
    if (
        receipt_hash != stable_hash(without_hash)
        or payload.get("selection_policy")
        != TYPED_ASSIGNMENT_PROVIDER_POLICY
        or payload.get("selected_provider_label") != provider_label
        or payload.get("selected_provider_label_hash")
        != stable_hash({"provider_label": provider_label})
        or payload.get("plus_probe_receipt") != plus_probe_receipt
        or payload.get("selected_model_response_receipt")
        != selected_response_receipt
        or (
            provider_label == "pro"
            and payload.get("pro_model_response_receipt")
            != selected_response_receipt
        )
        or not (plus_route_valid or pro_route_valid)
        or payload.get("plus_semantic_acceptance_used_for_selection")
        is not False
        or payload.get("selection_completed_before_crossfit_task_calls")
        is not True
        or payload.get("crossfit_task_calls_before_selection") != 0
        or payload.get("crossfit_model_calls_before_selection") != 0
        or payload.get(
            "selected_provider_fixed_for_complete_three_cell_batch"
        )
        is not True
        or payload.get("mid_batch_provider_switch_authorized") is not False
        or payload.get("mid_batch_retry_authorized") is not False
        or payload.get("valid_failure_retry_authorized") is not False
        or payload.get("resampling_authorized") is not False
        or payload.get("secret_value_persisted") is not False
        or payload.get("raw_canary_content_persisted") is not False
    ):
        raise TrainTypedAssignmentCrossfitError(
            "provider selection receipt drifted"
        )
    return {
        "selection_receipt_file_sha256": hashlib.sha256(raw).hexdigest(),
        "selection_receipt_hash": receipt_hash,
        "selected_provider_label_hash": stable_hash(
            {"provider_label": provider_label}
        ),
        "selected_model_response_receipt": selected_response_receipt,
        "plus_transport_failure_before_pro_selection": provider_label == "pro",
        "semantic_acceptance_used_for_provider_selection": False,
        "selected_provider_fixed_for_complete_three_cell_batch": True,
        "mid_batch_retry_authorized": False,
        "raw_canary_content_persisted": False,
    }


def _runtime_class_hash() -> str:
    from .typed_assignment_integration_v3 import (
        TYPED_ASSIGNMENT_RUNTIME_CLASS_HASH,
    )

    return _require_sha256(
        TYPED_ASSIGNMENT_RUNTIME_CLASS_HASH,
        "typed-assignment runtime class hash",
    )


def implementation_receipt_v3(project_root: Path) -> dict[str, Any]:
    project = project_root.resolve(strict=True)
    runtime_rows = [
        {
            "relative_path": relative_path,
            "file_sha256": _sha256_regular_file(project / relative_path),
        }
        for relative_path in IMPLEMENTATION_RELATIVE_PATHS
    ]
    execution_rows = [
        {
            "relative_path": relative_path,
            "file_sha256": _sha256_regular_file(project / relative_path),
        }
        for relative_path in EXECUTION_IMPLEMENTATION_RELATIVE_PATHS
    ]
    runtime_class_hash = _runtime_class_hash()
    receipt_without_hash = {
        "implementation_binding_policy": (
            "exact_runtime_and_execution_file_sha256_plus_class_hash_v1"
        ),
        "runtime_class_hash": runtime_class_hash,
        "implementation_files": runtime_rows,
        "implementation_file_count": len(runtime_rows),
        "implementation_file_set_hash": stable_hash(
            {"implementation_files": runtime_rows}
        ),
        "execution_implementation_files": execution_rows,
        "execution_implementation_file_count": len(execution_rows),
        "execution_implementation_file_set_hash": stable_hash(
            {"execution_implementation_files": execution_rows}
        ),
        "raw_implementation_content_persisted": False,
    }
    return {
        **receipt_without_hash,
        "implementation_receipt_hash": stable_hash(receipt_without_hash),
    }


def _candidate_id(
    *,
    heldout_item_id: str,
    runtime_class_hash: str,
    implementation_file_set_hash: str,
) -> str:
    if heldout_item_id not in REGISTERED_HELDOUT_ITEM_IDS:
        raise TrainTypedAssignmentCrossfitError(
            "typed-assignment candidate used an unregistered TRAIN fold"
        )
    class_hash = _require_sha256(runtime_class_hash, "runtime class hash")
    file_set_hash = _require_sha256(
        implementation_file_set_hash,
        "implementation file-set hash",
    )
    suffix = heldout_item_id.rsplit("-", maxsplit=1)[-1]
    return (
        f"v320-train-loo-organize-{suffix}-typed-assignment-v3-"
        f"class-{class_hash}-implementation-{file_set_hash}"
    )


@dataclass(frozen=True)
class TypedAssignmentCrossfitCellV3:
    base: TrainItemOutCompileV2 = field(compare=False, repr=False)
    candidate: TrainCandidateSpecV2
    work: TrainCandidateWorkUnitV2

    @property
    def heldout_item_id(self) -> str:
        return self.base.fold.heldout_item_id

    def verify(self, implementation_receipt: Mapping[str, Any]) -> None:
        self.base.verify()
        if self.base.fold.contract_variant != TRACE_REFINED_CONTRACT_VARIANT:
            raise TrainTypedAssignmentCrossfitError(
                "typed assignment must extend the registered trace-refined cell"
            )
        expected_id = _candidate_id(
            heldout_item_id=self.heldout_item_id,
            runtime_class_hash=str(
                implementation_receipt.get("runtime_class_hash") or ""
            ),
            implementation_file_set_hash=str(
                implementation_receipt.get(
                    "implementation_file_set_hash"
                )
                or ""
            ),
        )
        reconstructed = TrainCandidateSpecV2.from_verified_bundle(
            candidate_id=expected_id,
            bundle=self.base.bundle,
            static_complexity=TYPED_ASSIGNMENT_STATIC_COMPLEXITY,
        )
        expected_work = TrainCandidateWorkUnitV2(
            candidate=reconstructed,
            baseline=self.base.work.baseline,
        )
        if (
            self.candidate.safe_payload() != reconstructed.safe_payload()
            or self.candidate.candidate_id != expected_id
            or self.work.work_unit_hash != expected_work.work_unit_hash
            or not self.work.candidate_active
            or self.work.baseline.item_id != self.heldout_item_id
            or self.candidate.static_complexity
            != TYPED_ASSIGNMENT_STATIC_COMPLEXITY
        ):
            raise TrainTypedAssignmentCrossfitError(
                "typed-assignment cell identity drifted"
            )

    def preregistration_payload(self) -> dict[str, Any]:
        return {
            "heldout_item_id": self.heldout_item_id,
            "heldout_item_id_hash": self.work.baseline.item_id_hash,
            "graph_source_item_ids": list(
                self.base.fold.graph_source_item_ids
            ),
            "base_trace_refined_contract_hash": (
                self.base.contract.contract_hash
            ),
            "base_compile_bundle_manifest_hash": self.base.bundle.manifest_hash,
            "base_trace_refined_candidate_hash": (
                self.base.candidate.candidate_hash
            ),
            "typed_assignment_candidate_hash": self.candidate.candidate_hash,
            "typed_assignment_work_unit_hash": self.work.work_unit_hash,
            "candidate_static_complexity": self.candidate.static_complexity,
            "expected_active_execution_count": 1,
            "expected_inactive_raw_replay_count": 37,
        }


@dataclass(frozen=True)
class TypedAssignmentCrossfitCompileV3:
    output_root: Path = field(compare=False)
    cells: tuple[TypedAssignmentCrossfitCellV3, ...]
    implementation_receipt: Mapping[str, Any]
    report: Mapping[str, Any]

    @property
    def report_path(self) -> Path:
        return self.output_root / COMPILE_REPORT_FILENAME

    @property
    def candidates(self) -> tuple[TrainCandidateSpecV2, ...]:
        return tuple(cell.candidate for cell in self.cells)

    @property
    def baseline_set(self):
        return self.cells[0].base.raw_projection.baseline_set

    @property
    def candidate_bundles_by_hash(
        self,
    ) -> dict[str, ExecutionContractCompileBundleV2]:
        return {
            cell.candidate.candidate_hash: cell.base.bundle
            for cell in self.cells
        }

    def verify(self) -> None:
        if (
            len(self.cells) != REGISTERED_CELL_COUNT
            or tuple(cell.heldout_item_id for cell in self.cells)
            != REGISTERED_HELDOUT_ITEM_IDS
            or self.implementation_receipt.get(
                "implementation_receipt_hash"
            )
            != stable_hash(
                {
                    key: value
                    for key, value in self.implementation_receipt.items()
                    if key != "implementation_receipt_hash"
                }
            )
        ):
            raise TrainTypedAssignmentCrossfitError(
                "typed-assignment crossfit compilation drifted"
            )
        baseline_hash = self.baseline_set.baseline_set_hash
        for cell in self.cells:
            cell.verify(self.implementation_receipt)
            if (
                cell.base.raw_projection.baseline_set.baseline_set_hash
                != baseline_hash
            ):
                raise TrainTypedAssignmentCrossfitError(
                    "typed-assignment cells use different frozen RAW baselines"
                )
        if (
            len({row.candidate_hash for row in self.candidates})
            != REGISTERED_CELL_COUNT
            or len({row.candidate_behavior_hash for row in self.candidates})
            != REGISTERED_CELL_COUNT
        ):
            raise TrainTypedAssignmentCrossfitError(
                "typed-assignment candidate cells are not unique"
            )
        try:
            persisted = json.loads(
                self.report_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TrainTypedAssignmentCrossfitError(
                "typed-assignment compile report is unreadable"
            ) from exc
        without_hash = dict(persisted)
        report_hash = without_hash.pop("report_hash", None)
        if (
            persisted != dict(self.report)
            or report_hash != stable_hash(without_hash)
            or persisted.get("compile_passed") is not True
            or persisted.get("model_calls") != 0
            or persisted.get("evaluator_calls") != 0
            or persisted.get("online_judge_calls") != 0
            or persisted.get("validation_accessed") is not False
            or persisted.get("test_accessed") is not False
            or persisted.get("globally_unbiased_crossfit") is not False
        ):
            raise TrainTypedAssignmentCrossfitError(
                "typed-assignment compile report drifted"
            )

    def preregistration_without_hash(self) -> dict[str, Any]:
        return {
            "preregistration_policy": TYPED_ASSIGNMENT_PREREGISTRATION_POLICY,
            "candidate_search_policy": TYPED_ASSIGNMENT_CROSSFIT_VERSION,
            "candidate_objective": TYPED_ASSIGNMENT_CLASSIFICATION_OBJECTIVE,
            "implementation_receipt": dict(self.implementation_receipt),
            "runtime_class_hash": self.implementation_receipt[
                "runtime_class_hash"
            ],
            "implementation_file_set_hash": self.implementation_receipt[
                "implementation_file_set_hash"
            ],
            "cells": [cell.preregistration_payload() for cell in self.cells],
            "registered_cell_count": REGISTERED_CELL_COUNT,
            "registered_heldout_item_ids": list(
                REGISTERED_HELDOUT_ITEM_IDS
            ),
            "base_trace_refined_static_complexity": 8,
            "typed_assignment_static_complexity_delta": (
                TYPED_ASSIGNMENT_STATIC_COMPLEXITY_DELTA
            ),
            "static_complexity_per_cell": (
                TYPED_ASSIGNMENT_STATIC_COMPLEXITY
            ),
            "static_complexity_formula": (
                "trace_refined_complexity_plus_content_evidence_extraction_"
                "plus_typed_assignment_plus_one_to_one_reconciliation"
            ),
            "expected_active_execution_count": (
                EXPECTED_ACTIVE_EXECUTION_COUNT
            ),
            "expected_inactive_raw_replay_count": (
                EXPECTED_INACTIVE_RAW_REPLAY_COUNT
            ),
            "maximum_concurrent_runner_calls": (
                MAXIMUM_CONCURRENT_RUNNER_CALLS
            ),
            "maximum_concurrent_model_calls": (
                MAXIMUM_CONCURRENT_RUNNER_CALLS
            ),
            "candidate_search_success_definition": {
                "minimum_valid_active_executions": (
                    MINIMUM_VALID_ACTIVE_EXECUTIONS
                ),
                "minimum_runtime_reconciliation_receipts": (
                    EXPECTED_ACTIVE_EXECUTION_COUNT
                ),
                "minimum_distinct_fold_recoveries": (
                    MINIMUM_DISTINCT_FOLD_RECOVERIES
                ),
                "valid_failures_are_not_retried": True,
                "failure_stops_this_representation": True,
                "this_is_not_a_promotion_gate": True,
            },
            "provider_policy": {
                "policy": TYPED_ASSIGNMENT_PROVIDER_POLICY,
                "initial_probe_provider_label": "plus",
                "complete_plus_model_response_always_selected": True,
                "plus_semantic_acceptance_used_for_selection": False,
                "pro_requires_verified_plus_transport_or_no_response_failure": (
                    True
                ),
                "pro_requires_complete_model_response": True,
                "selection_must_complete_before_crossfit_task_calls": True,
                "selected_provider_fixed_for_complete_three_cell_batch": (
                    True
                ),
                "credential_identity_attested": False,
                "mid_batch_provider_switch_authorized": False,
                "mid_batch_retry_authorized": False,
                "valid_failure_retry_authorized": False,
                "resampling_authorized": False,
            },
            "historically_informed_candidate_execution": True,
            "prior_outcome_design_used": True,
            "score_cohort_previously_observed": True,
            "new_cell_outcomes_observed_at_registration_time": False,
            "globally_unbiased_crossfit": False,
            "train_candidate_search_only": True,
            "offline_evaluation_only": True,
            "online_judge_calls": 0,
            "validation_access_authorized": False,
            "test_access_authorized": False,
            "promotion_authorized": False,
            "incumbent_authorized": False,
            "actual_authorized_after_manifest_commit": True,
            "raw_task_evaluator_or_outcome_content_persisted": False,
        }


def compile_v320_typed_assignment_crossfit_v3(
    *,
    project_root: Path,
    output_root: Path,
) -> TypedAssignmentCrossfitCompileV3:
    """Compile all three historically informed TRAIN folds without scoring."""

    project = project_root.resolve(strict=True)
    destination = output_root.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            "typed-assignment crossfit compile output already exists"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    try:
        implementation_receipt = implementation_receipt_v3(project)

        def compile_cell(heldout_item_id: str) -> TrainItemOutCompileV2:
            return compile_v320_train_item_out_crossfit_v2(
                project_root=project,
                output_root=destination / "cells" / heldout_item_id,
                fold=ORGANIZE_TRACE_REFINED_ITEM_OUT_FOLDS[
                    heldout_item_id
                ],
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=REGISTERED_CELL_COUNT,
            thread_name_prefix="typed-assignment-compile-v3",
        ) as executor:
            base_cells = tuple(
                executor.map(compile_cell, REGISTERED_HELDOUT_ITEM_IDS)
            )

        cells_list: list[TypedAssignmentCrossfitCellV3] = []
        for base in base_cells:
            candidate = TrainCandidateSpecV2.from_verified_bundle(
                candidate_id=_candidate_id(
                    heldout_item_id=base.fold.heldout_item_id,
                    runtime_class_hash=str(
                        implementation_receipt["runtime_class_hash"]
                    ),
                    implementation_file_set_hash=str(
                        implementation_receipt[
                            "implementation_file_set_hash"
                        ]
                    ),
                ),
                bundle=base.bundle,
                static_complexity=TYPED_ASSIGNMENT_STATIC_COMPLEXITY,
            )
            cells_list.append(
                TypedAssignmentCrossfitCellV3(
                    base=base,
                    candidate=candidate,
                    work=TrainCandidateWorkUnitV2(
                        candidate=candidate,
                        baseline=base.work.baseline,
                    ),
                )
            )
        cells = tuple(
            sorted(cells_list, key=lambda row: row.heldout_item_id)
        )
        report_without_hash = {
            "compile_policy": TYPED_ASSIGNMENT_COMPILE_POLICY,
            "compile_passed": True,
            "candidate_search_policy": TYPED_ASSIGNMENT_CROSSFIT_VERSION,
            "implementation_receipt": implementation_receipt,
            "cells": [cell.preregistration_payload() for cell in cells],
            "registered_cell_count": len(cells),
            "expected_active_execution_count": (
                EXPECTED_ACTIVE_EXECUTION_COUNT
            ),
            "expected_inactive_raw_replay_count": (
                EXPECTED_INACTIVE_RAW_REPLAY_COUNT
            ),
            "maximum_concurrent_compile_calls": REGISTERED_CELL_COUNT,
            "historically_informed_candidate_execution": True,
            "prior_outcome_design_used": True,
            "score_cohort_previously_observed": True,
            "globally_unbiased_crossfit": False,
            "train_candidate_search_only": True,
            "compile_is_non_scoring_diagnostic": True,
            "model_calls": 0,
            "evaluator_calls": 0,
            "online_judge_calls": 0,
            "network_calls": 0,
            "validation_accessed": False,
            "test_accessed": False,
            "freeze_or_promotion_authorized": False,
            "raw_task_evaluator_or_outcome_content_persisted": False,
        }
        report = {
            **report_without_hash,
            "report_hash": stable_hash(report_without_hash),
        }
        (destination / COMPILE_REPORT_FILENAME).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = TypedAssignmentCrossfitCompileV3(
            output_root=destination,
            cells=cells,
            implementation_receipt=implementation_receipt,
            report=report,
        )
        result.verify()
        return result
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        raise


def write_typed_assignment_preregistration_v3(
    *,
    project_root: Path,
    compilation: TypedAssignmentCrossfitCompileV3,
) -> Path:
    """Write the fixed manifest; actual remains blocked until it is committed."""

    project = project_root.resolve(strict=True)
    compilation.verify()
    path = project / PREREGISTRATION_RELATIVE_PATH
    if path.exists() or path.is_symlink():
        raise FileExistsError("typed-assignment preregistration already exists")
    payload_without_hash = compilation.preregistration_without_hash()
    payload = {
        **payload_without_hash,
        "manifest_hash": stable_hash(payload_without_hash),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _read_and_verify_preregistration(
    *,
    project_root: Path,
    compilation: TypedAssignmentCrossfitCompileV3,
) -> tuple[dict[str, Any], str]:
    path = project_root / PREREGISTRATION_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise TrainTypedAssignmentCrossfitError(
            "typed-assignment preregistration is unavailable"
        )
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainTypedAssignmentCrossfitError(
            "typed-assignment preregistration is unreadable"
        ) from exc
    if not isinstance(payload, dict):
        raise TrainTypedAssignmentCrossfitError(
            "typed-assignment preregistration is malformed"
        )
    without_hash = dict(payload)
    manifest_hash = without_hash.pop("manifest_hash", None)
    expected = compilation.preregistration_without_hash()
    if (
        without_hash != expected
        or manifest_hash != stable_hash(without_hash)
        or payload.get("new_cell_outcomes_observed_at_registration_time")
        is not False
        or payload.get("actual_authorized_after_manifest_commit") is not True
    ):
        raise TrainTypedAssignmentCrossfitError(
            "typed-assignment preregistration drifted"
        )
    return payload, hashlib.sha256(raw).hexdigest()


def _git_output(project_root: Path, arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise TrainTypedAssignmentCrossfitError(
            "typed-assignment preregistration is not committed"
        )
    return completed.stdout.strip()


def verify_preregistration_commit_v3(project_root: Path) -> dict[str, Any]:
    """Require the manifest and all bound implementation files at clean HEAD."""

    project = project_root.resolve(strict=True)
    bound_paths = (
        PREREGISTRATION_RELATIVE_PATH,
        *IMPLEMENTATION_RELATIVE_PATHS,
        *EXECUTION_IMPLEMENTATION_RELATIVE_PATHS,
    )
    _git_output(project, ("rev-parse", "--show-toplevel"))
    for relative_path in bound_paths:
        _git_output(
            project,
            ("ls-files", "--error-unmatch", "--", relative_path),
        )
    dirty = _git_output(
        project,
        ("status", "--porcelain", "--", *bound_paths),
    )
    if dirty:
        raise TrainTypedAssignmentCrossfitError(
            "typed-assignment preregistration or implementation is dirty"
        )
    commit = _git_output(
        project,
        (
            "log",
            "-1",
            "--format=%H",
            "--",
            PREREGISTRATION_RELATIVE_PATH,
        ),
    )
    _require_git_object_id(commit, "preregistration commit")
    head = _git_output(project, ("rev-parse", "HEAD"))
    _require_git_object_id(head, "actual HEAD commit")
    _git_output(project, ("merge-base", "--is-ancestor", commit, head))
    return {
        "preregistration_commit_hash": stable_hash({"commit": commit}),
        "actual_head_hash": stable_hash({"commit": head}),
        "manifest_and_implementation_tracked_at_clean_head": True,
        "raw_commit_ids_persisted": False,
    }


@dataclass(frozen=True)
class TypedAssignmentCrossfitActualV3:
    output_root: Path = field(compare=False)
    compilation: TypedAssignmentCrossfitCompileV3 = field(
        compare=False,
        repr=False,
    )
    ranking: TrainOutcomeRankingResultV2 = field(compare=False, repr=False)
    report: Mapping[str, Any]

    @property
    def report_path(self) -> Path:
        return self.output_root / ACTUAL_REPORT_FILENAME

    def verify(self) -> None:
        self.compilation.verify()
        self.ranking.verify()
        try:
            persisted = json.loads(
                self.report_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TrainTypedAssignmentCrossfitError(
                "typed-assignment actual report is unreadable"
            ) from exc
        without_hash = dict(persisted)
        report_hash = without_hash.pop("report_hash", None)
        runtime_evidence_rows = persisted.get(
            "typed_assignment_runtime_evidence"
        )
        active_work_hashes = {
            cell.work.work_unit_hash for cell in self.compilation.cells
        }
        valid_count = sum(
            result.offline_evaluation.evaluation_valid
            for result in self.ranking.run_results
        )
        recovery_count = sum(
            row.recovery
            for row in self.ranking.outcomes
            if row.work_unit_hash in active_work_hashes
        )
        expected_search_success = (
            valid_count >= MINIMUM_VALID_ACTIVE_EXECUTIONS
            and recovery_count >= MINIMUM_DISTINCT_FOLD_RECOVERIES
        )
        run_request_by_work_hash = {
            result.work_unit_hash: result.observation.request.request_hash
            for result in self.ranking.run_results
        }
        runtime_class_hash = self.compilation.implementation_receipt[
            "runtime_class_hash"
        ]
        runtime_evidence_rows_valid = True
        if isinstance(runtime_evidence_rows, list):
            for row in runtime_evidence_rows:
                if not isinstance(row, Mapping):
                    runtime_evidence_rows_valid = False
                    break
                safe_payload = {
                    key: value
                    for key, value in row.items()
                    if key not in {"work_unit_hash", "evidence_hash"}
                }
                if (
                    row.get("evidence_hash") != stable_hash(safe_payload)
                    or row.get("runtime_class_hash") != runtime_class_hash
                    or run_request_by_work_hash.get(
                        str(row.get("work_unit_hash") or "")
                    )
                    != row.get("request_hash")
                    or row.get("agent_wrote_plan_only") is not True
                    or row.get("harness_applied_plan") is not True
                    or row.get("post_apply_reconciliation_passed")
                    is not True
                    or row.get(
                        "reconciliation_completed_before_verifier_invocation"
                    )
                    is not True
                    or row.get("verifier_invoked_at_receipt_time") is not False
                    or row.get("verifier_materialized_at_receipt_time")
                    is not False
                    or row.get("validation_or_test_content_accessed")
                    is not False
                    or row.get("online_judge_calls") != 0
                    or row.get("raw_document_content_persisted_host_side")
                    is not False
                    or not _runtime_receipt_bodies_are_bound(row)
                ):
                    runtime_evidence_rows_valid = False
                    break
        else:
            runtime_evidence_rows_valid = False
        if (
            persisted != dict(self.report)
            or report_hash != stable_hash(without_hash)
            or persisted.get("execution_completed") is not True
            or persisted.get("ranking_hash") != self.ranking.ranking_hash
            or persisted.get("active_execution_count")
            != EXPECTED_ACTIVE_EXECUTION_COUNT
            or persisted.get("inactive_raw_replay_count")
            != EXPECTED_INACTIVE_RAW_REPLAY_COUNT
            or persisted.get("offline_evaluation_only") is not True
            or persisted.get("online_judge_calls") != 0
            or persisted.get("validation_accessed") is not False
            or persisted.get("test_accessed") is not False
            or persisted.get("globally_unbiased_crossfit") is not False
            or persisted.get("incumbent_authorized") is not False
            or persisted.get("provider_policy")
            != TYPED_ASSIGNMENT_PROVIDER_POLICY
            or not isinstance(runtime_evidence_rows, list)
            or not runtime_evidence_rows_valid
            or len(runtime_evidence_rows)
            != EXPECTED_ACTIVE_EXECUTION_COUNT
            or persisted.get("typed_assignment_runtime_evidence_count")
            != EXPECTED_ACTIVE_EXECUTION_COUNT
            or persisted.get(
                "typed_assignment_runtime_evidence_set_hash"
            )
            != stable_hash({"runtime_evidence": runtime_evidence_rows})
            or {
                str(row.get("work_unit_hash") or "")
                for row in runtime_evidence_rows
                if isinstance(row, Mapping)
            }
            != active_work_hashes
            or persisted.get("valid_active_execution_count") != valid_count
            or persisted.get("distinct_fold_recovery_count")
            != recovery_count
            or persisted.get("candidate_search_success")
            is not expected_search_success
            or persisted.get(
                "candidate_class_eligible_for_fresh_development"
            )
            is not expected_search_success
        ):
            raise TrainTypedAssignmentCrossfitError(
                "typed-assignment actual report drifted"
            )


def run_v320_typed_assignment_crossfit_actual_v3(
    *,
    project_root: Path,
    output_root: Path,
    canary_report_path: Path,
    provider_selection_receipt_path: Path,
    provider_label: str,
    plus_canary_report_path: Path | None = None,
    plus_transport_failure_receipt_path: Path | None = None,
    plus_failure_event_ledger_path: Path | None = None,
    plus_expected_canary_report_path: Path | None = None,
    task_input_cache_root: Path | None = None,
) -> TypedAssignmentCrossfitActualV3:
    """Run one preregistered, pre-task-selected three-cell provider batch."""

    if provider_label not in {"plus", "pro"}:
        raise TrainTypedAssignmentCrossfitError(
            "typed-assignment provider selection is not registered"
        )
    project = project_root.resolve(strict=True)
    destination = output_root.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("typed-assignment actual output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    provider_selection_verified = False
    try:
        protocol = PaperProtocol.read(project / V320_PROTOCOL_RELATIVE_PATH)
        if (
            protocol.payload.get("protocol_version") != "3.20.0"
            or protocol.payload.get("model") != V320_MODEL
            or protocol.payload.get("max_steps") != 100
        ):
            raise TrainTypedAssignmentCrossfitError(
                "typed-assignment execution protocol drifted"
            )
        provider_selection_receipt = (
            _verify_provider_selection_receipt_v3(
                selection_receipt_path=(
                    provider_selection_receipt_path
                ),
                plus_canary_report_path=plus_canary_report_path,
                plus_transport_failure_receipt_path=(
                    plus_transport_failure_receipt_path
                ),
                plus_failure_event_ledger_path=(
                    plus_failure_event_ledger_path
                ),
                plus_expected_canary_report_path=(
                    plus_expected_canary_report_path
                ),
                selected_canary_report_path=canary_report_path,
                provider_label=provider_label,
            )
        )
        provider_selection_verified = True
        _configure_environment(protocol)
        canary = provider_selection_receipt[
            "selected_model_response_receipt"
        ]
        manifest = SplitManifest.read(project / V320_MANIFEST_RELATIVE_PATH)
        if manifest.manifest_hash != V320_MANIFEST_HASH:
            raise TrainTypedAssignmentCrossfitError(
                "typed-assignment execution manifest drifted"
            )
        compilation = compile_v320_typed_assignment_crossfit_v3(
            project_root=project,
            output_root=destination / "compile_batch",
        )
        preregistration, preregistration_file_sha256 = (
            _read_and_verify_preregistration(
                project_root=project,
                compilation=compilation,
            )
        )
        commit_receipt = verify_preregistration_commit_v3(project)
        event_sink = JsonlEventSink(destination / EXECUTION_EVENTS_FILENAME)
        active_item_hashes = {
            cell.work.baseline.item_id_hash for cell in compilation.cells
        }
        assets = _prepare_scoped_runtime_assets(
            project_root=project,
            destination=destination,
            protocol=protocol,
            manifest=manifest,
            baseline_set=compilation.baseline_set,
            active_item_hashes=active_item_hashes,
            expected_active_item_count=REGISTERED_CELL_COUNT,
            preflight_policy=(
                f"{ASSET_PREFLIGHT_POLICY}_typed_assignment_batch_v3"
            ),
            event_sink=event_sink,
            task_input_cache_root=task_input_cache_root,
        )
        benchmark_root = (
            project / SKILLLEARN_BENCHMARK_RELATIVE_ROOT
        ).resolve(strict=True)
        execution = protocol.payload["execution"]
        assert isinstance(execution, Mapping)
        trials_root = destination / "worker_state"

        from .typed_assignment_integration_v3 import (
            TypedAssignmentExecutionContractSubprocessBackendV3,
        )

        typed_backends: list[
            tuple[
                str,
                TypedAssignmentExecutionContractSubprocessBackendV3,
            ]
        ] = []
        typed_backends_lock = threading.Lock()

        def backend_factory(
            work: TrainCandidateWorkUnitV2,
            bundle: ExecutionContractCompileBundleV2,
        ) -> ExecutionContractSubprocessBackendV2:
            baseline_request = work.baseline.observation.request
            backend = TypedAssignmentExecutionContractSubprocessBackendV3(
                benchmark_root,
                agent_id=baseline_request.agent_id,
                model=baseline_request.model,
                max_steps=baseline_request.max_steps,
                provider_mode=str(protocol.payload["trial_provider_mode"]),
                trials_dir=trials_root / work.work_unit_hash,
                record_upstream=False,
                prebuilt_cache=assets.prebuilt_cache,
                offline_verifier_cache=assets.offline_cache,
                provider_circuit=assets.provider_circuit,
                model_inference_limiter=assets.model_limiter,
                train_action_design_policy=str(
                    execution["train_action_design_policy"]
                ),
                codex_agent_execution_policy=(
                    protocol.codex_agent_execution_policy
                ),
                event_sink=event_sink,
                execution_contract_bundle=bundle,
            )
            with typed_backends_lock:
                typed_backends.append((work.work_unit_hash, backend))
            return backend

        production_runner = ProductionTrainCandidateRunnerV2(
            baseline_set=compilation.baseline_set,
            candidate_bundles=compilation.candidate_bundles_by_hash,
            backend_factory=backend_factory,
            trace_prefix=(
                "v320-train-typed-assignment-crossfit-"
                f"{provider_label}-v3"
            ),
        )
        ranking = TrainOutcomeRankerV2(
            max_workers=MAXIMUM_CONCURRENT_RUNNER_CALLS
        ).rank(
            baseline_set=compilation.baseline_set,
            candidates=compilation.candidates,
            runner=production_runner,
        )
        ranking.verify()
        if (
            len(ranking.run_results) != EXPECTED_ACTIVE_EXECUTION_COUNT
            or len(ranking.replay_receipts)
            != EXPECTED_INACTIVE_RAW_REPLAY_COUNT
            or production_runner.retained_backend_count
            != EXPECTED_ACTIVE_EXECUTION_COUNT
            or len(production_runner.backend_instance_hashes)
            != EXPECTED_ACTIVE_EXECUTION_COUNT
            or ranking.maximum_concurrent_runner_calls
            > MAXIMUM_CONCURRENT_RUNNER_CALLS
            or assets.model_limiter.maximum_active
            > MAXIMUM_CONCURRENT_RUNNER_CALLS
            or assets.model_limiter.maximum_active > MODEL_INFERENCE_SLOTS
            or assets.provider_circuit.error_type is not None
        ):
            raise TrainTypedAssignmentCrossfitError(
                "typed-assignment batch execution boundary drifted"
            )
        active_outcomes = tuple(
            row
            for row in ranking.outcomes
            if row.work_unit_hash
            in {cell.work.work_unit_hash for cell in compilation.cells}
        )
        valid_count = sum(
            result.offline_evaluation.evaluation_valid
            for result in ranking.run_results
        )
        recovery_count = sum(row.recovery for row in active_outcomes)
        typed_runtime_evidence = tuple(
            sorted(
                (
                    (work_unit_hash, evidence)
                    for work_unit_hash, backend in typed_backends
                    for evidence in backend.typed_assignment_evidence
                ),
                key=lambda row: row[0],
            )
        )
        run_request_by_work_hash = {
            result.work_unit_hash: result.observation.request.request_hash
            for result in ranking.run_results
        }
        if (
            len(typed_backends) != EXPECTED_ACTIVE_EXECUTION_COUNT
            or {
                work_unit_hash for work_unit_hash, _ in typed_backends
            }
            != {cell.work.work_unit_hash for cell in compilation.cells}
            or any(
                len(backend.typed_assignment_evidence) != 1
                for _, backend in typed_backends
            )
            or len(
                {
                    evidence.request_hash
                    for _, evidence in typed_runtime_evidence
                }
            )
            != EXPECTED_ACTIVE_EXECUTION_COUNT
            or any(
                run_request_by_work_hash.get(work_unit_hash)
                != evidence.request_hash
                for work_unit_hash, evidence in typed_runtime_evidence
            )
            or any(
                evidence.runtime_class_hash
                != compilation.implementation_receipt["runtime_class_hash"]
                for _, evidence in typed_runtime_evidence
            )
        ):
            raise TrainTypedAssignmentCrossfitError(
                "typed-assignment runtime evidence boundary drifted"
            )
        runtime_evidence_rows = [
            {
                "work_unit_hash": work_unit_hash,
                **row.safe_payload(),
                "evidence_hash": row.evidence_hash,
            }
            for work_unit_hash, row in typed_runtime_evidence
        ]
        search_success = (
            valid_count >= MINIMUM_VALID_ACTIVE_EXECUTIONS
            and recovery_count >= MINIMUM_DISTINCT_FOLD_RECOVERIES
        )
        report_without_hash: dict[str, Any] = {
            "execution_policy": TYPED_ASSIGNMENT_ACTUAL_POLICY,
            "execution_completed": True,
            "provider_policy": TYPED_ASSIGNMENT_PROVIDER_POLICY,
            "provider_canary": canary,
            "provider_selection_receipt": provider_selection_receipt,
            "provider_label_hash": stable_hash(
                {"provider_label": provider_label}
            ),
            "preregistration_manifest_hash": preregistration[
                "manifest_hash"
            ],
            "preregistration_file_sha256": (
                preregistration_file_sha256
            ),
            "preregistration_commit_receipt": commit_receipt,
            "compile_report_hash": compilation.report["report_hash"],
            "implementation_receipt": dict(
                compilation.implementation_receipt
            ),
            "asset_preflight_report_hash": assets.preflight_report[
                "report_hash"
            ],
            "manifest_hash": manifest.manifest_hash,
            "evaluator_epoch": V320_EVALUATOR_EPOCH,
            "model_hash": stable_hash({"model": V320_MODEL}),
            "candidate_hashes": [
                row.candidate_hash for row in compilation.candidates
            ],
            "work_unit_hashes": [
                cell.work.work_unit_hash for cell in compilation.cells
            ],
            "ranking": ranking.to_dict(),
            "ranking_hash": ranking.ranking_hash,
            "outcomes": [row.safe_payload() for row in ranking.outcomes],
            "outcome_set_hash": ranking.outcome_set_hash,
            "run_receipts": [
                row.safe_payload() for row in ranking.run_results
            ],
            "replay_receipts": [
                row.safe_payload() for row in ranking.replay_receipts
            ],
            "active_execution_count": len(ranking.run_results),
            "inactive_raw_replay_count": len(ranking.replay_receipts),
            "maximum_concurrent_runner_calls": (
                ranking.maximum_concurrent_runner_calls
            ),
            "maximum_concurrent_model_calls": (
                assets.model_limiter.maximum_active
            ),
            "typed_assignment_runtime_evidence": runtime_evidence_rows,
            "typed_assignment_runtime_evidence_count": len(
                runtime_evidence_rows
            ),
            "typed_assignment_runtime_evidence_set_hash": stable_hash(
                {"runtime_evidence": runtime_evidence_rows}
            ),
            "valid_active_execution_count": valid_count,
            "distinct_fold_recovery_count": recovery_count,
            "minimum_valid_active_executions": (
                MINIMUM_VALID_ACTIVE_EXECUTIONS
            ),
            "minimum_distinct_fold_recoveries": (
                MINIMUM_DISTINCT_FOLD_RECOVERIES
            ),
            "candidate_search_success": search_success,
            "candidate_class_eligible_for_fresh_development": search_success,
            "valid_failures_retried": False,
            "pro_selected_after_plus_transport_or_no_response_failure": (
                provider_label == "pro"
            ),
            "selected_provider_fixed_for_complete_three_cell_batch": True,
            "mid_batch_provider_switch_used": False,
            "mid_batch_retry_used": False,
            "historically_informed_candidate_execution": True,
            "prior_outcome_design_used": True,
            "score_cohort_previously_observed": True,
            "globally_unbiased_crossfit": False,
            "train_candidate_search_only": True,
            "offline_evaluation_only": True,
            "online_judge_calls": 0,
            "network_fallback_used": False,
            "validation_accessed": False,
            "test_accessed": False,
            "promotion_gate_applied": False,
            "promotion_authorized": False,
            "incumbent_authorized": False,
            "raw_candidate_trial_artifacts_persisted": True,
            "raw_candidate_trial_artifacts_embedded_in_report": False,
            "secret_value_persisted": False,
        }
        report = {
            **report_without_hash,
            "report_hash": stable_hash(report_without_hash),
        }
        (destination / ACTUAL_REPORT_FILENAME).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = TypedAssignmentCrossfitActualV3(
            output_root=destination,
            compilation=compilation,
            ranking=ranking,
            report=report,
        )
        result.verify()
        return result
    except Exception as exc:
        failure_without_hash = {
            "execution_policy": TYPED_ASSIGNMENT_ACTUAL_POLICY,
            "execution_completed": False,
            "selected_provider_label_hash": stable_hash(
                {"provider_label": provider_label}
            ),
            "error_type": type(exc).__name__,
            "error_message_hash": stable_hash({"message": str(exc)}),
            "selected_provider_fixed_before_batch": (
                provider_selection_verified
            ),
            "mid_batch_provider_switch_authorized": False,
            "mid_batch_retry_authorized": False,
            "valid_failure_retry_authorized": False,
            "full_batch_resampling_authorized": False,
            "raw_error_persisted": False,
            "secret_value_persisted": False,
        }
        failure = {
            **failure_without_hash,
            "report_hash": stable_hash(failure_without_hash),
        }
        (destination / FAILURE_REPORT_FILENAME).write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Preregister or run the three-cell typed-assignment TRAIN "
            "crossfit batch."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preregister = subparsers.add_parser("preregister")
    preregister.add_argument("--project-root", type=Path, default=Path("."))
    preregister.add_argument("--output-root", type=Path, required=True)
    actual = subparsers.add_parser("actual")
    actual.add_argument("--project-root", type=Path, default=Path("."))
    actual.add_argument("--output-root", type=Path, required=True)
    actual.add_argument("--canary-report", type=Path, required=True)
    actual.add_argument(
        "--plus-canary-report",
        type=Path,
    )
    actual.add_argument("--plus-transport-failure-receipt", type=Path)
    actual.add_argument("--plus-failure-event-ledger", type=Path)
    actual.add_argument("--plus-expected-canary-report", type=Path)
    actual.add_argument(
        "--provider-selection-receipt",
        type=Path,
        required=True,
    )
    actual.add_argument(
        "--provider-label",
        choices=("plus", "pro"),
        required=True,
    )
    actual.add_argument("--task-input-cache-root", type=Path)
    args = parser.parse_args()
    if args.command == "preregister":
        compilation = compile_v320_typed_assignment_crossfit_v3(
            project_root=args.project_root,
            output_root=args.output_root,
        )
        path = write_typed_assignment_preregistration_v3(
            project_root=args.project_root,
            compilation=compilation,
        )
        output = {
            "compile_passed": True,
            "compile_report_hash": compilation.report["report_hash"],
            "manifest_path": str(path),
            "actual_authorized_before_manifest_commit": False,
            "model_calls": 0,
            "online_judge_calls": 0,
        }
    else:
        result = run_v320_typed_assignment_crossfit_actual_v3(
            project_root=args.project_root,
            output_root=args.output_root,
            canary_report_path=args.canary_report,
            provider_selection_receipt_path=(
                args.provider_selection_receipt
            ),
            provider_label=args.provider_label,
            plus_canary_report_path=args.plus_canary_report,
            plus_transport_failure_receipt_path=(
                args.plus_transport_failure_receipt
            ),
            plus_failure_event_ledger_path=(
                args.plus_failure_event_ledger
            ),
            plus_expected_canary_report_path=(
                args.plus_expected_canary_report
            ),
            task_input_cache_root=args.task_input_cache_root,
        )
        output = {
            "execution_completed": True,
            "report_hash": result.report["report_hash"],
            "ranking_hash": result.ranking.ranking_hash,
            "candidate_search_success": result.report[
                "candidate_search_success"
            ],
            "active_execution_count": EXPECTED_ACTIVE_EXECUTION_COUNT,
            "inactive_raw_replay_count": (
                EXPECTED_INACTIVE_RAW_REPLAY_COUNT
            ),
            "online_judge_calls": 0,
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
