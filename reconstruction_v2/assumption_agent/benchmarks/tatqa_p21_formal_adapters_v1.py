"""Fail-closed production boundaries for the frozen TAT-QA P21 controller.

The two adapters in this module deliberately keep custody and runtime
capabilities separate.  :class:`TrustedAcquisitionAdapter` is the only object
that may open acquisition views and late-label packs.  It hands complete,
label-free items to :class:`ProductionRuntimeAdapter`, which invokes injected
offline workers and returns only content-minimized tensors to the lifecycle
controller.

No formal TAT-QA source path or source loader exists here.  Worker launch is
also injected: the production launcher can isolate a subprocess without
making filesystem or process policy part of the pure custody boundary.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
from typing import Any, Mapping, Protocol, Sequence

from assumption_agent.benchmarks import tatqa_p21_acquisition_v1 as acquisition
from assumption_agent.benchmarks import tatqa_p21_formal_controller_v1 as controller
from assumption_agent.benchmarks import tatqa_p21_label_free_runtime_v1 as features
from assumption_agent.benchmarks import tatqa_p21_typed_evaluator_core_v1 as core
from replication_runtime.tatqa_p21_v1 import hipporag_contract
from replication_runtime.tatqa_p21_v1 import typed_plan_contract


VERSION = "tatqa_p21_formal_adapters_v1"
RUNTIME_FINGERPRINT_SCHEMA = "tatqa_p21_composite_runtime_fingerprint_v1"
PRODUCTION_CANARY_SCHEMA = "tatqa_p21_public_synthetic_production_canary_v1"
RUNTIME_FINGERPRINT_SELF_FIELD = "self_sha256"
PRODUCTION_CANARY_SELF_FIELD = "self_sha256"
STANDARD_EXECUTOR_CAP = 16
ACTUAL_HIPPO_INFERENCE_CAP = 8
OFFLINE_EVIDENCE_NAMES = frozenset(
    {
        "A_form_fit",
        "F_search_policy_freeze",
        "A_hold_score",
        "M_search_score",
    }
)
_OFFLINE_EVIDENCE_SCHEMAS = {
    "A_form_fit": f"{controller.VERSION}_A_form_evaluator_fit_evidence_v1",
    "F_search_policy_freeze": f"{controller.VERSION}_F_search_policy_freeze_v1",
    "A_hold_score": f"{controller.VERSION}_A_hold_offline_score_v1",
    "M_search_score": f"{controller.VERSION}_M_search_offline_score_v1",
}
SYSTEMD_NETWORK_PROPERTIES = ("IPAddressDeny=any", "RestrictAddressFamilies=AF_UNIX")
SYSTEMD_FILESYSTEM_ISOLATION = (
    "systemd_InaccessiblePaths_official_source_and_acquisition_v1"
)
TYPED_PLAN_TRANSPORT_SCHEMA = (
    "tatqa_p21_formal_runtime_v1_typed_plan_transport_receipt_v1"
)
HIPPO_TRANSPORT_SCHEMA = "tatqa_p21_formal_runtime_v1_hippo_transport_receipt_v1"
SYSTEMD_UNIT_CLOSURE_SCHEMA = (
    "tatqa_p21_formal_runtime_v1_systemd_unit_closure_v1"
)
SYSTEMD_START_POLICY_SCHEMA = (
    "tatqa_p21_formal_runtime_v1_systemd_start_policy_v1"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PRIVATE_BINDING_KEYS = frozenset(
    {"filename", "file_sha256", "semantic_sha256", "size_bytes", "mode"}
)
_VIEW_KEYS = frozenset(
    {
        "access_state",
        "block",
        "block_view_sha256",
        "item_count",
        "items",
        "late_fields_included",
        "schema",
        "version",
    }
)
_VIEW_ITEM_KEYS = frozenset(
    {"canonical_units", "item_commitment_sha256", "ordinal", "question"}
)
_UNIT_KEYS = frozenset({"text", "unit_id"})
_LABEL_KEYS = frozenset(
    {
        "access_state",
        "block",
        "block_view_sha256",
        "item_count",
        "items",
        "label_pack_sha256",
        "schema",
        "version",
    }
)
_LABEL_ITEM_KEYS = frozenset(
    {"family", "gold_unit_ids", "item_commitment_sha256", "ordinal"}
)
_PUBLIC_RECEIPT_KEYS = frozenset(
    {
        "F_search_label_pack_created",
        "M_search_view_and_labels_presealed",
        "acquisition_receipt_sha256",
        "aggregate_qualification",
        "fixed_block_counts",
        "fixed_per_family_quota",
        "implementation_freeze_self_sha256",
        "label_file_count",
        "ledger_file_count",
        "network_download_online_evaluator_or_model_calls",
        "private_file_bindings",
        "retry_replay_resample_or_smaller_blocks",
        "schema",
        "selected_context_count",
        "selected_question_count",
        "selection_secret_commitment_sha256",
        "selection_secret_persisted_publicly",
        "selection_secret_size_bytes",
        "source_custody_self_sha256",
        "source_download_receipt_self_sha256",
        "source_item_or_identifier_persisted_publicly",
        "status",
        "study_design_self_sha256",
        "version",
        "view_file_count",
    }
)


class TatqaP21FormalAdapterError(RuntimeError):
    """A custody, worker, or durable-receipt invariant failed closed."""


class TypedPlanBatchRunner(Protocol):
    """Run one local typed-plan worker over a canonical request envelope."""

    def __call__(self, block: str, canonical_input: bytes) -> bytes:
        """Return canonical :mod:`typed_plan_contract` output bytes."""

    def transport_receipt(self, block: str) -> Mapping[str, object]:
        """Return the exact systemd/PID receipt for the completed block."""

    def abort_all_workers(self) -> Sequence[Mapping[str, object]]:
        """Stop every named worker unit and return closure receipts."""

    def verify_all_workers_closed(self) -> Sequence[Mapping[str, object]]:
        """Re-query systemd and fail unless every owned unit is closed."""


class HippoRunner(Protocol):
    """Run one isolated official-Hippo subprocess for one item."""

    def __call__(
        self,
        block: str,
        item_commitment_sha256: str,
        canonical_input: bytes,
    ) -> bytes:
        """Return canonical :mod:`hipporag_contract` output bytes."""

    def transport_receipt(
        self, block: str, item_commitment_sha256: str
    ) -> Mapping[str, object]:
        """Return the exact systemd/PID receipt for the completed item."""

    def abort_all_workers(self) -> Sequence[Mapping[str, object]]:
        """Stop every named worker unit and return closure receipts."""

    def verify_all_workers_closed(self) -> Sequence[Mapping[str, object]]:
        """Re-query systemd and fail unless every owned unit is closed."""


@dataclass(frozen=True)
class RuntimeReceiptPaths:
    runtime_fingerprint: Path
    production_canary: Path


@dataclass(frozen=True)
class _ReceiptBundle:
    runtime_fingerprint_sha256: str
    runtime_fingerprint_file_sha256: str
    production_canary_sha256: str
    production_canary_file_sha256: str
    p0_behavior_sha256: str
    p1_behavior_sha256: str


@dataclass(frozen=True)
class _PreparedBlock:
    block_view: controller.BlockView
    runtime_items: Mapping[str, features.LabelFreeRuntimeItem]
    compiled_items: Mapping[str, features.CompiledLabelFreeItem]
    preparation_receipt_sha256: str
    typed_plan_input_sha256: str
    typed_plan_output_sha256: str
    typed_plan_transport_receipt: Mapping[str, object]
    typed_plan_transport_receipt_sha256: str
    typed_plan_worker_receipt_sha256: str
    typed_plan_worker_pid: int


@dataclass
class _InferencePreparation:
    """Mutable scheduler state; every public receipt is copied canonically."""

    block: str
    qwen_executor: concurrent.futures.ThreadPoolExecutor
    hippo_executor: concurrent.futures.ThreadPoolExecutor | None
    start_barrier: threading.Barrier | None
    qwen_future: concurrent.futures.Future[bytes] | None = None
    hippo_futures: dict[str, concurrent.futures.Future[bytes]] | None = None
    hippo_inputs: dict[str, tuple[bytes, str]] | None = None
    submitted_before_first_join: int = 0
    expected_submission_count: int = 0
    first_join_started: bool = False
    qwen_live: bool = False
    hippo_live: int = 0
    hippo_peak: int = 0
    overlap_observed: bool = False
    qwen_consumed: bool = False
    hippo_consumed: set[str] | None = None
    shutdown_started: bool = False
    executors_closed: bool = False
    shutdown_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False
    )

    def __post_init__(self) -> None:
        if self.hippo_futures is None:
            self.hippo_futures = {}
        if self.hippo_inputs is None:
            self.hippo_inputs = {}
        if self.hippo_consumed is None:
            self.hippo_consumed = set()


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise TatqaP21FormalAdapterError("value is not canonical JSON") from exc


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TatqaP21FormalAdapterError(f"{field} must be a lowercase SHA-256")
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("nonfinite JSON constant")


def _strict_json(raw: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TatqaP21FormalAdapterError(f"{field} is not strict JSON") from exc
    if not isinstance(value, dict) or _canonical_bytes(value) != raw:
        raise TatqaP21FormalAdapterError(f"{field} is not canonical JSON")
    return value


def _verify_self_hash(
    value: Mapping[str, Any], *, field: str, schema: str | None = None
) -> str:
    if schema is not None and value.get("schema") != schema:
        raise TatqaP21FormalAdapterError(f"{field} schema drifted")
    claimed = _require_sha256(value.get(field), field)
    body = dict(value)
    del body[field]
    if _semantic_hash(body) != claimed:
        raise TatqaP21FormalAdapterError(f"{field} self binding drifted")
    return claimed


def _read_regular(
    path: Path, *, field: str, expected_mode: int | None = None
) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TatqaP21FormalAdapterError(f"{field} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise TatqaP21FormalAdapterError(f"{field} is not a regular file")
    if expected_mode is not None and stat.S_IMODE(metadata.st_mode) != expected_mode:
        raise TatqaP21FormalAdapterError(f"{field} mode drifted")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise TatqaP21FormalAdapterError(f"{field} changed during open")
            raw = handle.read()
    except OSError as exc:
        raise TatqaP21FormalAdapterError(f"{field} cannot be read safely") from exc
    if len(raw) != metadata.st_size:
        raise TatqaP21FormalAdapterError(f"{field} size changed during read")
    return raw


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path: Path, *, mode: int = 0o700) -> None:
    try:
        path.mkdir(mode=mode)
    except FileExistsError as exc:
        raise TatqaP21FormalAdapterError("durable one-shot path already exists") from exc
    _fsync_directory(path.parent)


def _write_exclusive_verified(path: Path, value: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except OSError as exc:
        raise TatqaP21FormalAdapterError("exclusive durable receipt write failed") from exc
    reopened = _read_regular(path, field="durable receipt")
    if reopened != raw:
        raise TatqaP21FormalAdapterError("durable receipt reopen verification failed")
    return hashlib.sha256(raw).hexdigest()


def _receipt_body(value: Mapping[str, Any], self_field: str) -> dict[str, Any]:
    body = dict(value)
    body.pop(self_field, None)
    return body


def _load_public_runtime_receipts(paths: RuntimeReceiptPaths) -> _ReceiptBundle:
    fingerprint_raw = _read_regular(
        Path(paths.runtime_fingerprint), field="runtime fingerprint receipt"
    )
    fingerprint = _strict_json(fingerprint_raw, field="runtime fingerprint receipt")
    fingerprint_sha = _verify_self_hash(
        fingerprint,
        field=RUNTIME_FINGERPRINT_SELF_FIELD,
        schema=RUNTIME_FINGERPRINT_SCHEMA,
    )
    required_fingerprint = {
        "external_network_calls": 0,
        "api_or_online_evaluator_calls": 0,
        "retry_replay_resample_provider_switch": 0,
        "controller_or_worker_source_reads": 0,
        "controller_or_worker_label_reads": 0,
        "hippo_concurrency_cap": controller.HIPPO_CONCURRENCY_CAP,
        "maximum_cpu_threads_per_hippo_process": (
            controller.HIPPO_CPU_THREADS_PER_PROCESS
        ),
    }
    if (
        fingerprint.get("status") != "verified_before_formal_source_open"
        or fingerprint.get("study_design_self_sha256")
        != acquisition.DESIGN_SELF_SHA256
        or fingerprint.get("filesystem_isolation") != SYSTEMD_FILESYSTEM_ISOLATION
        or fingerprint.get("formal_source_opened") is not False
        or fingerprint.get(
            "source_identifiers_answers_families_mappings_or_labels_present"
        )
        is not False
        or any(
            fingerprint.get(key) != expected
            for key, expected in required_fingerprint.items()
        )
    ):
        raise TatqaP21FormalAdapterError("runtime fingerprint is not production-safe")

    canary_raw = _read_regular(
        Path(paths.production_canary), field="production canary receipt"
    )
    canary = _strict_json(canary_raw, field="production canary receipt")
    canary_sha = _verify_self_hash(
        canary,
        field=PRODUCTION_CANARY_SELF_FIELD,
        schema=PRODUCTION_CANARY_SCHEMA,
    )
    p0 = _require_sha256(canary.get("public_synthetic_p0_behavior_sha256"), "canary P0")
    p1 = _require_sha256(canary.get("public_synthetic_p1_behavior_sha256"), "canary P1")
    worker_fields = (
        "typed_plan_worker_receipt_sha256",
        "minilm_worker_receipt_sha256",
        "hippo_worker_receipt_sha256",
    )
    for field in worker_fields:
        _require_sha256(canary.get(field), field)
    if (
        canary.get("status") != "qualified_before_formal_source_open"
        or canary.get("qualified") is not True
        or canary.get("study_design_self_sha256")
        != acquisition.DESIGN_SELF_SHA256
        or canary.get("filesystem_isolation") != SYSTEMD_FILESYSTEM_ISOLATION
        or canary.get("formal_source_opened") is not False
        or canary.get(
            "source_identifiers_answers_families_mappings_or_labels_present"
        )
        is not False
        or canary.get("public_synthetic_distinct_rankings") is not True
        or p0 == p1
        or canary.get("runtime_fingerprint_self_sha256") != fingerprint_sha
        or canary.get("external_network_calls") != 0
        or canary.get("api_or_online_evaluator_calls") != 0
        or canary.get("retry_replay_resample_provider_switch") != 0
    ):
        raise TatqaP21FormalAdapterError("production canary is not qualified")
    return _ReceiptBundle(
        runtime_fingerprint_sha256=fingerprint_sha,
        runtime_fingerprint_file_sha256=hashlib.sha256(fingerprint_raw).hexdigest(),
        production_canary_sha256=canary_sha,
        production_canary_file_sha256=hashlib.sha256(canary_raw).hexdigest(),
        p0_behavior_sha256=p0,
        p1_behavior_sha256=p1,
    )


def _canonical_receipt_copy(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TatqaP21FormalAdapterError(f"{field} is not a mapping")
    return _strict_json(_canonical_bytes(dict(value)), field=field)


def _worker_pid(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 1:
        raise TatqaP21FormalAdapterError(f"{field} worker PID drifted")
    return value


def _validate_systemd_unit_closure(
    value: object, *, unit_name_sha256: str
) -> dict[str, Any]:
    receipt = _canonical_receipt_copy(value, field="systemd unit closure")
    expected_keys = {
        "active_state",
        "control_group_process_count",
        "control_group_sha256",
        "control_group_thread_count",
        "load_state",
        "main_pid",
        "schema",
        "sub_state",
        "systemctl_reset_failed_returncode",
        "systemctl_reset_failed_stderr_sha256",
        "systemctl_reset_failed_stdout_sha256",
        "systemctl_show_returncode",
        "systemctl_show_stderr_sha256",
        "systemctl_show_stdout_sha256",
        "unit_name_sha256",
    }
    hash_fields = {
        "control_group_sha256",
        "systemctl_reset_failed_stderr_sha256",
        "systemctl_reset_failed_stdout_sha256",
        "systemctl_show_stderr_sha256",
        "systemctl_show_stdout_sha256",
        "unit_name_sha256",
    }
    if (
        set(receipt) != expected_keys
        or receipt.get("schema") != SYSTEMD_UNIT_CLOSURE_SCHEMA
        or receipt.get("unit_name_sha256") != unit_name_sha256
        or any(
            not isinstance(receipt.get(field), str)
            or _SHA256.fullmatch(receipt[field]) is None
            for field in hash_fields
        )
        or receipt.get("load_state") != "not-found"
        or receipt.get("active_state") != "inactive"
        or receipt.get("sub_state") != "dead"
        or type(receipt.get("main_pid")) is not int
        or receipt["main_pid"] != 0
        or type(receipt.get("control_group_process_count")) is not int
        or receipt["control_group_process_count"] != 0
        or type(receipt.get("control_group_thread_count")) is not int
        or receipt["control_group_thread_count"] != 0
        or type(receipt.get("systemctl_show_returncode")) is not int
        or receipt["systemctl_show_returncode"] != 0
        or type(receipt.get("systemctl_reset_failed_returncode")) is not int
        or receipt["systemctl_reset_failed_returncode"] not in {0, 1}
    ):
        raise TatqaP21FormalAdapterError(
            "systemd unit closure receipt drifted"
        )
    return receipt


def _validate_systemd_start_policy(
    value: object, *, unit_name_sha256: str
) -> dict[str, Any]:
    receipt = _canonical_receipt_copy(value, field="systemd start policy")
    expected_keys = {
        "active_state",
        "control_group_sha256",
        "kill_mode",
        "load_state",
        "main_pid",
        "schema",
        "sub_state",
        "systemctl_show_returncode",
        "systemctl_show_stderr_sha256",
        "systemctl_show_stdout_sha256",
        "tasks_max",
        "unit_name_sha256",
    }
    if (
        set(receipt) != expected_keys
        or receipt.get("schema") != SYSTEMD_START_POLICY_SCHEMA
        or receipt.get("unit_name_sha256") != unit_name_sha256
        or receipt.get("load_state") != "loaded"
        or receipt.get("active_state") != "active"
        or receipt.get("sub_state") != "running"
        or type(receipt.get("main_pid")) is not int
        or receipt["main_pid"] <= 1
        or type(receipt.get("tasks_max")) is not int
        or receipt["tasks_max"] != 3
        or receipt.get("kill_mode") != "control-group"
        or type(receipt.get("systemctl_show_returncode")) is not int
        or receipt["systemctl_show_returncode"] != 0
        or any(
            not isinstance(receipt.get(field), str)
            or _SHA256.fullmatch(receipt[field]) is None
            for field in (
                "control_group_sha256",
                "systemctl_show_stderr_sha256",
                "systemctl_show_stdout_sha256",
                "unit_name_sha256",
            )
        )
    ):
        raise TatqaP21FormalAdapterError("systemd start policy drifted")
    return receipt


def _validate_typed_plan_transport(
    value: object,
    *,
    block: str,
    item_count: int,
    input_sha256: str,
    output_sha256: str,
) -> tuple[dict[str, Any], str, int]:
    receipt = _canonical_receipt_copy(value, field="typed-plan transport receipt")
    expected_keys = {
        "batch_size",
        "block",
        "filesystem_isolation",
        "input_sha256",
        "item_count",
        "model_execution_finished_monotonic_ns",
        "model_execution_started_monotonic_ns",
        "model_context_tokens",
        "network_properties",
        "output_sha256",
        "physical_GPU",
        "schema",
        "stderr_sha256",
        "stdout_sha256",
        "systemd_unit_closure",
        "systemd_unit_name_sha256",
        "worker_pid",
    }
    for field in ("stderr_sha256", "stdout_sha256"):
        _require_sha256(receipt.get(field), field)
    pid = _worker_pid(receipt.get("worker_pid"), field="typed-plan")
    unit_name_sha256 = _require_sha256(
        receipt.get("systemd_unit_name_sha256"), "typed-plan systemd unit name"
    )
    _validate_systemd_unit_closure(
        receipt.get("systemd_unit_closure"),
        unit_name_sha256=unit_name_sha256,
    )
    started = receipt.get("model_execution_started_monotonic_ns")
    finished = receipt.get("model_execution_finished_monotonic_ns")
    if (
        set(receipt) != expected_keys
        or receipt.get("schema") != TYPED_PLAN_TRANSPORT_SCHEMA
        or receipt.get("block") != block
        or type(receipt.get("item_count")) is not int
        or receipt["item_count"] != item_count
        or receipt.get("input_sha256") != input_sha256
        or receipt.get("output_sha256") != output_sha256
        or type(receipt.get("batch_size")) is not int
        or receipt["batch_size"] != 4
        or receipt.get("physical_GPU") != "1"
        or receipt.get("filesystem_isolation") != SYSTEMD_FILESYSTEM_ISOLATION
        or receipt.get("network_properties") != list(SYSTEMD_NETWORK_PROPERTIES)
        or isinstance(started, bool)
        or not isinstance(started, int)
        or started <= 0
        or isinstance(finished, bool)
        or not isinstance(finished, int)
        or finished <= started
        or isinstance(receipt.get("model_context_tokens"), bool)
        or not isinstance(receipt.get("model_context_tokens"), int)
        or receipt["model_context_tokens"] < 16_640
    ):
        raise TatqaP21FormalAdapterError("typed-plan transport binding drifted")
    return receipt, _semantic_hash(receipt), pid


def _validate_hippo_transport(
    value: object,
    *,
    block: str,
    item_commitment_sha256: str,
    input_file_sha256: str,
    input_semantic_sha256: str,
    output_file_sha256: str,
) -> tuple[dict[str, Any], str, int]:
    receipt = _canonical_receipt_copy(value, field="Hippo transport receipt")
    expected_keys = {
        "CPU_threads",
        "block",
        "filesystem_isolation",
        "input_file_sha256",
        "input_semantic_sha256",
        "item_commitment_sha256",
        "model_execution_finished_monotonic_ns",
        "model_execution_started_monotonic_ns",
        "network_properties",
        "observed_process_thread_peak",
        "output_file_sha256",
        "schema",
        "stderr_sha256",
        "stdout_sha256",
        "systemd_start_policy",
        "systemd_start_policy_sha256",
        "systemd_tasks_max",
        "systemd_unit_closure",
        "systemd_unit_name_sha256",
        "thread_monitor_process_reservation",
        "visible_GPU",
        "worker_pid",
        "maximum_worker_process_threads",
        "configured_torch_interop_threads",
        "configured_torch_intraop_threads",
    }
    for field in ("stderr_sha256", "stdout_sha256"):
        _require_sha256(receipt.get(field), field)
    pid = _worker_pid(receipt.get("worker_pid"), field="Hippo")
    unit_name_sha256 = _require_sha256(
        receipt.get("systemd_unit_name_sha256"), "Hippo systemd unit name"
    )
    start_policy = _validate_systemd_start_policy(
        receipt.get("systemd_start_policy"),
        unit_name_sha256=unit_name_sha256,
    )
    closure = _validate_systemd_unit_closure(
        receipt.get("systemd_unit_closure"),
        unit_name_sha256=unit_name_sha256,
    )
    start_policy_sha256 = _require_sha256(
        receipt.get("systemd_start_policy_sha256"),
        "Hippo systemd start policy",
    )
    started = receipt.get("model_execution_started_monotonic_ns")
    finished = receipt.get("model_execution_finished_monotonic_ns")
    peak = receipt.get("observed_process_thread_peak")
    intraop = receipt.get("configured_torch_intraop_threads")
    interop = receipt.get("configured_torch_interop_threads")
    if (
        set(receipt) != expected_keys
        or receipt.get("schema") != HIPPO_TRANSPORT_SCHEMA
        or receipt.get("block") != block
        or receipt.get("item_commitment_sha256") != item_commitment_sha256
        or receipt.get("input_file_sha256") != input_file_sha256
        or receipt.get("input_semantic_sha256") != input_semantic_sha256
        or receipt.get("output_file_sha256") != output_file_sha256
        or type(receipt.get("CPU_threads")) is not int
        or receipt["CPU_threads"] != controller.HIPPO_CPU_THREADS_PER_PROCESS
        or receipt.get("visible_GPU") != ""
        or receipt.get("filesystem_isolation") != SYSTEMD_FILESYSTEM_ISOLATION
        or receipt.get("network_properties") != list(SYSTEMD_NETWORK_PROPERTIES)
        or type(receipt.get("systemd_tasks_max")) is not int
        or receipt["systemd_tasks_max"] != 3
        or type(receipt.get("thread_monitor_process_reservation")) is not int
        or receipt["thread_monitor_process_reservation"] != 1
        or type(receipt.get("maximum_worker_process_threads")) is not int
        or receipt["maximum_worker_process_threads"] != 2
        or receipt["thread_monitor_process_reservation"]
        + receipt["maximum_worker_process_threads"]
        != receipt["systemd_tasks_max"]
        or start_policy.get("main_pid") != pid
        or start_policy.get("control_group_sha256")
        != closure.get("control_group_sha256")
        or _semantic_hash(start_policy) != start_policy_sha256
        or isinstance(started, bool)
        or not isinstance(started, int)
        or started <= 0
        or isinstance(finished, bool)
        or not isinstance(finished, int)
        or finished <= started
        or any(
            isinstance(row, bool) or not isinstance(row, int) or not 1 <= row <= 2
            for row in (peak, intraop, interop)
        )
    ):
        raise TatqaP21FormalAdapterError("Hippo transport binding drifted")
    return receipt, _semantic_hash(receipt), pid


class ProductionRuntimeAdapter:
    """Offline runtime boundary with content caches and durable stage receipts."""

    def __init__(
        self,
        *,
        control_root: str | Path,
        receipt_paths: RuntimeReceiptPaths,
        typed_plan_runner: TypedPlanBatchRunner,
        minilm_encoder: features.TextEncoder,
        hippo_runner: HippoRunner,
    ) -> None:
        self.control_root = Path(control_root)
        self.receipt_paths = receipt_paths
        self._typed_plan_runner = typed_plan_runner
        self._minilm_encoder = minilm_encoder
        self._hippo_runner = hippo_runner
        if (
            not callable(typed_plan_runner)
            or not callable(getattr(typed_plan_runner, "transport_receipt", None))
            or not callable(getattr(typed_plan_runner, "abort_all_workers", None))
            or not callable(
                getattr(typed_plan_runner, "verify_all_workers_closed", None)
            )
            or not callable(hippo_runner)
            or not callable(getattr(hippo_runner, "transport_receipt", None))
            or not callable(getattr(hippo_runner, "abort_all_workers", None))
            or not callable(
                getattr(hippo_runner, "verify_all_workers_closed", None)
            )
        ):
            raise TatqaP21FormalAdapterError("worker runner capability is unavailable")
        if not hasattr(minilm_encoder, "encode") or not callable(minilm_encoder.encode):
            raise TatqaP21FormalAdapterError("MiniLM encoder capability is unavailable")
        self._lock = threading.Lock()
        self._preflight_called = False
        self._receipts: _ReceiptBundle | None = None
        self._prepared_attempts: set[str] = set()
        self._prepared: dict[str, _PreparedBlock] = {}
        self._raw_attempts: set[tuple[str, str]] = set()
        self._hippo_attempts: set[tuple[str, str]] = set()
        self._hippo_worker_receipts: dict[tuple[str, str], str] = {}
        self._hippo_transport_receipts: dict[
            tuple[str, str], tuple[dict[str, Any], str, int]
        ] = {}
        self._inference_preparations: dict[str, _InferencePreparation] = {}
        self._postflight_attempts: set[str] = set()
        self._stage_seals: dict[str, tuple[str, str]] = {}
        self._offline_evidence_attempts: set[str] = set()
        self._offline_evidence_seals: dict[str, str] = {}

    @property
    def receipt_bundle(self) -> _ReceiptBundle:
        if self._receipts is None:
            raise TatqaP21FormalAdapterError("runtime preflight has not completed")
        return self._receipts

    def preflight(self) -> controller.RuntimePreflight:
        with self._lock:
            if self._preflight_called:
                raise TatqaP21FormalAdapterError("runtime preflight replay is forbidden")
            self._preflight_called = True
        receipts = _load_public_runtime_receipts(self.receipt_paths)
        self._receipts = receipts
        return controller.RuntimePreflight(
            qualified=True,
            public_synthetic_distinct_rankings=True,
            public_synthetic_p0_behavior_sha256=receipts.p0_behavior_sha256,
            public_synthetic_p1_behavior_sha256=receipts.p1_behavior_sha256,
            external_network_calls=0,
            api_or_online_evaluator_calls=0,
            retry_replay_resample_provider_switch=0,
        )

    def _invoke_typed_plan(
        self,
        state: _InferencePreparation,
        canonical_input: bytes,
    ) -> bytes:
        with self._lock:
            state.qwen_live = True
            if state.hippo_live:
                state.overlap_observed = True
        try:
            if state.start_barrier is not None:
                state.start_barrier.wait(timeout=30.0)
            return self._typed_plan_runner(state.block, canonical_input)
        finally:
            with self._lock:
                state.qwen_live = False

    def _invoke_hippo(
        self,
        state: _InferencePreparation,
        item_commitment_sha256: str,
        canonical_input: bytes,
        synchronize_start: bool,
    ) -> bytes:
        with self._lock:
            state.hippo_live += 1
            state.hippo_peak = max(state.hippo_peak, state.hippo_live)
            if state.hippo_live > ACTUAL_HIPPO_INFERENCE_CAP:
                state.hippo_live -= 1
                raise TatqaP21FormalAdapterError(
                    "actual Hippo inference cap was exceeded"
                )
            if state.qwen_live:
                state.overlap_observed = True
        try:
            if synchronize_start:
                if state.start_barrier is None:
                    raise TatqaP21FormalAdapterError(
                        "inference start barrier is unavailable"
                    )
                state.start_barrier.wait(timeout=30.0)
            return self._hippo_runner(
                state.block, item_commitment_sha256, canonical_input
            )
        finally:
            with self._lock:
                if state.hippo_live <= 0:
                    raise TatqaP21FormalAdapterError(
                        "actual Hippo inference counter underflow"
                    )
                state.hippo_live -= 1

    @staticmethod
    def _hippo_input(
        item: features.LabelFreeRuntimeItem,
    ) -> tuple[bytes, str]:
        units = [
            {"ordinal": ordinal, "text": row.text, "unit_id": row.unit_id}
            for ordinal, row in enumerate(item.units)
        ]
        payload = hipporag_contract.input_payload(query=item.question, units=units)
        return hipporag_contract.canonical_json_bytes(payload), payload["input_sha256"]

    def _abort_inference_preparation(
        self, state: _InferencePreparation
    ) -> None:
        """Cancel queued work and synchronously reap both model executors."""

        self._close_inference_preparation(state, cancel_futures=True)

    def _close_inference_preparation(
        self,
        state: _InferencePreparation,
        *,
        cancel_futures: bool,
    ) -> None:
        """Idempotently and synchronously close one prepared model cohort."""

        # A separate per-state mutex makes a controller-level terminal abort
        # safe even if an exception races the normal postflight close.  The
        # caller does not return until both executor shutdowns have joined.
        with state.shutdown_lock:
            with self._lock:
                if state.executors_closed:
                    return
                state.shutdown_started = True
            if cancel_futures:
                if state.qwen_future is not None:
                    state.qwen_future.cancel()
                for future in state.hippo_futures.values():
                    future.cancel()
            failures: list[BaseException] = []
            executors_joined = True
            if cancel_futures:
                # Stop the actual named systemd units before joining wrapper
                # threads; Future.cancel() alone cannot terminate a running
                # subprocess or its control group.
                for runner in (self._typed_plan_runner, self._hippo_runner):
                    try:
                        runner.abort_all_workers()
                    except BaseException as exc:
                        failures.append(exc)
            try:
                state.qwen_executor.shutdown(
                    wait=True, cancel_futures=cancel_futures
                )
            except BaseException as exc:
                executors_joined = False
                failures.append(exc)
            if state.hippo_executor is not None:
                try:
                    state.hippo_executor.shutdown(
                        wait=True, cancel_futures=cancel_futures
                    )
                except BaseException as exc:
                    executors_joined = False
                    failures.append(exc)
            # A second abort closes a unit that could have registered in the
            # narrow cancel-vs-launch race; the calls are supervisor-idempotent.
            if cancel_futures:
                for runner in (self._typed_plan_runner, self._hippo_runner):
                    try:
                        runner.abort_all_workers()
                    except BaseException as exc:
                        failures.append(exc)
            closure_proved = True
            for runner in (self._typed_plan_runner, self._hippo_runner):
                try:
                    runner.verify_all_workers_closed()
                except BaseException as exc:
                    closure_proved = False
                    failures.append(exc)
            if closure_proved and executors_joined:
                with self._lock:
                    state.executors_closed = True
            if failures:
                raise TatqaP21FormalAdapterError(
                    "actual worker/executor closure failed"
                ) from failures[0]

    def abort_all_inference(self) -> None:
        """Synchronously reap every cohort after any one-shot failure."""

        with self._lock:
            states = tuple(self._inference_preparations.values())
        failures: list[BaseException] = []
        for state in states:
            try:
                self._abort_inference_preparation(state)
            except BaseException as exc:  # fail closed after trying all states
                failures.append(exc)
        if failures:
            raise TatqaP21FormalAdapterError(
                "one or more inference cohorts could not be reaped"
            ) from failures[0]

    def verify_all_inference_closed(self) -> None:
        """Prove no prepared executor remains live at a terminal boundary."""

        with self._lock:
            if any(
                not state.executors_closed
                for state in self._inference_preparations.values()
            ):
                raise TatqaP21FormalAdapterError(
                    "an inference cohort remains live at terminal disposition"
                )
        for runner in (self._typed_plan_runner, self._hippo_runner):
            try:
                runner.verify_all_workers_closed()
            except Exception as exc:
                raise TatqaP21FormalAdapterError(
                    "a named systemd worker remains live at terminal disposition"
                ) from exc

    def prepare_block(
        self,
        *,
        block: str,
        block_view_sha256: str,
        items: tuple[features.LabelFreeRuntimeItem, ...],
    ) -> controller.BlockView:
        receipts = self.receipt_bundle
        if block not in controller.BLOCK_ITEM_COUNTS:
            raise TatqaP21FormalAdapterError("unknown preparation block")
        _require_sha256(block_view_sha256, "block view")
        if len(items) != controller.BLOCK_ITEM_COUNTS[block]:
            raise TatqaP21FormalAdapterError("prepared block item count drifted")
        with self._lock:
            if block in self._prepared_attempts:
                raise TatqaP21FormalAdapterError("block preparation replay is forbidden")
            self._prepared_attempts.add(block)

        projected = tuple(
            typed_plan_contract.project_item(item, ordinal)
            for ordinal, item in enumerate(items)
        )
        plan_input_payload = typed_plan_contract.input_payload(projected)
        plan_input_raw = typed_plan_contract.canonical_json_bytes(plan_input_payload)
        hippo_enabled = block in {"A_hold", "M_search"}
        if (
            hippo_enabled
            and controller.HIPPO_CONCURRENCY_CAP
            != ACTUAL_HIPPO_INFERENCE_CAP
        ):
            raise TatqaP21FormalAdapterError(
                "frozen Hippo inference cap drifted"
            )
        qwen_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"p18-{block}-qwen-inference"
        )
        hippo_inference_executor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=ACTUAL_HIPPO_INFERENCE_CAP,
                thread_name_prefix=f"p18-{block}-hippo-inference",
            )
            if hippo_enabled
            else None
        )
        inference = _InferencePreparation(
            block=block,
            qwen_executor=qwen_executor,
            hippo_executor=hippo_inference_executor,
            start_barrier=threading.Barrier(2) if hippo_enabled else None,
        )
        # Register before the first submit so controller-level terminal abort
        # can always discover this cohort, including failures late in view
        # construction after the expensive workers have already completed.
        with self._lock:
            self._inference_preparations[block] = inference
        try:
            inference.qwen_future = qwen_executor.submit(
                self._invoke_typed_plan, inference, plan_input_raw
            )
            if hippo_enabled:
                if hippo_inference_executor is None:
                    raise TatqaP21FormalAdapterError(
                        "Hippo inference executor is unavailable"
                    )
                for ordinal, item in enumerate(items):
                    hippo_input_raw, hippo_semantic_sha = self._hippo_input(item)
                    inference.hippo_inputs[item.item_id] = (
                        hippo_input_raw,
                        hippo_semantic_sha,
                    )
                    inference.hippo_futures[item.item_id] = (
                        hippo_inference_executor.submit(
                            self._invoke_hippo,
                            inference,
                            item.item_id,
                            hippo_input_raw,
                            ordinal == 0,
                        )
                    )
            inference.expected_submission_count = 1 + len(
                inference.hippo_futures
            )
            inference.submitted_before_first_join = (
                1 + len(inference.hippo_futures)
            )
            if (
                inference.submitted_before_first_join
                != inference.expected_submission_count
            ):
                raise TatqaP21FormalAdapterError(
                    "inference submission accounting drifted"
                )
            inference.first_join_started = True
            # This is the first Future.result() in the production path.  The
            # whole-block Qwen job and every item-local Hippo job are already
            # submitted to their physically separate inference executors.
            plan_output_raw = inference.qwen_future.result()
            inference.qwen_consumed = True
        except Exception as exc:
            self._abort_inference_preparation(inference)
            raise TatqaP21FormalAdapterError("typed-plan worker failed") from exc
        if not isinstance(plan_output_raw, bytes):
            self._abort_inference_preparation(inference)
            raise TatqaP21FormalAdapterError("typed-plan worker did not return bytes")
        try:
            plan_output = typed_plan_contract.parse_output(plan_output_raw)
        except Exception as exc:
            self._abort_inference_preparation(inference)
            raise TatqaP21FormalAdapterError("typed-plan worker output drifted") from exc
        output_rows = plan_output["items"]
        if len(output_rows) != len(items):
            self._abort_inference_preparation(inference)
            raise TatqaP21FormalAdapterError("typed-plan output cardinality drifted")
        plan_input_sha = hashlib.sha256(plan_input_raw).hexdigest()
        plan_output_sha = hashlib.sha256(plan_output_raw).hexdigest()
        try:
            transport_value = self._typed_plan_runner.transport_receipt(block)
        except Exception as exc:
            self._abort_inference_preparation(inference)
            raise TatqaP21FormalAdapterError(
                "typed-plan transport receipt is unavailable"
            ) from exc
        try:
            plan_transport, plan_transport_sha, plan_worker_pid = (
                _validate_typed_plan_transport(
                    transport_value,
                    block=block,
                    item_count=len(items),
                    input_sha256=plan_input_sha,
                    output_sha256=plan_output_sha,
                )
            )
        except Exception:
            self._abort_inference_preparation(inference)
            raise

        compiled_rows: list[features.CompiledLabelFreeItem] = []
        item_receipts: list[dict[str, Any]] = []
        for ordinal, (item, output_row) in enumerate(
            zip(items, output_rows, strict=True)
        ):
            if output_row.get("ordinal") != ordinal:
                self._abort_inference_preparation(inference)
                raise TatqaP21FormalAdapterError("typed-plan output order drifted")
            try:
                plan = core.validate_typed_plan(output_row["plan"])
                compiled = features.compile_with_encoder(
                    item, plan, self._minilm_encoder
                )
            except Exception as exc:
                self._abort_inference_preparation(inference)
                raise TatqaP21FormalAdapterError(
                    "label-free semantic compilation failed"
                ) from exc
            prompt_receipt = {
                key: output_row[key]
                for key in (
                    "completion_sha256",
                    "completion_token_count",
                    "generation_valid",
                    "ordinal",
                    "prompt_projection_sha256",
                    "prompt_sha256",
                    "prompt_token_count",
                )
            }
            compiled_rows.append(compiled)
            item_receipts.append(
                {
                    "item_commitment_sha256": item.item_id,
                    "prompt_receipt_sha256": _semantic_hash(prompt_receipt),
                    "raw_behavior_sha256": core.canonical_behavior_hash(
                        compiled.raw_top5
                    ),
                    "tensor_sha256": compiled.tensor_sha256,
                }
            )

        runtime_by_id = {row.item_id: row for row in items}
        compiled_by_id = {row.item_id: row for row in compiled_rows}
        if len(runtime_by_id) != len(items) or len(compiled_by_id) != len(items):
            self._abort_inference_preparation(inference)
            raise TatqaP21FormalAdapterError("prepared item commitments are duplicated")
        plan_worker_receipt = _semantic_hash(
            {
                "block": block,
                "input_sha256": plan_input_sha,
                "output_sha256": plan_output_sha,
                "production_canary_sha256": receipts.production_canary_sha256,
                "runtime_fingerprint_sha256": receipts.runtime_fingerprint_sha256,
                "schema": f"{VERSION}_typed_plan_worker_receipt_v1",
                "transport_receipt_sha256": plan_transport_sha,
                "worker_pid": plan_worker_pid,
            }
        )
        preparation_body = {
            "actual_model_future_expected_count": (
                inference.expected_submission_count
            ),
            "actual_model_future_submit_count_before_first_join": (
                inference.submitted_before_first_join
            ),
            "all_actual_model_futures_submitted_before_first_join": (
                inference.first_join_started
                and inference.submitted_before_first_join
                == inference.expected_submission_count
            ),
            "block": block,
            "block_view_sha256": block_view_sha256,
            "hippo_actual_inference_cap": (
                ACTUAL_HIPPO_INFERENCE_CAP if hippo_enabled else 0
            ),
            "hippo_actual_submitted_count": len(inference.hippo_futures),
            "item_count": len(items),
            "items": item_receipts,
            "minilm_raw_compiled_item_count": len(compiled_rows),
            "production_canary_sha256": receipts.production_canary_sha256,
            "qwen_hippo_dedicated_inference_executors": hippo_enabled,
            # This parent-side hint is scheduling telemetry only.  The
            # auditable overlap claim is computed later from worker-process
            # monotonic model intervals in preparation_inference_receipt().
            "parent_scheduler_invocation_overlap_hint": (
                inference.overlap_observed
            ),
            "retry_replay_resample_provider_switch": 0,
            "runtime_fingerprint_sha256": receipts.runtime_fingerprint_sha256,
            "schema": f"{VERSION}_block_preparation_receipt_v1",
            "typed_plan_input_sha256": plan_input_sha,
            "typed_plan_output_sha256": plan_output_sha,
            "typed_plan_transport_receipt_sha256": plan_transport_sha,
            "typed_plan_worker_receipt_sha256": plan_worker_receipt,
            "typed_plan_worker_pid": plan_worker_pid,
        }
        preparation_sha = _semantic_hash(preparation_body)
        preparation = {**preparation_body, "preparation_receipt_sha256": preparation_sha}
        stage_root = self.control_root / "stages" / block
        try:
            stages_metadata = stage_root.parent.lstat()
        except OSError as exc:
            self._abort_inference_preparation(inference)
            raise TatqaP21FormalAdapterError("durable stages root is absent") from exc
        if (
            stat.S_ISLNK(stages_metadata.st_mode)
            or not stat.S_ISDIR(stages_metadata.st_mode)
            or stat.S_IMODE(stages_metadata.st_mode) != 0o700
        ):
            self._abort_inference_preparation(inference)
            raise TatqaP21FormalAdapterError("durable stages root drifted")
        try:
            _mkdir_durable(stage_root)
            _write_exclusive_verified(
                stage_root / "block.preparation.json", preparation
            )
        except Exception:
            self._abort_inference_preparation(inference)
            raise

        try:
            views = tuple(
                controller.ItemView(
                    item_commitment_sha256=row.item_id,
                    plan=compiled_by_id[row.item_id].plan,
                    units=compiled_by_id[row.item_id].units,
                    redundancy_features=tuple(
                        (left, right, value)
                        for (left, right), value in compiled_by_id[
                            row.item_id
                        ].redundancy_features.items()
                    ),
                )
                for row in items
            )
            block_view = controller.BlockView(block=block, items=views)
            prepared = _PreparedBlock(
                block_view=block_view,
                runtime_items=runtime_by_id,
                compiled_items=compiled_by_id,
                preparation_receipt_sha256=preparation_sha,
                typed_plan_input_sha256=plan_input_sha,
                typed_plan_output_sha256=plan_output_sha,
                typed_plan_transport_receipt=plan_transport,
                typed_plan_transport_receipt_sha256=plan_transport_sha,
                typed_plan_worker_receipt_sha256=plan_worker_receipt,
                typed_plan_worker_pid=plan_worker_pid,
            )
        except Exception:
            self._abort_inference_preparation(inference)
            raise
        with self._lock:
            self._prepared[block] = prepared
        return block_view

    def action_executor(
        self, block: str, standard_work_count: int
    ) -> concurrent.futures.ThreadPoolExecutor:
        if (
            block not in self._prepared
            or isinstance(standard_work_count, bool)
            or not isinstance(standard_work_count, int)
            or standard_work_count < 1
        ):
            raise TatqaP21FormalAdapterError("standard executor request drifted")
        return concurrent.futures.ThreadPoolExecutor(
            max_workers=min(STANDARD_EXECUTOR_CAP, standard_work_count),
            thread_name_prefix=f"p18-{block}-standard",
        )

    def hippo_executor(
        self, block: str, concurrency_cap: int
    ) -> concurrent.futures.ThreadPoolExecutor:
        if (
            block not in {"A_hold", "M_search"}
            or block not in self._prepared
            or concurrency_cap != controller.HIPPO_CONCURRENCY_CAP
        ):
            raise TatqaP21FormalAdapterError("Hippo executor request drifted")
        return concurrent.futures.ThreadPoolExecutor(
            max_workers=controller.HIPPO_CONCURRENCY_CAP,
            thread_name_prefix=f"p18-{block}-hippo",
        )

    def _bound_item(
        self, block: str, item: controller.ItemView
    ) -> tuple[features.LabelFreeRuntimeItem, features.CompiledLabelFreeItem]:
        prepared = self._prepared.get(block)
        if prepared is None or not isinstance(item, controller.ItemView):
            raise TatqaP21FormalAdapterError("runtime item block is not prepared")
        expected = next(
            (
                row
                for row in prepared.block_view.items
                if row.item_commitment_sha256 == item.item_commitment_sha256
            ),
            None,
        )
        if expected is None or expected != item:
            raise TatqaP21FormalAdapterError("runtime item is not the prepared view object")
        return (
            prepared.runtime_items[item.item_commitment_sha256],
            prepared.compiled_items[item.item_commitment_sha256],
        )

    def run_raw(self, block: str, item: controller.ItemView) -> Sequence[str]:
        _runtime_item, compiled = self._bound_item(block, item)
        key = (block, item.item_commitment_sha256)
        with self._lock:
            if key in self._raw_attempts:
                raise TatqaP21FormalAdapterError("RAW action replay is forbidden")
            self._raw_attempts.add(key)
        return compiled.raw_top5

    def run_hippo(self, block: str, item: controller.ItemView) -> Sequence[str]:
        runtime_item, _compiled = self._bound_item(block, item)
        key = (block, item.item_commitment_sha256)
        with self._lock:
            if key in self._hippo_attempts:
                raise TatqaP21FormalAdapterError("Hippo action replay is forbidden")
            self._hippo_attempts.add(key)
        inference = self._inference_preparations.get(block)
        if inference is None:
            raise TatqaP21FormalAdapterError(
                "Hippo preparation state is unavailable"
            )
        input_binding = inference.hippo_inputs.get(
            item.item_commitment_sha256
        )
        future = inference.hippo_futures.get(item.item_commitment_sha256)
        if input_binding is None or future is None:
            raise TatqaP21FormalAdapterError(
                "prestarted Hippo future is unavailable"
            )
        input_raw, expected_input_sha = input_binding
        try:
            # Consume the exact item future submitted by prepare_block.  No
            # model invocation is started from the controller action wrapper.
            output_raw = future.result()
        except Exception as exc:
            self._abort_inference_preparation(inference)
            raise TatqaP21FormalAdapterError(
                "prestarted official Hippo worker failed"
            ) from exc
        if not isinstance(output_raw, bytes):
            self._abort_inference_preparation(inference)
            raise TatqaP21FormalAdapterError("official Hippo worker did not return bytes")
        try:
            output = hipporag_contract.parse_output(output_raw)
        except Exception as exc:
            self._abort_inference_preparation(inference)
            raise TatqaP21FormalAdapterError("official Hippo output drifted") from exc
        top5 = tuple(output["top_unit_ids"])
        corpus = {row.unit_id for row in runtime_item.units}
        if (
            output["input_sha256"] != expected_input_sha
            or output["unit_count"] != len(runtime_item.units)
            or not set(top5).issubset(corpus)
        ):
            self._abort_inference_preparation(inference)
            raise TatqaP21FormalAdapterError("official Hippo item binding drifted")
        input_file_sha = hashlib.sha256(input_raw).hexdigest()
        output_file_sha = hashlib.sha256(output_raw).hexdigest()
        try:
            transport_value = self._hippo_runner.transport_receipt(
                block, item.item_commitment_sha256
            )
        except Exception as exc:
            self._abort_inference_preparation(inference)
            raise TatqaP21FormalAdapterError(
                "Hippo transport receipt is unavailable"
            ) from exc
        try:
            transport, transport_sha, worker_pid = _validate_hippo_transport(
                transport_value,
                block=block,
                item_commitment_sha256=item.item_commitment_sha256,
                input_file_sha256=input_file_sha,
                input_semantic_sha256=expected_input_sha,
                output_file_sha256=output_file_sha,
            )
        except Exception:
            self._abort_inference_preparation(inference)
            raise
        receipts = self.receipt_bundle
        worker_receipt = _semantic_hash(
            {
                "block": block,
                "input_sha256": input_file_sha,
                "item_commitment_sha256": item.item_commitment_sha256,
                "output_sha256": output_file_sha,
                "production_canary_sha256": receipts.production_canary_sha256,
                "runtime_fingerprint_sha256": receipts.runtime_fingerprint_sha256,
                "schema": f"{VERSION}_hippo_worker_receipt_v1",
                "transport_receipt_sha256": transport_sha,
                "worker_pid": worker_pid,
            }
        )
        with self._lock:
            self._hippo_worker_receipts[key] = worker_receipt
            self._hippo_transport_receipts[key] = (
                transport,
                transport_sha,
                worker_pid,
            )
            inference.hippo_consumed.add(item.item_commitment_sha256)
        return top5

    def preparation_inference_receipt(self, block: str) -> Mapping[str, object]:
        """Return immutable terminal accounting for actual model futures.

        The controller's action executors account for logical wrappers.  This
        separate receipt proves that the expensive Qwen/Hippo model work was
        submitted eagerly by :meth:`prepare_block` and was not re-launched by
        those wrappers.
        """

        prepared = self._prepared.get(block)
        inference = self._inference_preparations.get(block)
        if prepared is None or inference is None:
            raise TatqaP21FormalAdapterError(
                "inference preparation receipt is unavailable"
            )
        hippo_enabled = block in {"A_hold", "M_search"}
        expected_hippo = (
            controller.BLOCK_ITEM_COUNTS[block] if hippo_enabled else 0
        )
        with self._lock:
            hippo_receipts = tuple(
                self._hippo_worker_receipts[row]
                for row in sorted(self._hippo_worker_receipts)
                if row[0] == block
            )
            hippo_transport_rows = tuple(
                (row[1], self._hippo_transport_receipts[row])
                for row in sorted(self._hippo_transport_receipts)
                if row[0] == block
            )
            hippo_consumed = set(inference.hippo_consumed)
            live = inference.hippo_live
            peak = inference.hippo_peak
        qwen_transport, qwen_transport_sha, qwen_pid = (
            _validate_typed_plan_transport(
                prepared.typed_plan_transport_receipt,
                block=block,
                item_count=len(prepared.runtime_items),
                input_sha256=prepared.typed_plan_input_sha256,
                output_sha256=prepared.typed_plan_output_sha256,
            )
        )
        if (
            qwen_transport_sha
            != prepared.typed_plan_transport_receipt_sha256
            or qwen_pid != prepared.typed_plan_worker_pid
        ):
            raise TatqaP21FormalAdapterError(
                "typed-plan transport receipt changed after preparation"
            )
        validated_hippo: list[tuple[str, dict[str, Any], str, int]] = []
        for item_commitment, (transport_value, expected_sha, expected_pid) in (
            hippo_transport_rows
        ):
            input_raw, input_semantic_sha = inference.hippo_inputs[
                item_commitment
            ]
            output_sha = transport_value.get("output_file_sha256")
            if not isinstance(output_sha, str):
                raise TatqaP21FormalAdapterError(
                    "Hippo transport output binding is absent"
                )
            transport, observed_sha, observed_pid = _validate_hippo_transport(
                transport_value,
                block=block,
                item_commitment_sha256=item_commitment,
                input_file_sha256=hashlib.sha256(input_raw).hexdigest(),
                input_semantic_sha256=input_semantic_sha,
                output_file_sha256=output_sha,
            )
            if observed_sha != expected_sha or observed_pid != expected_pid:
                raise TatqaP21FormalAdapterError(
                    "Hippo transport receipt changed after consumption"
                )
            validated_hippo.append(
                (item_commitment, transport, observed_sha, observed_pid)
            )
        qwen_started = qwen_transport["model_execution_started_monotonic_ns"]
        qwen_finished = qwen_transport["model_execution_finished_monotonic_ns"]
        overlap_witnesses = [
            item_commitment
            for item_commitment, transport, _sha, _pid in validated_hippo
            if max(
                qwen_started,
                transport["model_execution_started_monotonic_ns"],
            )
            < min(
                qwen_finished,
                transport["model_execution_finished_monotonic_ns"],
            )
        ]
        overlap = bool(overlap_witnesses)
        qwen_done = (
            inference.qwen_future is not None
            and inference.qwen_future.done()
            and inference.qwen_consumed
        )
        hippo_done = sum(
            future.done() for future in inference.hippo_futures.values()
        )
        if (
            not qwen_done
            or len(inference.hippo_futures) != expected_hippo
            or hippo_done != expected_hippo
            or len(hippo_consumed) != expected_hippo
            or len(hippo_receipts) != expected_hippo
            or len(validated_hippo) != expected_hippo
            or live != 0
            or inference.submitted_before_first_join
            != 1 + expected_hippo
            or inference.expected_submission_count != 1 + expected_hippo
            or inference.first_join_started is not True
            or (
                hippo_enabled
                and (
                    not 1 <= peak <= ACTUAL_HIPPO_INFERENCE_CAP
                    or overlap is not True
                    or inference.hippo_executor is None
                    or inference.hippo_executor is inference.qwen_executor
                )
            )
            or (
                not hippo_enabled
                and (peak != 0 or overlap is not False)
            )
        ):
            raise TatqaP21FormalAdapterError(
                "actual inference terminal accounting is incomplete"
            )
        body: dict[str, object] = {
            "actual_model_future_expected_count": 1 + expected_hippo,
            "actual_model_future_submit_count_before_first_join": (
                inference.submitted_before_first_join
            ),
            "all_actual_model_futures_submitted_before_first_join": True,
            "block": block,
            "hippo_actual_concurrency_cap": (
                ACTUAL_HIPPO_INFERENCE_CAP if hippo_enabled else 0
            ),
            "hippo_actual_observed_peak": peak,
            "hippo_future_submitted_count": len(inference.hippo_futures),
            "hippo_future_terminal_count": hippo_done,
            "hippo_future_consumed_count": len(hippo_consumed),
            "hippo_transport_receipt_sha256s": [
                row[2] for row in validated_hippo
            ],
            "hippo_transport_receipts": [
                row[1] for row in validated_hippo
            ],
            "hippo_worker_pids": [row[3] for row in validated_hippo],
            "hippo_worker_receipt_sha256s": list(hippo_receipts),
            "hippo_executor_dedicated": hippo_enabled,
            "minilm_raw_compiled_item_count": len(prepared.compiled_items),
            "qwen_batch_item_count": len(prepared.runtime_items),
            "qwen_batch_submitted_count": 1,
            "qwen_batch_terminal_count": 1,
            "qwen_executor_dedicated": True,
            "qwen_hippo_independent_executors": hippo_enabled,
            "qwen_hippo_overlap_observed": overlap,
            "qwen_hippo_overlap_witness_item_commitments": overlap_witnesses,
            "qwen_transport_receipt": qwen_transport,
            "qwen_transport_receipt_sha256": (
                qwen_transport_sha
            ),
            "qwen_worker_pid": qwen_pid,
            "qwen_worker_receipt_sha256": (
                prepared.typed_plan_worker_receipt_sha256
            ),
            "retry_replay_resample_provider_switch": 0,
            "schema": f"{VERSION}_preparation_inference_receipt_v1",
        }
        return {
            **body,
            "preparation_inference_receipt_sha256": _semantic_hash(body),
        }

    def _revalidate_runner_transport_receipts(
        self, block: str
    ) -> tuple[dict[str, Any], ...]:
        """Reopen runner-owned receipts and prove exact byte semantics again."""

        prepared = self._prepared.get(block)
        inference = self._inference_preparations.get(block)
        if prepared is None or inference is None:
            raise TatqaP21FormalAdapterError(
                "transport receipt revalidation lacks preparation state"
            )
        qwen, qwen_sha, qwen_pid = _validate_typed_plan_transport(
            self._typed_plan_runner.transport_receipt(block),
            block=block,
            item_count=len(prepared.runtime_items),
            input_sha256=prepared.typed_plan_input_sha256,
            output_sha256=prepared.typed_plan_output_sha256,
        )
        if (
            qwen != dict(prepared.typed_plan_transport_receipt)
            or qwen_sha != prepared.typed_plan_transport_receipt_sha256
            or qwen_pid != prepared.typed_plan_worker_pid
        ):
            raise TatqaP21FormalAdapterError(
                "typed-plan runner transport receipt drifted"
            )
        rows: list[dict[str, Any]] = [qwen]
        for key in sorted(self._hippo_transport_receipts):
            if key[0] != block:
                continue
            item_commitment = key[1]
            stored, stored_sha, stored_pid = self._hippo_transport_receipts[key]
            input_raw, semantic_sha = inference.hippo_inputs[item_commitment]
            output_sha = stored.get("output_file_sha256")
            if not isinstance(output_sha, str):
                raise TatqaP21FormalAdapterError(
                    "stored Hippo output binding is absent"
                )
            reopened, reopened_sha, reopened_pid = _validate_hippo_transport(
                self._hippo_runner.transport_receipt(block, item_commitment),
                block=block,
                item_commitment_sha256=item_commitment,
                input_file_sha256=hashlib.sha256(input_raw).hexdigest(),
                input_semantic_sha256=semantic_sha,
                output_file_sha256=output_sha,
            )
            if (
                reopened != stored
                or reopened_sha != stored_sha
                or reopened_pid != stored_pid
            ):
                raise TatqaP21FormalAdapterError(
                    "Hippo runner transport receipt drifted"
                )
            rows.append(reopened)
        return tuple(rows)

    def postflight(
        self, block: str, archive: controller.StageArchive
    ) -> controller.RuntimePostflight:
        receipts = self.receipt_bundle
        prepared = self._prepared.get(block)
        if prepared is None or not isinstance(archive, controller.StageArchive):
            raise TatqaP21FormalAdapterError("postflight stage is not prepared")
        if archive.block != block:
            raise TatqaP21FormalAdapterError("postflight archive block drifted")
        with self._lock:
            if block in self._postflight_attempts:
                raise TatqaP21FormalAdapterError("postflight replay is forbidden")
            self._postflight_attempts.add(block)
            raw_count = sum(row[0] == block for row in self._raw_attempts)
            hippo_count = sum(row[0] == block for row in self._hippo_attempts)
            hippo_receipts = tuple(
                self._hippo_worker_receipts[row]
                for row in sorted(self._hippo_worker_receipts)
                if row[0] == block
            )
            hippo_transport = tuple(
                self._hippo_transport_receipts[row]
                for row in sorted(self._hippo_transport_receipts)
                if row[0] == block
            )
        expected_baseline_count = (
            controller.BLOCK_ITEM_COUNTS[block]
            if block in {"A_hold", "M_search"}
            else 0
        )
        if (
            raw_count != expected_baseline_count
            or hippo_count != expected_baseline_count
            or len(hippo_receipts) != expected_baseline_count
            or len(hippo_transport) != expected_baseline_count
        ):
            raise TatqaP21FormalAdapterError("worker terminal receipts are incomplete")
        inference_receipt = _canonical_receipt_copy(
            self.preparation_inference_receipt(block),
            field="preparation inference receipt",
        )
        inference_receipt_sha = _verify_self_hash(
            inference_receipt,
            field="preparation_inference_receipt_sha256",
            schema=f"{VERSION}_preparation_inference_receipt_v1",
        )
        transport_receipts = self._revalidate_runner_transport_receipts(block)
        inference = self._inference_preparations[block]
        with self._lock:
            if inference.shutdown_started or inference.executors_closed:
                raise TatqaP21FormalAdapterError(
                    "inference executors closed before postflight"
                )
        self._close_inference_preparation(
            inference, cancel_futures=False
        )
        if (
            self.preparation_inference_receipt(block) != inference_receipt
            or dict(archive.inference_preparation_receipt)
            != inference_receipt
            or not inference.executors_closed
        ):
            raise TatqaP21FormalAdapterError(
                "inference executor shutdown changed terminal accounting"
            )
        transport_hashes = (
            prepared.typed_plan_transport_receipt_sha256,
            *(row[1] for row in hippo_transport),
        )
        worker_pids = (
            prepared.typed_plan_worker_pid,
            *(row[2] for row in hippo_transport),
        )
        if tuple(_semantic_hash(row) for row in transport_receipts) != (
            transport_hashes
        ):
            raise TatqaP21FormalAdapterError(
                "full transport receipt/hash binding drifted"
            )
        transport_aggregate_sha = _semantic_hash(
            {
                "transport_receipts": list(transport_receipts),
                "transport_receipt_sha256s": list(transport_hashes),
                "worker_pids": list(worker_pids),
            }
        )
        stage_root = self.control_root / "stages" / block
        _write_exclusive_verified(
            stage_root / "preparation.inference.json", inference_receipt
        )
        archive_body: dict[str, Any] = {
            "archive": archive.payload(),
            "archive_sha256": archive.archive_sha256,
            "block": block,
            "block_preparation_receipt_sha256": (
                prepared.preparation_receipt_sha256
            ),
            "hippo_worker_receipt_sha256s": list(hippo_receipts),
            "inference_executors_closed_after_terminal_validation": True,
            "preparation_inference_receipt_sha256": inference_receipt_sha,
            "production_canary_sha256": receipts.production_canary_sha256,
            "runtime_fingerprint_sha256": receipts.runtime_fingerprint_sha256,
            "schema": f"{VERSION}_durable_action_archive_v1",
            "transport_receipt_aggregate_sha256": transport_aggregate_sha,
            "transport_receipts": list(transport_receipts),
            "transport_receipt_sha256s": list(transport_hashes),
            "worker_pids": list(worker_pids),
        }
        archive_envelope = {
            **archive_body,
            "durable_archive_receipt_sha256": _semantic_hash(archive_body),
        }
        _write_exclusive_verified(
            stage_root / "action.archive.json", archive_envelope
        )
        postflight = controller.RuntimePostflight(
            block=block,
            archive_sha256=archive.archive_sha256,
            runtime_ok=True,
            external_network_calls=0,
            api_or_online_evaluator_calls=0,
            retry_replay_resample_provider_switch=0,
            controller_or_worker_source_reads=0,
            controller_or_worker_label_reads=0,
            maximum_cpu_threads_per_hippo_process=(
                controller.HIPPO_CPU_THREADS_PER_PROCESS
                if block in {"A_hold", "M_search"}
                else 0
            ),
        )
        postflight_body: dict[str, Any] = {
            "block": block,
            "postflight": postflight.payload(),
            "postflight_sha256": postflight.postflight_sha256,
            "inference_executors_closed_after_terminal_validation": True,
            "preparation_inference_receipt_sha256": inference_receipt_sha,
            "production_canary_sha256": receipts.production_canary_sha256,
            "runtime_fingerprint_sha256": receipts.runtime_fingerprint_sha256,
            "schema": f"{VERSION}_durable_runtime_postflight_v1",
            "transport_receipt_aggregate_sha256": transport_aggregate_sha,
            "transport_receipts": list(transport_receipts),
            "transport_receipt_sha256s": list(transport_hashes),
            "worker_pids": list(worker_pids),
        }
        postflight_envelope = {
            **postflight_body,
            "durable_postflight_receipt_sha256": _semantic_hash(postflight_body),
        }
        _write_exclusive_verified(
            stage_root / "runtime.postflight.json", postflight_envelope
        )
        # Re-read both in their final location before granting the custody
        # adapter permission to open any late-label file.
        self.verify_stage_seal(
            block, archive.archive_sha256, postflight.postflight_sha256
        )
        with self._lock:
            self._stage_seals[block] = (
                archive.archive_sha256,
                postflight.postflight_sha256,
            )
        return postflight

    def verify_stage_seal(
        self, block: str, archive_sha256: str, postflight_sha256: str
    ) -> None:
        _require_sha256(archive_sha256, "archive")
        _require_sha256(postflight_sha256, "postflight")
        receipts = self.receipt_bundle
        prepared = self._prepared.get(block)
        if prepared is None:
            raise TatqaP21FormalAdapterError("durable stage is not prepared")
        expected_hippo_worker_receipts = [
            self._hippo_worker_receipts[row]
            for row in sorted(self._hippo_worker_receipts)
            if row[0] == block
        ]
        expected_hippo_transport = [
            self._hippo_transport_receipts[row]
            for row in sorted(self._hippo_transport_receipts)
            if row[0] == block
        ]
        expected_transport_hashes = [
            prepared.typed_plan_transport_receipt_sha256,
            *(row[1] for row in expected_hippo_transport),
        ]
        expected_worker_pids = [
            prepared.typed_plan_worker_pid,
            *(row[2] for row in expected_hippo_transport),
        ]
        expected_transport_receipts = [
            dict(prepared.typed_plan_transport_receipt),
            *(row[0] for row in expected_hippo_transport),
        ]
        stage_root = self.control_root / "stages" / block
        inference_receipt = _strict_json(
            _read_regular(
                stage_root / "preparation.inference.json",
                field="preparation inference receipt",
            ),
            field="preparation inference receipt",
        )
        inference_receipt_sha = _verify_self_hash(
            inference_receipt,
            field="preparation_inference_receipt_sha256",
            schema=f"{VERSION}_preparation_inference_receipt_v1",
        )
        archive = _strict_json(
            _read_regular(stage_root / "action.archive.json", field="action archive"),
            field="action archive",
        )
        postflight = _strict_json(
            _read_regular(
                stage_root / "runtime.postflight.json", field="runtime postflight"
            ),
            field="runtime postflight",
        )
        _verify_self_hash(
            archive, field="durable_archive_receipt_sha256"
        )
        _verify_self_hash(
            postflight, field="durable_postflight_receipt_sha256"
        )
        transport_hashes = archive.get("transport_receipt_sha256s")
        transport_receipts = archive.get("transport_receipts")
        worker_pids = archive.get("worker_pids")
        if (
            not isinstance(transport_hashes, list)
            or not transport_hashes
            or any(
                _SHA256.fullmatch(row) is None
                for row in transport_hashes
                if isinstance(row, str)
            )
            or any(not isinstance(row, str) for row in transport_hashes)
            or not isinstance(transport_receipts, list)
            or len(transport_receipts) != len(transport_hashes)
            or any(not isinstance(row, Mapping) for row in transport_receipts)
            or [
                _semantic_hash(dict(row)) for row in transport_receipts
            ]
            != transport_hashes
            or not isinstance(worker_pids, list)
            or len(worker_pids) != len(transport_hashes)
            or any(
                isinstance(row, bool) or not isinstance(row, int) or row <= 1
                for row in worker_pids
            )
        ):
            raise TatqaP21FormalAdapterError("durable transport aggregate drifted")
        durable_qwen, durable_qwen_sha, durable_qwen_pid = (
            _validate_typed_plan_transport(
                transport_receipts[0],
                block=block,
                item_count=len(prepared.runtime_items),
                input_sha256=prepared.typed_plan_input_sha256,
                output_sha256=prepared.typed_plan_output_sha256,
            )
        )
        durable_validated: list[dict[str, Any]] = [durable_qwen]
        durable_hashes: list[str] = [durable_qwen_sha]
        durable_pids: list[int] = [durable_qwen_pid]
        inference_state = self._inference_preparations[block]
        durable_hippo_values = transport_receipts[1:]
        expected_hippo_keys = [
            row
            for row in sorted(self._hippo_transport_receipts)
            if row[0] == block
        ]
        if len(durable_hippo_values) != len(expected_hippo_keys):
            raise TatqaP21FormalAdapterError(
                "durable Hippo receipt count drifted"
            )
        for key, receipt_value in zip(
            expected_hippo_keys, durable_hippo_values, strict=True
        ):
            item_commitment = key[1]
            stored = self._hippo_transport_receipts[key][0]
            input_raw, semantic_sha = inference_state.hippo_inputs[
                item_commitment
            ]
            output_sha = stored.get("output_file_sha256")
            if not isinstance(output_sha, str):
                raise TatqaP21FormalAdapterError(
                    "durable Hippo output binding is absent"
                )
            validated, validated_sha, validated_pid = _validate_hippo_transport(
                receipt_value,
                block=block,
                item_commitment_sha256=item_commitment,
                input_file_sha256=hashlib.sha256(input_raw).hexdigest(),
                input_semantic_sha256=semantic_sha,
                output_file_sha256=output_sha,
            )
            durable_validated.append(validated)
            durable_hashes.append(validated_sha)
            durable_pids.append(validated_pid)
        if (
            durable_validated != transport_receipts
            or durable_hashes != transport_hashes
            or durable_pids != worker_pids
        ):
            raise TatqaP21FormalAdapterError(
                "durable full transport receipts failed independent replay"
            )
        expected_transport_aggregate = _semantic_hash(
            {
                "transport_receipts": transport_receipts,
                "transport_receipt_sha256s": transport_hashes,
                "worker_pids": worker_pids,
            }
        )
        expected_archive_keys = {
            "archive",
            "archive_sha256",
            "block",
            "block_preparation_receipt_sha256",
            "durable_archive_receipt_sha256",
            "hippo_worker_receipt_sha256s",
            "inference_executors_closed_after_terminal_validation",
            "preparation_inference_receipt_sha256",
            "production_canary_sha256",
            "runtime_fingerprint_sha256",
            "schema",
            "transport_receipt_aggregate_sha256",
            "transport_receipts",
            "transport_receipt_sha256s",
            "worker_pids",
        }
        expected_postflight_keys = {
            "block",
            "durable_postflight_receipt_sha256",
            "postflight",
            "postflight_sha256",
            "inference_executors_closed_after_terminal_validation",
            "preparation_inference_receipt_sha256",
            "production_canary_sha256",
            "runtime_fingerprint_sha256",
            "schema",
            "transport_receipt_aggregate_sha256",
            "transport_receipts",
            "transport_receipt_sha256s",
            "worker_pids",
        }
        if (
            set(archive) != expected_archive_keys
            or set(postflight) != expected_postflight_keys
            or archive.get("schema") != f"{VERSION}_durable_action_archive_v1"
            or archive.get("block") != block
            or archive.get("archive_sha256") != archive_sha256
            or _semantic_hash(archive.get("archive")) != archive_sha256
            or archive.get("block_preparation_receipt_sha256")
            != prepared.preparation_receipt_sha256
            or archive.get("hippo_worker_receipt_sha256s")
            != expected_hippo_worker_receipts
            or archive.get("preparation_inference_receipt_sha256")
            != inference_receipt_sha
            or postflight.get("preparation_inference_receipt_sha256")
            != inference_receipt_sha
            or archive.get(
                "inference_executors_closed_after_terminal_validation"
            )
            is not True
            or postflight.get(
                "inference_executors_closed_after_terminal_validation"
            )
            is not True
            or self._inference_preparations[block].executors_closed is not True
            or inference_receipt != self.preparation_inference_receipt(block)
            or not isinstance(archive.get("archive"), Mapping)
            or archive["archive"].get("actual_inference_preparation")
            != inference_receipt
            or [
                inference_receipt.get("qwen_transport_receipt"),
                *inference_receipt.get("hippo_transport_receipts", []),
            ]
            != transport_receipts
            or archive.get("production_canary_sha256")
            != receipts.production_canary_sha256
            or postflight.get("production_canary_sha256")
            != receipts.production_canary_sha256
            or archive.get("runtime_fingerprint_sha256")
            != receipts.runtime_fingerprint_sha256
            or postflight.get("runtime_fingerprint_sha256")
            != receipts.runtime_fingerprint_sha256
            or postflight.get("schema")
            != f"{VERSION}_durable_runtime_postflight_v1"
            or postflight.get("block") != block
            or postflight.get("postflight_sha256") != postflight_sha256
            or _semantic_hash(postflight.get("postflight")) != postflight_sha256
            or archive.get("transport_receipt_aggregate_sha256")
            != expected_transport_aggregate
            or postflight.get("transport_receipt_aggregate_sha256")
            != expected_transport_aggregate
            or postflight.get("transport_receipts") != transport_receipts
            or postflight.get("transport_receipt_sha256s") != transport_hashes
            or postflight.get("worker_pids") != worker_pids
            or transport_hashes != expected_transport_hashes
            or worker_pids != expected_worker_pids
            or transport_receipts != expected_transport_receipts
        ):
            raise TatqaP21FormalAdapterError("durable stage seal binding drifted")

    def sealed_stage_receipts(self, block: str) -> tuple[str, str] | None:
        with self._lock:
            return self._stage_seals.get(block)

    def _offline_evidence_path(self, name: str) -> Path:
        if name not in OFFLINE_EVIDENCE_NAMES:
            raise TatqaP21FormalAdapterError(
                "offline evidence name is not predeclared"
            )
        return self.control_root / "evidence" / f"{name}.json"

    def persist_offline_evidence(
        self,
        name: str,
        payload: Mapping[str, object],
        evidence_sha256: str,
    ) -> str:
        """Exclusively seal one exact offline artifact and burn replays."""

        path = self._offline_evidence_path(name)
        with self._lock:
            if name in self._offline_evidence_attempts:
                raise TatqaP21FormalAdapterError(
                    "offline evidence persistence replay is forbidden"
                )
            self._offline_evidence_attempts.add(name)
        expected = _require_sha256(evidence_sha256, "offline evidence")
        canonical_payload = _canonical_receipt_copy(
            payload, field=f"{name} offline evidence payload"
        )
        if (
            canonical_payload.get("schema") != _OFFLINE_EVIDENCE_SCHEMAS[name]
            or _semantic_hash(canonical_payload) != expected
        ):
            raise TatqaP21FormalAdapterError(
                "offline evidence semantic binding drifted"
            )
        try:
            root_metadata = path.parent.lstat()
        except OSError as exc:
            raise TatqaP21FormalAdapterError(
                "durable offline evidence root is absent"
            ) from exc
        if (
            stat.S_ISLNK(root_metadata.st_mode)
            or not stat.S_ISDIR(root_metadata.st_mode)
            or stat.S_IMODE(root_metadata.st_mode) != 0o700
        ):
            raise TatqaP21FormalAdapterError(
                "durable offline evidence root drifted"
            )
        receipts = self.receipt_bundle
        body: dict[str, object] = {
            "evidence_sha256": expected,
            "name": name,
            "payload": canonical_payload,
            "production_canary_sha256": receipts.production_canary_sha256,
            "runtime_fingerprint_sha256": receipts.runtime_fingerprint_sha256,
            "schema": f"{VERSION}_durable_offline_evidence_v1",
        }
        envelope = {
            **body,
            "durable_evidence_receipt_sha256": _semantic_hash(body),
        }
        _write_exclusive_verified(path, envelope)
        with self._lock:
            self._offline_evidence_seals[name] = expected
        self.verify_offline_evidence(name, expected)
        return expected

    def verify_offline_evidence(
        self, name: str, evidence_sha256: str
    ) -> None:
        path = self._offline_evidence_path(name)
        expected = _require_sha256(evidence_sha256, "offline evidence")
        with self._lock:
            if self._offline_evidence_seals.get(name) != expected:
                raise TatqaP21FormalAdapterError(
                    "offline evidence was not sealed by this execution"
                )
        value = _strict_json(
            _read_regular(path, field=f"{name} durable offline evidence"),
            field=f"{name} durable offline evidence",
        )
        durable_sha = _verify_self_hash(
            value, field="durable_evidence_receipt_sha256"
        )
        receipts = self.receipt_bundle
        payload = value.get("payload")
        expected_keys = {
            "durable_evidence_receipt_sha256",
            "evidence_sha256",
            "name",
            "payload",
            "production_canary_sha256",
            "runtime_fingerprint_sha256",
            "schema",
        }
        if (
            set(value) != expected_keys
            or value.get("schema")
            != f"{VERSION}_durable_offline_evidence_v1"
            or value.get("name") != name
            or value.get("evidence_sha256") != expected
            or not isinstance(payload, Mapping)
            or payload.get("schema") != _OFFLINE_EVIDENCE_SCHEMAS[name]
            or _semantic_hash(payload) != expected
            or value.get("production_canary_sha256")
            != receipts.production_canary_sha256
            or value.get("runtime_fingerprint_sha256")
            != receipts.runtime_fingerprint_sha256
            or _SHA256.fullmatch(durable_sha) is None
        ):
            raise TatqaP21FormalAdapterError(
                "durable offline evidence binding drifted"
            )

    def require_offline_evidence(self, name: str) -> str:
        """Reopen the exact currently sealed evidence before a release."""

        if name not in OFFLINE_EVIDENCE_NAMES:
            raise TatqaP21FormalAdapterError(
                "offline evidence name is not predeclared"
            )
        with self._lock:
            expected = self._offline_evidence_seals.get(name)
        if expected is None:
            raise TatqaP21FormalAdapterError(
                "required offline evidence is absent"
            )
        self.verify_offline_evidence(name, expected)
        return expected


class TrustedAcquisitionAdapter:
    """Trusted view/late-label custody boundary for one formal execution."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        runtime: ProductionRuntimeAdapter,
        control_root: str | Path | None = None,
    ) -> None:
        project = Path(project_root).resolve(strict=True)
        if not project.is_dir() or not isinstance(runtime, ProductionRuntimeAdapter):
            raise TatqaP21FormalAdapterError("adapter construction drifted")
        self.project_root = project
        self.acquisition_root = project / acquisition.ACQUISITION_ROOT_RELATIVE
        self.control_root = (
            Path(control_root)
            if control_root is not None
            else project / acquisition.FORMAL_ROOT_RELATIVE / "execution"
        )
        if self.control_root != runtime.control_root:
            raise TatqaP21FormalAdapterError("runtime/custody control roots differ")
        self._runtime = runtime
        self._receipt: dict[str, Any] | None = None
        self._claim_attempted = False
        self._claimed = False
        self._view_attempts: set[str] = set()
        self._views: dict[str, tuple[str, controller.BlockView]] = {}
        self._label_attempts: set[str] = set()
        self._m_authorization_sha256: str | None = None

    def _load_public_receipt(self) -> dict[str, Any]:
        raw = _read_regular(
            self.acquisition_root / acquisition.PUBLIC_RECEIPT_FILENAME,
            field="acquisition public receipt",
        )
        value = _strict_json(raw, field="acquisition public receipt")
        receipt_sha = _verify_self_hash(
            value, field="acquisition_receipt_sha256"
        )
        if (
            set(value) != _PUBLIC_RECEIPT_KEYS
            or value.get("schema") != f"{acquisition.VERSION}_public_receipt"
            or value.get("version") != acquisition.VERSION
            or value.get("status") != "trusted_one_shot_acquisition_complete"
            or value.get("study_design_self_sha256") != acquisition.DESIGN_SELF_SHA256
            or value.get("source_custody_self_sha256")
            != acquisition.CUSTODY_SELF_SHA256
            or value.get("fixed_block_counts") != acquisition.BLOCK_COUNTS
            or value.get("fixed_per_family_quota") != acquisition.PER_FAMILY_QUOTA
            or value.get("selected_context_count")
            != acquisition.TOTAL_SELECTED_ITEMS
            or value.get("selected_question_count")
            != acquisition.TOTAL_SELECTED_ITEMS
            or value.get("selection_secret_size_bytes") != 32
            or value.get("selection_secret_persisted_publicly") is not False
            or value.get("view_file_count") != 4
            or value.get("label_file_count") != 3
            or value.get("ledger_file_count") != 1
            or value.get("F_search_label_pack_created") is not False
            or value.get("M_search_view_and_labels_presealed") is not True
            or value.get("source_item_or_identifier_persisted_publicly") is not False
            or value.get("network_download_online_evaluator_or_model_calls") != 0
            or value.get("retry_replay_resample_or_smaller_blocks") != 0
        ):
            raise TatqaP21FormalAdapterError("acquisition public receipt drifted")
        for field in (
            "implementation_freeze_self_sha256",
            "selection_secret_commitment_sha256",
            "source_download_receipt_self_sha256",
        ):
            _require_sha256(value.get(field), field)
        if not isinstance(value.get("aggregate_qualification"), Mapping):
            raise TatqaP21FormalAdapterError("aggregate qualification drifted")
        try:
            acquisition._assert_public_receipt_safe(value)
        except Exception as exc:
            raise TatqaP21FormalAdapterError(
                "acquisition public receipt firewall failed"
            ) from exc
        bindings = value.get("private_file_bindings")
        expected_names = set(acquisition.VIEW_FILENAMES.values()).union(
            acquisition.LABEL_FILENAMES.values(), {acquisition.LEDGER_FILENAME}
        )
        if not isinstance(bindings, Mapping) or set(bindings) != expected_names:
            raise TatqaP21FormalAdapterError("private binding registry drifted")
        for name, binding in bindings.items():
            if (
                not isinstance(binding, Mapping)
                or set(binding) != _PRIVATE_BINDING_KEYS
                or binding.get("filename") != name
                or binding.get("mode") != "0600"
                or isinstance(binding.get("size_bytes"), bool)
                or not isinstance(binding.get("size_bytes"), int)
                or binding["size_bytes"] <= 0
            ):
                raise TatqaP21FormalAdapterError("private file binding drifted")
            _require_sha256(binding.get("file_sha256"), "private file")
            _require_sha256(binding.get("semantic_sha256"), "private semantic")
        _require_sha256(receipt_sha, "acquisition receipt")
        return value

    def claim_one_shot(self) -> str:
        if self._claim_attempted:
            raise TatqaP21FormalAdapterError("acquisition claim replay is forbidden")
        self._claim_attempted = True
        receipts = self._runtime.receipt_bundle
        self.control_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _mkdir_durable(self.control_root)
        attempt_body = {
            "production_canary_sha256": receipts.production_canary_sha256,
            "retry_replay_resample_provider_switch": 0,
            "runtime_fingerprint_sha256": receipts.runtime_fingerprint_sha256,
            "schema": f"{VERSION}_one_shot_execution_attempt_v1",
            "status": "consumed_before_acquisition_public_receipt_open",
        }
        _write_exclusive_verified(
            self.control_root / "execution.attempt.json",
            {
                **attempt_body,
                "attempt_sha256": _semantic_hash(attempt_body),
            },
        )
        _mkdir_durable(self.control_root / "stages")
        _mkdir_durable(self.control_root / "evidence")
        receipt = self._load_public_receipt()
        claim_body = {
            "acquisition_receipt_sha256": receipt["acquisition_receipt_sha256"],
            "production_canary_sha256": receipts.production_canary_sha256,
            "retry_replay_resample_provider_switch": 0,
            "runtime_fingerprint_sha256": receipts.runtime_fingerprint_sha256,
            "schema": f"{VERSION}_one_shot_execution_claim_v1",
            "status": "consumed_before_private_view_open",
        }
        claim = {
            **claim_body,
            "claim_sha256": _semantic_hash(claim_body),
        }
        _write_exclusive_verified(self.control_root / "execution.claim.json", claim)
        self._receipt = receipt
        self._claimed = True
        return receipt["acquisition_receipt_sha256"]

    def _binding(self, filename: str) -> Mapping[str, Any]:
        if self._receipt is None:
            raise TatqaP21FormalAdapterError("acquisition is not claimed")
        binding = self._receipt["private_file_bindings"].get(filename)
        if not isinstance(binding, Mapping):
            raise TatqaP21FormalAdapterError("private binding is absent")
        return binding

    def _read_bound_private(
        self, *, filename: str, semantic_field: str, field: str
    ) -> dict[str, Any]:
        if Path(filename).name != filename:
            raise TatqaP21FormalAdapterError("private filename is unsafe")
        binding = self._binding(filename)
        path = self.acquisition_root / filename
        raw = _read_regular(path, field=field, expected_mode=0o600)
        if (
            len(raw) != binding["size_bytes"]
            or hashlib.sha256(raw).hexdigest() != binding["file_sha256"]
        ):
            raise TatqaP21FormalAdapterError("private file binding failed")
        value = _strict_json(raw, field=field)
        semantic = _verify_self_hash(value, field=semantic_field)
        if semantic != binding["semantic_sha256"]:
            raise TatqaP21FormalAdapterError("private semantic binding failed")
        return value

    def _seal_m_authorization(
        self, authorization: controller.EpochAuthorization
    ) -> None:
        if self._m_authorization_sha256 is not None:
            raise TatqaP21FormalAdapterError("M authorization replay is forbidden")
        # Reopen the exact promotion and frozen-policy artifacts before even
        # creating the authorization seal that can unlock the M_search path.
        self._runtime.verify_offline_evidence(
            "A_hold_score", authorization.a_hold_score_sha256
        )
        self._runtime.verify_offline_evidence(
            "F_search_policy_freeze", authorization.policy_freeze_sha256
        )
        receipts = self._runtime.receipt_bundle
        body = {
            "authorization": authorization.payload(),
            "authorization_sha256": authorization.authorization_sha256,
            "production_canary_sha256": receipts.production_canary_sha256,
            "runtime_fingerprint_sha256": receipts.runtime_fingerprint_sha256,
            "schema": f"{VERSION}_M_search_epoch_authorization_seal_v1",
        }
        path = self.control_root / "M_search.epoch_authorization.sealed.json"
        _write_exclusive_verified(path, body)
        reopened = _strict_json(_read_regular(path, field="M authorization"), field="M authorization")
        if (
            reopened != body
            or _semantic_hash(reopened["authorization"])
            != authorization.authorization_sha256
        ):
            raise TatqaP21FormalAdapterError("M authorization seal drifted")
        self._m_authorization_sha256 = authorization.authorization_sha256

    def load_block_view(
        self,
        block: str,
        authorization: controller.EpochAuthorization | None,
    ) -> controller.BlockView:
        if not self._claimed or block not in acquisition.BLOCK_ORDER:
            raise TatqaP21FormalAdapterError("view request is not authorized")
        if block in self._view_attempts:
            raise TatqaP21FormalAdapterError("view release replay is forbidden")
        self._view_attempts.add(block)
        if block == "M_search":
            if not isinstance(authorization, controller.EpochAuthorization):
                # This branch occurs before resolving, statting, or opening the
                # presealed M view path.
                raise TatqaP21FormalAdapterError("M view lacks epoch authorization")
            self._seal_m_authorization(authorization)
        elif authorization is not None:
            raise TatqaP21FormalAdapterError("premature epoch authorization")
        elif block == "F_search":
            self._runtime.require_offline_evidence("A_form_fit")
        elif block == "A_hold":
            self._runtime.require_offline_evidence(
                "F_search_policy_freeze"
            )
        filename = acquisition.VIEW_FILENAMES[block]
        value = self._read_bound_private(
            filename=filename,
            semantic_field="block_view_sha256",
            field=f"{block} private view",
        )
        expected_access = (
            "presealed_until_valid_A_hold_promotion"
            if block == "M_search"
            else "available_only_at_frozen_lifecycle_stage"
        )
        if (
            set(value) != _VIEW_KEYS
            or value.get("schema") != f"{acquisition.VERSION}_block_view"
            or value.get("version") != acquisition.VERSION
            or value.get("block") != block
            or value.get("access_state") != expected_access
            or value.get("item_count") != acquisition.BLOCK_COUNTS[block]
            or value.get("late_fields_included") is not False
        ):
            raise TatqaP21FormalAdapterError("private block view shape drifted")
        try:
            acquisition.assert_view_firewall(value)
        except Exception as exc:
            raise TatqaP21FormalAdapterError("private block view firewall failed") from exc
        raw_items = value.get("items")
        if not isinstance(raw_items, list) or len(raw_items) != acquisition.BLOCK_COUNTS[block]:
            raise TatqaP21FormalAdapterError("private block item count drifted")
        runtime_items: list[features.LabelFreeRuntimeItem] = []
        for ordinal, row in enumerate(raw_items):
            if (
                not isinstance(row, Mapping)
                or set(row) != _VIEW_ITEM_KEYS
                or row.get("ordinal") != ordinal
                or not isinstance(row.get("canonical_units"), list)
            ):
                raise TatqaP21FormalAdapterError("private block item shape drifted")
            units_raw = row["canonical_units"]
            if any(not isinstance(unit, Mapping) or set(unit) != _UNIT_KEYS for unit in units_raw):
                raise TatqaP21FormalAdapterError("private canonical unit shape drifted")
            try:
                units = tuple(
                    features.RuntimeUnit(unit_id=unit["unit_id"], text=unit["text"])
                    for unit in units_raw
                )
                runtime_item = features.LabelFreeRuntimeItem(
                    item_id=row["item_commitment_sha256"],
                    question=row["question"],
                    units=units,
                )
                custody_units = tuple(
                    acquisition.CanonicalUnit(unit_id=unit.unit_id, text=unit.text)
                    for unit in units
                )
                expected_commitment = acquisition.item_commitment(
                    block=block,
                    ordinal=ordinal,
                    question=runtime_item.question,
                    units=custody_units,
                )
            except Exception as exc:
                raise TatqaP21FormalAdapterError("private block item is invalid") from exc
            if runtime_item.item_id != expected_commitment:
                raise TatqaP21FormalAdapterError("item commitment binding drifted")
            runtime_items.append(runtime_item)
        if len({row.item_id for row in runtime_items}) != len(runtime_items):
            raise TatqaP21FormalAdapterError("private block commitments are duplicated")
        block_view = self._runtime.prepare_block(
            block=block,
            block_view_sha256=value["block_view_sha256"],
            items=tuple(runtime_items),
        )
        self._views[block] = (value["block_view_sha256"], block_view)
        return block_view

    def release_label_pack(
        self,
        block: str,
        archive_sha256: str,
        postflight_sha256: str,
    ) -> controller.LabelPack:
        if (
            not self._claimed
            or block not in acquisition.LABEL_FILENAMES
            or block not in self._views
        ):
            raise TatqaP21FormalAdapterError("label release is not authorized")
        if block in self._label_attempts:
            raise TatqaP21FormalAdapterError("label release replay is forbidden")
        self._label_attempts.add(block)
        if self._runtime.sealed_stage_receipts(block) != (
            archive_sha256,
            postflight_sha256,
        ):
            raise TatqaP21FormalAdapterError("labels lack durable stage receipts")
        # Both files are reopened and semantically checked before the label
        # filename is resolved or touched.
        self._runtime.verify_stage_seal(block, archive_sha256, postflight_sha256)
        if block == "M_search":
            authorization_path = self.control_root / "M_search.epoch_authorization.sealed.json"
            authorization = _strict_json(
                _read_regular(authorization_path, field="M authorization"),
                field="M authorization",
            )
            if authorization.get("authorization_sha256") != self._m_authorization_sha256:
                raise TatqaP21FormalAdapterError("M label authorization drifted")
        filename = acquisition.LABEL_FILENAMES[block]
        value = self._read_bound_private(
            filename=filename,
            semantic_field="label_pack_sha256",
            field=f"{block} sealed labels",
        )
        view_sha, view = self._views[block]
        expected_access = (
            "presealed_until_valid_A_hold_promotion"
            if block == "M_search"
            else "sealed_until_corresponding_actions_and_postflight"
        )
        if (
            set(value) != _LABEL_KEYS
            or value.get("schema") != f"{acquisition.VERSION}_sealed_labels"
            or value.get("version") != acquisition.VERSION
            or value.get("block") != block
            or value.get("access_state") != expected_access
            or value.get("item_count") != acquisition.BLOCK_COUNTS[block]
            or value.get("block_view_sha256") != view_sha
        ):
            raise TatqaP21FormalAdapterError("sealed label-pack shape drifted")
        raw_items = value.get("items")
        if not isinstance(raw_items, list) or len(raw_items) != len(view.items):
            raise TatqaP21FormalAdapterError("sealed label item count drifted")
        rows: list[controller.LabelRow] = []
        for ordinal, (raw, item) in enumerate(zip(raw_items, view.items, strict=True)):
            if (
                not isinstance(raw, Mapping)
                or set(raw) != _LABEL_ITEM_KEYS
                or raw.get("ordinal") != ordinal
                or raw.get("item_commitment_sha256")
                != item.item_commitment_sha256
                or not isinstance(raw.get("gold_unit_ids"), list)
            ):
                raise TatqaP21FormalAdapterError("sealed label item shape drifted")
            try:
                row = controller.LabelRow(
                    item_commitment_sha256=item.item_commitment_sha256,
                    family=raw["family"],
                    canonical_gold_units=tuple(raw["gold_unit_ids"]),
                )
            except Exception as exc:
                raise TatqaP21FormalAdapterError("sealed label item is invalid") from exc
            corpus = {unit.unit_id for unit in item.units}
            if not set(row.canonical_gold_units).issubset(corpus):
                raise TatqaP21FormalAdapterError("sealed gold lies outside item corpus")
            rows.append(row)
        return controller.LabelPack(block=block, rows=tuple(rows))


__all__ = [
    "ACTUAL_HIPPO_INFERENCE_CAP",
    "HippoRunner",
    "HIPPO_TRANSPORT_SCHEMA",
    "OFFLINE_EVIDENCE_NAMES",
    "PRODUCTION_CANARY_SCHEMA",
    "PRODUCTION_CANARY_SELF_FIELD",
    "ProductionRuntimeAdapter",
    "RUNTIME_FINGERPRINT_SCHEMA",
    "RUNTIME_FINGERPRINT_SELF_FIELD",
    "RuntimeReceiptPaths",
    "STANDARD_EXECUTOR_CAP",
    "SYSTEMD_NETWORK_PROPERTIES",
    "SYSTEMD_FILESYSTEM_ISOLATION",
    "TatqaP21FormalAdapterError",
    "TrustedAcquisitionAdapter",
    "TYPED_PLAN_TRANSPORT_SCHEMA",
    "TypedPlanBatchRunner",
    "VERSION",
]
