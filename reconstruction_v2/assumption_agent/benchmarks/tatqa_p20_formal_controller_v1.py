"""Injected, offline-only lifecycle controller for the frozen TAT-QA P20 study.

This module cannot locate, stat, or open TAT-QA source files or sealed label
packs.  A trusted :class:`AcquisitionBoundary` supplies already-sealed,
content-free views and releases a label pack only after receiving both a
validated action-archive hash and a zero-side-effect runtime-postflight hash.
Likewise, all runtime capabilities and executors are injected through
:class:`RuntimeBoundary`; the controller has no network or model-provider API.

The controller is deliberately one shot.  Every available logical action in a
stage is submitted before the first future result is joined.  Official
HippoRAG work uses a distinct injected executor plus an eight-party bounded
semaphore and a live-call peak counter around the actual runtime call.
"""

from __future__ import annotations

from collections import Counter
import concurrent.futures
from contextlib import AbstractContextManager
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re
import threading
from typing import Mapping, Protocol, Sequence

from . import tatqa_p20_typed_evaluator_core_v1 as evaluator


VERSION = "tatqa_p20_formal_controller_v1"
HIPPO_CONCURRENCY_CAP = 8
HIPPO_CPU_THREADS_PER_PROCESS = 2
SYSTEMD_NETWORK_PROPERTIES = (
    "IPAddressDeny=any",
    "RestrictAddressFamilies=AF_UNIX",
)
SYSTEMD_FILESYSTEM_ISOLATION = (
    "systemd_InaccessiblePaths_official_source_and_acquisition_v1"
)
TYPED_PLAN_TRANSPORT_SCHEMA = (
    "tatqa_p20_formal_runtime_v1_typed_plan_transport_receipt_v1"
)
HIPPO_TRANSPORT_SCHEMA = (
    "tatqa_p20_formal_runtime_v1_hippo_transport_receipt_v1"
)
SYSTEMD_UNIT_CLOSURE_SCHEMA = (
    "tatqa_p20_formal_runtime_v1_systemd_unit_closure_v1"
)
SYSTEMD_START_POLICY_SCHEMA = (
    "tatqa_p20_formal_runtime_v1_systemd_start_policy_v1"
)

FAMILY_ORDER = ("TABLE", "TEXT", "TABLE_TEXT")
BLOCK_ITEM_COUNTS = {
    "A_form": 48,
    "F_search": 36,
    "A_hold": 30,
    "M_search": 30,
}
BLOCK_FAMILY_COUNTS = {
    "A_form": 16,
    "A_hold": 10,
    "M_search": 10,
}
HIPPO_ARM = "official_HippoRAG_retrieve_only"
BLOCK_ARMS = {
    "A_form": ("P0", "P1"),
    "F_search": ("P0", "P1"),
    "A_hold": ("E0", "E1", "RAW", HIPPO_ARM),
    "M_search": ("E0", "E1", "RAW", HIPPO_ARM),
}
# E0/E1 are deterministic label-free selections over P0/P1 terminals.  The
# actual eager work in A_hold and M_search is therefore P0/P1/RAW/Hippo; both
# derived evaluator actions are added to the same archive before postflight.
BLOCK_WORK_ARMS = {
    "A_form": ("P0", "P1"),
    "F_search": ("P0", "P1"),
    "A_hold": ("P0", "P1", "RAW", HIPPO_ARM),
    "M_search": ("P0", "P1", "RAW", HIPPO_ARM),
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_UNIT_ID = re.compile(r"(?:T:(?:0|[1-9][0-9]*)|P:[1-9][0-9]*)\Z")


class TatqaP20FormalControllerError(RuntimeError):
    """Raised internally when a frozen lifecycle invariant drifts."""


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TatqaP20FormalControllerError(f"{field} must be a lowercase SHA-256")
    return value


def _require_unit_id(value: object) -> str:
    if not isinstance(value, str) or _UNIT_ID.fullmatch(value) is None:
        raise TatqaP20FormalControllerError("canonical unit ID drifted")
    return value


def _fraction_payload(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _canonical_hash(value: object) -> str:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise TatqaP20FormalControllerError(
            "controller hash payload is not canonical JSON"
        ) from exc
    return hashlib.sha256(raw).hexdigest()


def _exception_type_sha256(exc: BaseException) -> str:
    return hashlib.sha256(
        f"{type(exc).__module__}.{type(exc).__qualname__}".encode("utf-8")
    ).hexdigest()


def _strict_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TatqaP20FormalControllerError(f"{field} must be a positive integer")
    return value


def _validate_systemd_unit_closure(
    value: object, *, unit_name_sha256: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TatqaP20FormalControllerError("systemd unit closure is absent")
    receipt = dict(value)
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
        raise TatqaP20FormalControllerError("systemd unit closure drifted")
    return receipt


def _validate_systemd_start_policy(
    value: object, *, unit_name_sha256: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TatqaP20FormalControllerError("systemd start policy is absent")
    receipt = dict(value)
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
        raise TatqaP20FormalControllerError("systemd start policy drifted")
    return receipt


def _validate_qwen_transport_receipt(
    value: object, *, block: str, item_count: int
) -> tuple[dict[str, object], int, int, int]:
    if not isinstance(value, Mapping):
        raise TatqaP20FormalControllerError("Qwen transport receipt is absent")
    receipt = dict(value)
    expected_keys = {
        "batch_size",
        "block",
        "filesystem_isolation",
        "input_sha256",
        "item_count",
        "model_context_tokens",
        "model_execution_finished_monotonic_ns",
        "model_execution_started_monotonic_ns",
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
    started = _strict_positive_int(
        receipt.get("model_execution_started_monotonic_ns"),
        "Qwen model interval start",
    )
    finished = _strict_positive_int(
        receipt.get("model_execution_finished_monotonic_ns"),
        "Qwen model interval finish",
    )
    pid = _strict_positive_int(receipt.get("worker_pid"), "Qwen worker PID")
    unit_name_sha256 = _require_sha256(
        receipt.get("systemd_unit_name_sha256"), "Qwen systemd unit name"
    )
    _validate_systemd_unit_closure(
        receipt.get("systemd_unit_closure"),
        unit_name_sha256=unit_name_sha256,
    )
    context = _strict_positive_int(
        receipt.get("model_context_tokens"), "Qwen context"
    )
    if (
        set(receipt) != expected_keys
        or receipt.get("schema") != TYPED_PLAN_TRANSPORT_SCHEMA
        or receipt.get("block") != block
        or type(receipt.get("item_count")) is not int
        or receipt["item_count"] != item_count
        or type(receipt.get("batch_size")) is not int
        or receipt["batch_size"] != 4
        or receipt.get("physical_GPU") != "1"
        or receipt.get("filesystem_isolation") != SYSTEMD_FILESYSTEM_ISOLATION
        or receipt.get("network_properties") != list(SYSTEMD_NETWORK_PROPERTIES)
        or context < 16_640
        or finished <= started
        or any(
            not isinstance(receipt.get(field), str)
            or _SHA256.fullmatch(receipt[field]) is None
            for field in (
                "input_sha256",
                "output_sha256",
                "stderr_sha256",
                "stdout_sha256",
            )
        )
    ):
        raise TatqaP20FormalControllerError("Qwen transport receipt drifted")
    return receipt, started, finished, pid


def _validate_hippo_transport_receipt(
    value: object, *, block: str, item_commitments: set[str]
) -> tuple[dict[str, object], str, int, int, int]:
    if not isinstance(value, Mapping):
        raise TatqaP20FormalControllerError("Hippo transport receipt is absent")
    receipt = dict(value)
    expected_keys = {
        "CPU_threads",
        "block",
        "configured_torch_interop_threads",
        "configured_torch_intraop_threads",
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
    }
    item = receipt.get("item_commitment_sha256")
    started = _strict_positive_int(
        receipt.get("model_execution_started_monotonic_ns"),
        "Hippo model interval start",
    )
    finished = _strict_positive_int(
        receipt.get("model_execution_finished_monotonic_ns"),
        "Hippo model interval finish",
    )
    pid = _strict_positive_int(receipt.get("worker_pid"), "Hippo worker PID")
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
    thread_values = tuple(
        _strict_positive_int(receipt.get(field), field)
        for field in (
            "configured_torch_intraop_threads",
            "configured_torch_interop_threads",
            "observed_process_thread_peak",
        )
    )
    if (
        set(receipt) != expected_keys
        or receipt.get("schema") != HIPPO_TRANSPORT_SCHEMA
        or receipt.get("block") != block
        or not isinstance(item, str)
        or item not in item_commitments
        or type(receipt.get("CPU_threads")) is not int
        or receipt["CPU_threads"] != HIPPO_CPU_THREADS_PER_PROCESS
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
        or _canonical_hash(start_policy) != start_policy_sha256
        or finished <= started
        or any(row > HIPPO_CPU_THREADS_PER_PROCESS for row in thread_values)
        or any(
            not isinstance(receipt.get(field), str)
            or _SHA256.fullmatch(receipt[field]) is None
            for field in (
                "input_file_sha256",
                "input_semantic_sha256",
                "output_file_sha256",
                "stderr_sha256",
                "stdout_sha256",
            )
        )
    ):
        raise TatqaP20FormalControllerError("Hippo transport receipt drifted")
    return receipt, item, started, finished, pid


@dataclass(frozen=True)
class RuntimePreflight:
    qualified: bool
    public_synthetic_distinct_rankings: bool
    public_synthetic_p0_behavior_sha256: str
    public_synthetic_p1_behavior_sha256: str
    external_network_calls: int
    api_or_online_evaluator_calls: int
    retry_replay_resample_provider_switch: int

    def __post_init__(self) -> None:
        p0_hash = _require_sha256(
            self.public_synthetic_p0_behavior_sha256,
            "public synthetic P0 behavior",
        )
        p1_hash = _require_sha256(
            self.public_synthetic_p1_behavior_sha256,
            "public synthetic P1 behavior",
        )
        if (
            self.qualified is not True
            or self.public_synthetic_distinct_rankings is not True
            or p0_hash == p1_hash
            or self.external_network_calls != 0
            or self.api_or_online_evaluator_calls != 0
            or self.retry_replay_resample_provider_switch != 0
        ):
            raise TatqaP20FormalControllerError("runtime preflight is not qualified")

    def payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_runtime_preflight_v1",
            "qualified": self.qualified,
            "public_synthetic_distinct_rankings": (
                self.public_synthetic_distinct_rankings
            ),
            "public_synthetic_p0_behavior_sha256": (
                self.public_synthetic_p0_behavior_sha256
            ),
            "public_synthetic_p1_behavior_sha256": (
                self.public_synthetic_p1_behavior_sha256
            ),
            "external_network_calls": self.external_network_calls,
            "api_or_online_evaluator_calls": self.api_or_online_evaluator_calls,
            "retry_replay_resample_provider_switch": (
                self.retry_replay_resample_provider_switch
            ),
        }

    @property
    def receipt_sha256(self) -> str:
        return _canonical_hash(self.payload())


@dataclass(frozen=True)
class ItemView:
    """One label-free item commitment and its already-quantized core inputs."""

    item_commitment_sha256: str
    plan: evaluator.TypedPlan
    units: tuple[evaluator.CanonicalUnit, ...]
    redundancy_features: tuple[tuple[str, str, int], ...] = ()

    def __post_init__(self) -> None:
        _require_sha256(self.item_commitment_sha256, "item commitment")
        plan = evaluator.validate_typed_plan(self.plan)
        units = tuple(self.units)
        if (
            not 5 <= len(units) <= 96
            or any(not isinstance(row, evaluator.CanonicalUnit) for row in units)
        ):
            raise TatqaP20FormalControllerError(
                "item view must contain five through 96 canonical units"
            )
        unit_ids = tuple(_require_unit_id(row.unit_id) for row in units)
        if len(set(unit_ids)) != len(unit_ids):
            raise TatqaP20FormalControllerError("item view unit IDs are not unique")
        if any(len(row.facet_coverage) != plan.facet_width for row in units):
            raise TatqaP20FormalControllerError(
                "item view facet width does not match the typed plan"
            )
        known_ids = set(unit_ids)
        redundancy: dict[tuple[str, str], int] = {}
        for row in self.redundancy_features:
            if (
                not isinstance(row, tuple)
                or len(row) != 3
                or not isinstance(row[0], str)
                or not isinstance(row[1], str)
                or type(row[2]) is not int
                or row[2] < 0
                or row[0] == row[1]
                or row[0] not in known_ids
                or row[1] not in known_ids
            ):
                raise TatqaP20FormalControllerError(
                    "item redundancy feature row drifted"
                )
            pair = tuple(sorted((row[0], row[1])))
            if pair in redundancy:
                raise TatqaP20FormalControllerError("duplicate redundancy feature row")
            redundancy[pair] = row[2]
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "redundancy_features", tuple(self.redundancy_features))

    def redundancy_mapping(self) -> dict[tuple[str, str], int]:
        return {(left, right): value for left, right, value in self.redundancy_features}


@dataclass(frozen=True)
class BlockView:
    block: str
    items: tuple[ItemView, ...]

    def __post_init__(self) -> None:
        if self.block not in BLOCK_ITEM_COUNTS:
            raise TatqaP20FormalControllerError("unknown block view")
        rows = tuple(self.items)
        if (
            len(rows) != BLOCK_ITEM_COUNTS[self.block]
            or any(not isinstance(row, ItemView) for row in rows)
        ):
            raise TatqaP20FormalControllerError("block view item count drifted")
        commitments = tuple(row.item_commitment_sha256 for row in rows)
        if len(set(commitments)) != len(commitments):
            raise TatqaP20FormalControllerError("block view commitments are not unique")
        object.__setattr__(self, "items", rows)


@dataclass(frozen=True)
class LabelRow:
    """Trusted late label row; family is absent from every ItemView."""

    item_commitment_sha256: str
    family: str
    canonical_gold_units: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.item_commitment_sha256, "label item commitment")
        if self.family not in FAMILY_ORDER:
            raise TatqaP20FormalControllerError("label family drifted")
        gold = tuple(self.canonical_gold_units)
        if not 1 <= len(gold) <= evaluator.TOP_K or len(set(gold)) != len(gold):
            raise TatqaP20FormalControllerError("canonical gold-unit count drifted")
        for unit_id in gold:
            _require_unit_id(unit_id)
        object.__setattr__(self, "canonical_gold_units", gold)

    def payload(self) -> dict[str, object]:
        return {
            "item_commitment_sha256": self.item_commitment_sha256,
            "family": self.family,
            "canonical_gold_units": list(self.canonical_gold_units),
        }


@dataclass(frozen=True)
class LabelPack:
    block: str
    rows: tuple[LabelRow, ...]

    def __post_init__(self) -> None:
        if self.block not in BLOCK_FAMILY_COUNTS:
            raise TatqaP20FormalControllerError("labels are forbidden for this block")
        rows = tuple(self.rows)
        if (
            len(rows) != BLOCK_ITEM_COUNTS[self.block]
            or any(not isinstance(row, LabelRow) for row in rows)
        ):
            raise TatqaP20FormalControllerError("label-pack row count drifted")
        commitments = tuple(row.item_commitment_sha256 for row in rows)
        if len(set(commitments)) != len(commitments):
            raise TatqaP20FormalControllerError("label-pack commitments are not unique")
        family_counts = Counter(row.family for row in rows)
        expected = BLOCK_FAMILY_COUNTS[self.block]
        if family_counts != Counter({family: expected for family in FAMILY_ORDER}):
            raise TatqaP20FormalControllerError("label-pack family balance drifted")
        object.__setattr__(self, "rows", rows)

    def payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_trusted_label_pack_commitment_v1",
            "block": self.block,
            "rows": [row.payload() for row in self.rows],
        }

    @property
    def label_pack_sha256(self) -> str:
        return _canonical_hash(self.payload())


@dataclass(frozen=True)
class LogicalActionResult:
    """One strict terminal for a single item/logical-arm work unit."""

    block: str
    item_commitment_sha256: str
    logical_arm: str
    selected_policy_id: str
    selected_unit_ids: tuple[str, ...]
    p1_minus_p0_features: tuple[int, ...]
    source_action_sha256: str

    def __post_init__(self) -> None:
        allowed_arms = set(BLOCK_WORK_ARMS.get(self.block, ())).union(
            BLOCK_ARMS.get(self.block, ())
        )
        if self.block not in BLOCK_ARMS or self.logical_arm not in allowed_arms:
            raise TatqaP20FormalControllerError("logical action block/arm drifted")
        _require_sha256(self.item_commitment_sha256, "action item commitment")
        _require_sha256(self.source_action_sha256, "source action")
        selected = tuple(self.selected_unit_ids)
        for unit_id in selected:
            _require_unit_id(unit_id)
        # The core behavior hash additionally validates distinct top-five
        # cardinality after the controller's stricter paragraph-order check.
        evaluator.canonical_behavior_hash(selected)
        features = evaluator.feature_vector(self.p1_minus_p0_features)
        if self.logical_arm == "P0" and self.selected_policy_id != evaluator.P0_POLICY_ID:
            raise TatqaP20FormalControllerError("P0 terminal policy drifted")
        if self.logical_arm == "P1" and self.selected_policy_id != evaluator.P1_POLICY_ID:
            raise TatqaP20FormalControllerError("P1 terminal policy drifted")
        if self.logical_arm == "E0" and self.selected_policy_id != evaluator.P0_POLICY_ID:
            raise TatqaP20FormalControllerError("E0 terminal policy drifted")
        if self.logical_arm == "E1" and self.selected_policy_id not in {
            evaluator.P0_POLICY_ID,
            evaluator.P1_POLICY_ID,
        }:
            raise TatqaP20FormalControllerError("E1 terminal policy drifted")
        if self.logical_arm in {"RAW", HIPPO_ARM}:
            if self.selected_policy_id != self.logical_arm or any(features):
                raise TatqaP20FormalControllerError("baseline terminal drifted")
        object.__setattr__(self, "selected_unit_ids", selected)
        object.__setattr__(self, "p1_minus_p0_features", features)

    @property
    def behavior_sha256(self) -> str:
        return evaluator.canonical_behavior_hash(self.selected_unit_ids)

    def payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_logical_action_result_v1",
            "block": self.block,
            "item_commitment_sha256": self.item_commitment_sha256,
            "logical_arm": self.logical_arm,
            "selected_policy_id": self.selected_policy_id,
            "ordered_top5": list(self.selected_unit_ids),
            "fixed_feature_order": list(evaluator.FEATURE_ORDER),
            "P1_minus_P0_features": list(self.p1_minus_p0_features),
            "source_action_sha256": self.source_action_sha256,
            "behavior_sha256": self.behavior_sha256,
        }

    @property
    def action_result_sha256(self) -> str:
        return _canonical_hash(self.payload())


@dataclass(frozen=True)
class StageArchive:
    block: str
    actions: tuple[LogicalActionResult, ...]
    candidate_work_actions: tuple[LogicalActionResult, ...]
    inference_preparation_receipt: Mapping[str, object]
    submitted_work_count: int
    submitted_work_terminal_count: int
    logical_action_count: int
    logical_action_terminal_count: int
    submit_count_before_first_join: int
    eager_submit_complete_before_first_join: bool
    hippo_executor_dedicated: bool
    hippo_concurrency_cap: int
    hippo_observed_peak: int

    def __post_init__(self) -> None:
        if self.block not in BLOCK_ARMS:
            raise TatqaP20FormalControllerError("unknown stage archive block")
        actions = tuple(self.actions)
        expected_submitted_count = (
            BLOCK_ITEM_COUNTS[self.block] * len(BLOCK_WORK_ARMS[self.block])
        )
        expected_logical_count = (
            BLOCK_ITEM_COUNTS[self.block] * len(BLOCK_ARMS[self.block])
        )
        if (
            self.submitted_work_count != expected_submitted_count
            or self.submitted_work_terminal_count != expected_submitted_count
            or self.logical_action_count != expected_logical_count
            or self.logical_action_terminal_count != expected_logical_count
            or self.submit_count_before_first_join != expected_submitted_count
            or self.eager_submit_complete_before_first_join is not True
            or len(actions) != self.logical_action_terminal_count
            or any(
                not isinstance(row, LogicalActionResult) or row.block != self.block
                for row in actions
            )
        ):
            raise TatqaP20FormalControllerError("stage eager/terminal accounting drifted")
        keys = tuple((row.item_commitment_sha256, row.logical_arm) for row in actions)
        if len(set(keys)) != len(keys):
            raise TatqaP20FormalControllerError("duplicate logical action terminal")
        by_item: dict[str, set[str]] = {}
        for item, arm in keys:
            by_item.setdefault(item, set()).add(arm)
        if (
            len(by_item) != BLOCK_ITEM_COUNTS[self.block]
            or any(arms != set(BLOCK_ARMS[self.block]) for arms in by_item.values())
        ):
            raise TatqaP20FormalControllerError(
                "predeclared logical arm coverage drifted"
            )
        candidate_actions = tuple(self.candidate_work_actions)
        if self.block in {"A_hold", "M_search"}:
            if (
                len(candidate_actions) != 2 * BLOCK_ITEM_COUNTS[self.block]
                or any(
                    not isinstance(row, LogicalActionResult)
                    or row.block != self.block
                    or row.logical_arm not in {"P0", "P1"}
                    for row in candidate_actions
                )
            ):
                raise TatqaP20FormalControllerError(
                    "shared P0/P1 candidate-work coverage drifted"
                )
            candidate_keys = {
                (row.item_commitment_sha256, row.logical_arm)
                for row in candidate_actions
            }
            expected_candidate_keys = {
                (item, arm) for item in by_item for arm in ("P0", "P1")
            }
            if candidate_keys != expected_candidate_keys:
                raise TatqaP20FormalControllerError(
                    "shared P0/P1 candidate-work keys drifted"
                )
            candidate_by_key = {
                (row.item_commitment_sha256, row.logical_arm): row
                for row in candidate_actions
            }
            logical_by_key = {
                (row.item_commitment_sha256, row.logical_arm): row
                for row in actions
            }
            for item in by_item:
                p0 = candidate_by_key[(item, "P0")]
                p1 = candidate_by_key[(item, "P1")]
                e0 = logical_by_key[(item, "E0")]
                e1 = logical_by_key[(item, "E1")]
                if (
                    e0.selected_unit_ids != p0.selected_unit_ids
                    or e0.source_action_sha256 != p0.source_action_sha256
                    or (
                        e1.selected_unit_ids,
                        e1.source_action_sha256,
                    )
                    not in {
                        (p0.selected_unit_ids, p0.source_action_sha256),
                        (p1.selected_unit_ids, p1.source_action_sha256),
                    }
                ):
                    raise TatqaP20FormalControllerError(
                        "E0/E1 action is not derived from sealed P0/P1 work"
                    )
        elif candidate_actions:
            raise TatqaP20FormalControllerError(
                "formation stage unexpectedly contains shared candidate work"
            )
        submitted_actions = self._submitted_actions(actions, candidate_actions)
        submitted_keys = {
            (row.item_commitment_sha256, row.logical_arm)
            for row in submitted_actions
        }
        expected_submitted_keys = {
            (item, arm) for item in by_item for arm in BLOCK_WORK_ARMS[self.block]
        }
        if (
            len(submitted_actions) != self.submitted_work_terminal_count
            or len(submitted_keys) != len(submitted_actions)
            or submitted_keys != expected_submitted_keys
        ):
            raise TatqaP20FormalControllerError(
                "submitted-work terminal coverage drifted"
            )
        has_hippo = HIPPO_ARM in BLOCK_WORK_ARMS[self.block]
        inference = dict(self.inference_preparation_receipt)
        inference_self = inference.get("preparation_inference_receipt_sha256")
        inference_body = dict(inference)
        inference_body.pop("preparation_inference_receipt_sha256", None)
        expected_hippo_items = BLOCK_ITEM_COUNTS[self.block] if has_hippo else 0
        expected_inference_keys = {
            "actual_model_future_expected_count",
            "actual_model_future_submit_count_before_first_join",
            "all_actual_model_futures_submitted_before_first_join",
            "block",
            "hippo_actual_concurrency_cap",
            "hippo_actual_observed_peak",
            "hippo_future_submitted_count",
            "hippo_future_terminal_count",
            "hippo_future_consumed_count",
            "hippo_transport_receipts",
            "hippo_transport_receipt_sha256s",
            "hippo_worker_pids",
            "hippo_worker_receipt_sha256s",
            "hippo_executor_dedicated",
            "minilm_raw_compiled_item_count",
            "qwen_batch_item_count",
            "qwen_batch_submitted_count",
            "qwen_batch_terminal_count",
            "qwen_executor_dedicated",
            "qwen_hippo_independent_executors",
            "qwen_hippo_overlap_observed",
            "qwen_hippo_overlap_witness_item_commitments",
            "qwen_transport_receipt",
            "qwen_transport_receipt_sha256",
            "qwen_worker_pid",
            "qwen_worker_receipt_sha256",
            "retry_replay_resample_provider_switch",
            "schema",
            "preparation_inference_receipt_sha256",
        }
        hippo_transport = inference.get("hippo_transport_receipt_sha256s")
        hippo_transport_receipts = inference.get("hippo_transport_receipts")
        hippo_workers = inference.get("hippo_worker_receipt_sha256s")
        hippo_pids = inference.get("hippo_worker_pids")
        qwen_transport, qwen_started, qwen_finished, qwen_pid = (
            _validate_qwen_transport_receipt(
                inference.get("qwen_transport_receipt"),
                block=self.block,
                item_count=BLOCK_ITEM_COUNTS[self.block],
            )
        )
        if not isinstance(hippo_transport_receipts, list):
            raise TatqaP20FormalControllerError(
                "full Hippo transport receipts are absent"
            )
        validated_hippo = tuple(
            _validate_hippo_transport_receipt(
                row, block=self.block, item_commitments=set(by_item)
            )
            for row in hippo_transport_receipts
        )
        overlap_witnesses = [
            row[1]
            for row in validated_hippo
            if max(qwen_started, row[2]) < min(qwen_finished, row[3])
        ]
        declared_witnesses = inference.get(
            "qwen_hippo_overlap_witness_item_commitments"
        )
        if (
            set(inference) != expected_inference_keys
            or inference.get("schema")
            != "tatqa_p20_formal_adapters_v1_preparation_inference_receipt_v1"
            or inference.get("block") != self.block
            or not isinstance(inference_self, str)
            or _SHA256.fullmatch(inference_self) is None
            or _canonical_hash(inference_body) != inference_self
            or inference.get("actual_model_future_expected_count")
            != 1 + expected_hippo_items
            or inference.get("actual_model_future_submit_count_before_first_join")
            != 1 + expected_hippo_items
            or inference.get("all_actual_model_futures_submitted_before_first_join")
            is not True
            or inference.get("qwen_batch_item_count")
            != BLOCK_ITEM_COUNTS[self.block]
            or inference.get("qwen_batch_submitted_count") != 1
            or inference.get("qwen_batch_terminal_count") != 1
            or inference.get("qwen_executor_dedicated") is not True
            or inference.get("minilm_raw_compiled_item_count")
            != BLOCK_ITEM_COUNTS[self.block]
            or inference.get("retry_replay_resample_provider_switch") != 0
            or isinstance(inference.get("qwen_worker_pid"), bool)
            or not isinstance(inference.get("qwen_worker_pid"), int)
            or inference["qwen_worker_pid"] <= 1
            or qwen_pid != inference["qwen_worker_pid"]
            or _canonical_hash(qwen_transport)
            != inference.get("qwen_transport_receipt_sha256")
            or any(
                not isinstance(inference.get(field), str)
                or _SHA256.fullmatch(inference[field]) is None
                for field in (
                    "qwen_transport_receipt_sha256",
                    "qwen_worker_receipt_sha256",
                )
            )
            or not isinstance(hippo_transport, list)
            or len(hippo_transport_receipts) != expected_hippo_items
            or not isinstance(hippo_workers, list)
            or not isinstance(hippo_pids, list)
            or len(hippo_transport) != expected_hippo_items
            or len(hippo_workers) != expected_hippo_items
            or len(hippo_pids) != expected_hippo_items
            or len(validated_hippo) != expected_hippo_items
            or len({row[1] for row in validated_hippo})
            != expected_hippo_items
            or [row[1] for row in validated_hippo]
            != sorted(row[1] for row in validated_hippo)
            or [_canonical_hash(row[0]) for row in validated_hippo]
            != hippo_transport
            or [row[4] for row in validated_hippo] != hippo_pids
            or not isinstance(declared_witnesses, list)
            or declared_witnesses != overlap_witnesses
            or any(not isinstance(row, str) or _SHA256.fullmatch(row) is None for row in (*hippo_transport, *hippo_workers))
            or any(isinstance(row, bool) or not isinstance(row, int) or row <= 1 for row in hippo_pids)
        ):
            raise TatqaP20FormalControllerError(
                "actual inference preparation receipt drifted"
            )
        if has_hippo:
            if (
                self.hippo_executor_dedicated is not True
                or self.hippo_concurrency_cap != HIPPO_CONCURRENCY_CAP
                or not 1 <= self.hippo_observed_peak <= HIPPO_CONCURRENCY_CAP
                or inference.get("hippo_actual_concurrency_cap")
                != HIPPO_CONCURRENCY_CAP
                or inference.get("hippo_actual_observed_peak")
                != self.hippo_observed_peak
                or inference.get("hippo_future_submitted_count")
                != expected_hippo_items
                or inference.get("hippo_future_terminal_count")
                != expected_hippo_items
                or inference.get("hippo_future_consumed_count")
                != expected_hippo_items
                or inference.get("hippo_executor_dedicated") is not True
                or inference.get("qwen_hippo_independent_executors") is not True
                or inference.get("qwen_hippo_overlap_observed") is not True
                or not overlap_witnesses
            ):
                raise TatqaP20FormalControllerError("HippoRAG peak/cap receipt drifted")
        elif (
            self.hippo_executor_dedicated is not False
            or self.hippo_concurrency_cap != 0
            or self.hippo_observed_peak != 0
            or inference.get("hippo_actual_concurrency_cap") != 0
            or inference.get("hippo_actual_observed_peak") != 0
            or inference.get("hippo_future_submitted_count") != 0
            or inference.get("hippo_future_terminal_count") != 0
            or inference.get("hippo_future_consumed_count") != 0
            or inference.get("hippo_executor_dedicated") is not False
            or inference.get("qwen_hippo_independent_executors") is not False
            or inference.get("qwen_hippo_overlap_observed") is not False
            or overlap_witnesses
            or declared_witnesses != []
        ):
            raise TatqaP20FormalControllerError("non-Hippo stage reported Hippo runtime")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "candidate_work_actions", candidate_actions)
        object.__setattr__(self, "inference_preparation_receipt", inference)

    def _submitted_actions(
        self,
        actions: tuple[LogicalActionResult, ...] | None = None,
        candidate_actions: tuple[LogicalActionResult, ...] | None = None,
    ) -> tuple[LogicalActionResult, ...]:
        logical = self.actions if actions is None else actions
        candidate = (
            self.candidate_work_actions
            if candidate_actions is None
            else candidate_actions
        )
        by_key = {
            (row.item_commitment_sha256, row.logical_arm): row
            for row in (*logical, *candidate)
        }
        item_order = tuple(dict.fromkeys(row.item_commitment_sha256 for row in logical))
        return tuple(
            by_key[(item, arm)]
            for item in item_order
            for arm in BLOCK_WORK_ARMS[self.block]
        )

    def payload(self) -> dict[str, object]:
        submitted = self._submitted_actions()
        return {
            "schema": f"{VERSION}_stage_archive_v1",
            "block": self.block,
            "submitted_work_count": self.submitted_work_count,
            "submitted_work_terminal_count": self.submitted_work_terminal_count,
            "logical_action_count": self.logical_action_count,
            "logical_action_terminal_count": self.logical_action_terminal_count,
            "submit_count_before_first_join": self.submit_count_before_first_join,
            "eager_submit_complete_before_first_join": (
                self.eager_submit_complete_before_first_join
            ),
            "submitted_work_arms": list(BLOCK_WORK_ARMS[self.block]),
            "hippo_executor_dedicated": self.hippo_executor_dedicated,
            "hippo_concurrency_cap": self.hippo_concurrency_cap,
            "hippo_observed_peak": self.hippo_observed_peak,
            "predeclared_logical_arms": list(BLOCK_ARMS[self.block]),
            "logical_action_submission_accounting_only_not_model_inference": True,
            "actual_inference_preparation": dict(
                self.inference_preparation_receipt
            ),
            "logical_action_result_sha256s": [
                row.action_result_sha256 for row in self.actions
            ],
            # Persist the complete, label-free actions as well as their hashes.
            # The hashes alone cannot support an independent offline replay of
            # the frozen utility calculation after labels are released.
            "logical_action_results": [row.payload() for row in self.actions],
            "submitted_work_action_result_sha256s": [
                row.action_result_sha256 for row in submitted
            ],
            "submitted_work_action_results": [row.payload() for row in submitted],
            "shared_candidate_work_action_result_sha256s": [
                row.action_result_sha256 for row in self.candidate_work_actions
            ],
            "shared_candidate_work_action_results": [
                row.payload() for row in self.candidate_work_actions
            ],
            "logical_behavior_sha256s": [
                row.behavior_sha256 for row in self.actions
            ],
            "shared_candidate_work_only_not_effect_gate": True,
            "effect_gate_count": 1,
            "effect_gate_scope": "single_predeclared_E1_effect_rule",
            "candidate_work_additional_effect_gate_count": 0,
            "external_network_calls": 0,
            "api_or_online_evaluator_calls": 0,
            "retry_replay_resample_provider_switch": 0,
        }

    @property
    def archive_sha256(self) -> str:
        return _canonical_hash(self.payload())


@dataclass(frozen=True)
class RuntimePostflight:
    block: str
    archive_sha256: str
    runtime_ok: bool
    external_network_calls: int
    api_or_online_evaluator_calls: int
    retry_replay_resample_provider_switch: int
    controller_or_worker_source_reads: int
    controller_or_worker_label_reads: int
    maximum_cpu_threads_per_hippo_process: int

    def __post_init__(self) -> None:
        if self.block not in BLOCK_ARMS:
            raise TatqaP20FormalControllerError("postflight block drifted")
        _require_sha256(self.archive_sha256, "postflight archive")
        expected_threads = (
            HIPPO_CPU_THREADS_PER_PROCESS
            if HIPPO_ARM in BLOCK_ARMS[self.block]
            else 0
        )
        if (
            self.runtime_ok is not True
            or self.external_network_calls != 0
            or self.api_or_online_evaluator_calls != 0
            or self.retry_replay_resample_provider_switch != 0
            or self.controller_or_worker_source_reads != 0
            or self.controller_or_worker_label_reads != 0
            or self.maximum_cpu_threads_per_hippo_process != expected_threads
        ):
            raise TatqaP20FormalControllerError("runtime postflight failed")

    def payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_runtime_postflight_v1",
            "block": self.block,
            "archive_sha256": self.archive_sha256,
            "runtime_ok": self.runtime_ok,
            "external_network_calls": self.external_network_calls,
            "api_or_online_evaluator_calls": self.api_or_online_evaluator_calls,
            "retry_replay_resample_provider_switch": (
                self.retry_replay_resample_provider_switch
            ),
            "controller_or_worker_source_reads": (
                self.controller_or_worker_source_reads
            ),
            "controller_or_worker_label_reads": (
                self.controller_or_worker_label_reads
            ),
            "maximum_cpu_threads_per_hippo_process": (
                self.maximum_cpu_threads_per_hippo_process
            ),
        }

    @property
    def postflight_sha256(self) -> str:
        return _canonical_hash(self.payload())


@dataclass(frozen=True)
class PolicySelection:
    item_commitment_sha256: str
    e0_action_result_sha256: str
    e1_action_result_sha256: str
    e0_behavior_sha256: str
    e1_behavior_sha256: str
    e1_selected_policy_id: str
    same_behavior: bool

    def __post_init__(self) -> None:
        _require_sha256(self.item_commitment_sha256, "policy item")
        for value, field in (
            (self.e0_action_result_sha256, "E0 action result"),
            (self.e1_action_result_sha256, "E1 action result"),
            (self.e0_behavior_sha256, "E0 behavior"),
            (self.e1_behavior_sha256, "E1 behavior"),
        ):
            _require_sha256(value, field)
        if self.e1_selected_policy_id not in {
            evaluator.P0_POLICY_ID,
            evaluator.P1_POLICY_ID,
        }:
            raise TatqaP20FormalControllerError("frozen E1 policy drifted")
        if self.same_behavior is not (
            self.e0_behavior_sha256 == self.e1_behavior_sha256
        ):
            raise TatqaP20FormalControllerError("same-behavior flag drifted")

    def payload(self) -> dict[str, object]:
        return {
            "item_commitment_sha256": self.item_commitment_sha256,
            "E0_action_result_sha256": self.e0_action_result_sha256,
            "E1_action_result_sha256": self.e1_action_result_sha256,
            "E0_behavior_sha256": self.e0_behavior_sha256,
            "E1_behavior_sha256": self.e1_behavior_sha256,
            "E1_selected_policy_id": self.e1_selected_policy_id,
            "same_behavior": self.same_behavior,
        }


@dataclass(frozen=True)
class PolicyFreeze:
    rows: tuple[PolicySelection, ...]
    f_search_archive_sha256: str
    f_search_postflight_sha256: str
    model: evaluator.PairedDeltaRidgeModel

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        if (
            len(rows) != BLOCK_ITEM_COUNTS["F_search"]
            or any(not isinstance(row, PolicySelection) for row in rows)
            or len({row.item_commitment_sha256 for row in rows}) != len(rows)
        ):
            raise TatqaP20FormalControllerError("F_search policy freeze drifted")
        _require_sha256(self.f_search_archive_sha256, "F archive")
        _require_sha256(self.f_search_postflight_sha256, "F postflight")
        if not isinstance(self.model, evaluator.PairedDeltaRidgeModel):
            raise TatqaP20FormalControllerError("E1 model drifted")
        object.__setattr__(self, "rows", rows)

    @property
    def model_sha256(self) -> str:
        return _canonical_hash(self.model.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_F_search_policy_freeze_v1",
            "F_search_archive_sha256": self.f_search_archive_sha256,
            "F_search_postflight_sha256": self.f_search_postflight_sha256,
            "E1_model_sha256": self.model_sha256,
            "E1_model": self.model.payload(),
            "rows": [row.payload() for row in self.rows],
            "label_pack_created_or_released": False,
            "behavior_hashes_always_reported": True,
        }

    @property
    def policy_freeze_sha256(self) -> str:
        return _canonical_hash(self.payload())


@dataclass(frozen=True)
class EvaluatorFitEvidence:
    """Self-contained A_form evidence for independent offline model replay."""

    archive_sha256: str
    postflight_sha256: str
    label_pack: LabelPack
    feature_rows: tuple[tuple[int, ...], ...]
    utility_deltas: tuple[Fraction, ...]
    model: evaluator.PairedDeltaRidgeModel

    def __post_init__(self) -> None:
        _require_sha256(self.archive_sha256, "formation archive")
        _require_sha256(self.postflight_sha256, "formation postflight")
        if (
            not isinstance(self.label_pack, LabelPack)
            or self.label_pack.block != "A_form"
        ):
            raise TatqaP20FormalControllerError("formation label pack drifted")
        features = tuple(evaluator.feature_vector(row) for row in self.feature_rows)
        deltas = tuple(self.utility_deltas)
        if (
            len(features) != BLOCK_ITEM_COUNTS["A_form"]
            or len(deltas) != BLOCK_ITEM_COUNTS["A_form"]
            or any(not isinstance(row, Fraction) for row in deltas)
            or not isinstance(self.model, evaluator.PairedDeltaRidgeModel)
        ):
            raise TatqaP20FormalControllerError("formation replay rows drifted")
        replayed = evaluator.fit_paired_delta_ridge(features, deltas)
        if replayed.payload() != self.model.payload():
            raise TatqaP20FormalControllerError("formation model replay drifted")
        object.__setattr__(self, "feature_rows", features)
        object.__setattr__(self, "utility_deltas", deltas)

    @property
    def model_sha256(self) -> str:
        return _canonical_hash(self.model.payload())

    def payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_A_form_evaluator_fit_evidence_v1",
            "archive_sha256": self.archive_sha256,
            "postflight_sha256": self.postflight_sha256,
            "label_pack_sha256": self.label_pack.label_pack_sha256,
            "label_pack": self.label_pack.payload(),
            "fixed_feature_order": list(evaluator.FEATURE_ORDER),
            "feature_rows": [list(row) for row in self.feature_rows],
            "utility_deltas": [
                _fraction_payload(row) for row in self.utility_deltas
            ],
            "E1_model": self.model.payload(),
            "E1_model_sha256": self.model_sha256,
            "scoring": "local_offline_exact_only",
        }

    @property
    def fit_evidence_sha256(self) -> str:
        return _canonical_hash(self.payload())


@dataclass(frozen=True)
class PairedComparison:
    left_arm: str
    right_arm: str
    deltas: tuple[Fraction, ...]
    family_nets: tuple[tuple[str, Fraction], ...]
    exact_test: evaluator.ExactSignFlipResult

    def __post_init__(self) -> None:
        deltas = tuple(self.deltas)
        if not deltas or any(not isinstance(row, Fraction) for row in deltas):
            raise TatqaP20FormalControllerError("paired comparison deltas drifted")
        family_nets = tuple(self.family_nets)
        if tuple(row[0] for row in family_nets) != FAMILY_ORDER or any(
            not isinstance(row[1], Fraction) for row in family_nets
        ):
            raise TatqaP20FormalControllerError("paired family nets drifted")
        if self.exact_test != evaluator.exact_magnitude_preserving_sign_flip(deltas):
            raise TatqaP20FormalControllerError("paired exact test drifted")
        object.__setattr__(self, "deltas", deltas)
        object.__setattr__(self, "family_nets", family_nets)

    @property
    def net_u(self) -> Fraction:
        return sum(self.deltas, Fraction(0))

    @property
    def aggregate_and_all_families_positive(self) -> bool:
        return self.net_u > 0 and all(value > 0 for _family, value in self.family_nets)

    def payload(self) -> dict[str, object]:
        return {
            "left_arm": self.left_arm,
            "right_arm": self.right_arm,
            "net_U": _fraction_payload(self.net_u),
            "gain_count": sum(row > 0 for row in self.deltas),
            "harm_count": sum(row < 0 for row in self.deltas),
            "tie_count": sum(row == 0 for row in self.deltas),
            "family_nets": {
                family: _fraction_payload(value) for family, value in self.family_nets
            },
            "exact_test": self.exact_test.payload(),
            "paired_deltas": [_fraction_payload(row) for row in self.deltas],
            "paired_delta_sha256": _canonical_hash(
                [_fraction_payload(row) for row in self.deltas]
            ),
        }


@dataclass(frozen=True)
class ItemScoreRow:
    """One fully exact item row persisted for offline score recomputation."""

    item_commitment_sha256: str
    family: str
    canonical_gold_units: tuple[str, ...]
    arm_utilities: tuple[tuple[str, Fraction], ...]

    def __post_init__(self) -> None:
        _require_sha256(self.item_commitment_sha256, "item score commitment")
        if self.family not in FAMILY_ORDER:
            raise TatqaP20FormalControllerError("item score family drifted")
        gold = tuple(self.canonical_gold_units)
        if not 1 <= len(gold) <= evaluator.TOP_K or len(set(gold)) != len(gold):
            raise TatqaP20FormalControllerError("item score gold units drifted")
        for unit_id in gold:
            _require_unit_id(unit_id)
        utilities = tuple(self.arm_utilities)
        if (
            not utilities
            or any(
                not isinstance(arm, str)
                or not isinstance(value, Fraction)
                or value < 0
                or value > 2
                for arm, value in utilities
            )
        ):
            raise TatqaP20FormalControllerError("item score utilities drifted")
        object.__setattr__(self, "canonical_gold_units", gold)
        object.__setattr__(self, "arm_utilities", utilities)

    def payload(self) -> dict[str, object]:
        return {
            "item_commitment_sha256": self.item_commitment_sha256,
            "family": self.family,
            "canonical_gold_units": list(self.canonical_gold_units),
            "arm_utilities": {
                arm: _fraction_payload(value) for arm, value in self.arm_utilities
            },
        }


@dataclass(frozen=True)
class StageScore:
    block: str
    archive_sha256: str
    postflight_sha256: str
    label_pack: LabelPack
    arm_totals: tuple[tuple[str, Fraction], ...]
    arm_complete_counts: tuple[tuple[str, int], ...]
    candidate_expansion_item_count: int
    candidate_expansion_unit_count: int
    item_rows: tuple[ItemScoreRow, ...]
    comparisons: tuple[PairedComparison, ...]

    def __post_init__(self) -> None:
        if self.block not in {"A_hold", "M_search"}:
            raise TatqaP20FormalControllerError("scored stage drifted")
        for value, field in (
            (self.archive_sha256, "score archive"),
            (self.postflight_sha256, "score postflight"),
        ):
            _require_sha256(value, field)
        if not isinstance(self.label_pack, LabelPack) or self.label_pack.block != self.block:
            raise TatqaP20FormalControllerError("score label pack drifted")
        totals = tuple(self.arm_totals)
        if tuple(row[0] for row in totals) != BLOCK_ARMS[self.block] or any(
            not isinstance(row[1], Fraction) for row in totals
        ):
            raise TatqaP20FormalControllerError("arm utility totals drifted")
        complete_counts = tuple(self.arm_complete_counts)
        if tuple(row[0] for row in complete_counts) != BLOCK_ARMS[self.block] or any(
            type(row[1]) is not int
            or not 0 <= row[1] <= BLOCK_ITEM_COUNTS[self.block]
            for row in complete_counts
        ):
            raise TatqaP20FormalControllerError("arm complete counts drifted")
        item_rows = tuple(self.item_rows)
        if (
            len(item_rows) != BLOCK_ITEM_COUNTS[self.block]
            or any(not isinstance(row, ItemScoreRow) for row in item_rows)
            or len({row.item_commitment_sha256 for row in item_rows})
            != len(item_rows)
            or any(
                tuple(arm for arm, _value in row.arm_utilities)
                != BLOCK_ARMS[self.block]
                for row in item_rows
            )
        ):
            raise TatqaP20FormalControllerError("item score row registry drifted")
        labels_by_item = {
            row.item_commitment_sha256: row for row in self.label_pack.rows
        }
        if any(
            row.item_commitment_sha256 not in labels_by_item
            or labels_by_item[row.item_commitment_sha256].family != row.family
            or labels_by_item[row.item_commitment_sha256].canonical_gold_units
            != row.canonical_gold_units
            for row in item_rows
        ):
            raise TatqaP20FormalControllerError("item score/label binding drifted")
        recomputed_totals = tuple(
            (
                arm,
                sum(
                    (
                        dict(row.arm_utilities)[arm]
                        for row in item_rows
                    ),
                    Fraction(0),
                ),
            )
            for arm in BLOCK_ARMS[self.block]
        )
        recomputed_complete = tuple(
            (
                arm,
                sum(dict(row.arm_utilities)[arm] == 2 for row in item_rows),
            )
            for arm in BLOCK_ARMS[self.block]
        )
        if recomputed_totals != totals or recomputed_complete != complete_counts:
            raise TatqaP20FormalControllerError("item score aggregate drifted")
        if (
            type(self.candidate_expansion_item_count) is not int
            or type(self.candidate_expansion_unit_count) is not int
            or not 0
            <= self.candidate_expansion_item_count
            <= BLOCK_ITEM_COUNTS[self.block]
            or not 0
            <= self.candidate_expansion_unit_count
            <= 2 * BLOCK_ITEM_COUNTS[self.block]
        ):
            raise TatqaP20FormalControllerError(
                "candidate-expansion count drifted"
            )
        comparisons = tuple(self.comparisons)
        expected_pairs = (("E1", "E0"), ("E1", "RAW"), ("E1", HIPPO_ARM))
        if tuple((row.left_arm, row.right_arm) for row in comparisons) != expected_pairs:
            raise TatqaP20FormalControllerError("paired comparison registry drifted")
        for comparison in comparisons:
            expected_deltas = tuple(
                dict(row.arm_utilities)[comparison.left_arm]
                - dict(row.arm_utilities)[comparison.right_arm]
                for row in item_rows
            )
            expected_family_nets = tuple(
                (
                    family,
                    sum(
                        (
                            delta
                            for row, delta in zip(
                                item_rows, expected_deltas, strict=True
                            )
                            if row.family == family
                        ),
                        Fraction(0),
                    ),
                )
                for family in FAMILY_ORDER
            )
            if (
                comparison.deltas != expected_deltas
                or comparison.family_nets != expected_family_nets
            ):
                raise TatqaP20FormalControllerError(
                    "paired comparison/item score binding drifted"
                )
        object.__setattr__(self, "arm_totals", totals)
        object.__setattr__(self, "arm_complete_counts", complete_counts)
        object.__setattr__(self, "item_rows", item_rows)
        object.__setattr__(self, "comparisons", comparisons)

    @property
    def label_pack_sha256(self) -> str:
        return self.label_pack.label_pack_sha256

    def comparison(self, right_arm: str) -> PairedComparison:
        return next(row for row in self.comparisons if row.right_arm == right_arm)

    def payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_{self.block}_offline_score_v1",
            "block": self.block,
            "archive_sha256": self.archive_sha256,
            "postflight_sha256": self.postflight_sha256,
            "label_pack_sha256": self.label_pack_sha256,
            "label_pack": self.label_pack.payload(),
            "arm_totals": {
                arm: _fraction_payload(value) for arm, value in self.arm_totals
            },
            "arm_complete_counts": dict(self.arm_complete_counts),
            "candidate_expansion_outside_P0_top5": {
                "item_count": self.candidate_expansion_item_count,
                "unit_count": self.candidate_expansion_unit_count,
            },
            "item_exact_utility_rows": [row.payload() for row in self.item_rows],
            "comparisons": [row.payload() for row in self.comparisons],
            "scoring": "local_offline_exact_only",
        }

    @property
    def score_sha256(self) -> str:
        return _canonical_hash(self.payload())


@dataclass(frozen=True)
class EpochAuthorization:
    a_hold_score_sha256: str
    policy_freeze_sha256: str
    previous_evaluator: str = "E0"
    active_evaluator: str = "E1"
    m_search_authorized: bool = True
    transition_index: int = 1

    def __post_init__(self) -> None:
        _require_sha256(self.a_hold_score_sha256, "A_hold score")
        _require_sha256(self.policy_freeze_sha256, "policy freeze")
        if (
            self.previous_evaluator != "E0"
            or self.active_evaluator != "E1"
            or self.m_search_authorized is not True
            or self.transition_index != 1
        ):
            raise TatqaP20FormalControllerError("epoch authorization drifted")

    def payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_epoch_transition_authorization_v1",
            "A_hold_score_sha256": self.a_hold_score_sha256,
            "policy_freeze_sha256": self.policy_freeze_sha256,
            "previous_evaluator": self.previous_evaluator,
            "active_evaluator": self.active_evaluator,
            "E0_counterfactual_retained": True,
            "M_search_authorized": self.m_search_authorized,
            "transition_index": self.transition_index,
            "rollback_authorized": False,
        }

    @property
    def authorization_sha256(self) -> str:
        return _canonical_hash(self.payload())


@dataclass(frozen=True)
class FormalDisposition:
    status: str
    primary_evaluated: bool
    primary_value: bool | None
    efficacy: str
    a_hold_promoted: bool
    epoch_transition_count: int
    m_view_released: bool
    m_labels_released: bool
    replay_authorized: bool
    failure_stage: str | None = None
    failure_type_sha256: str | None = None
    acquisition_receipt_sha256: str | None = None
    preflight: RuntimePreflight | None = None
    model_sha256: str | None = None
    a_form_fit: EvaluatorFitEvidence | None = None
    a_form_archive: StageArchive | None = None
    f_search_archive: StageArchive | None = None
    policy_freeze: PolicyFreeze | None = None
    a_hold_archive: StageArchive | None = None
    a_hold_score: StageScore | None = None
    epoch_authorization: EpochAuthorization | None = None
    m_search_archive: StageArchive | None = None
    m_search_score: StageScore | None = None

    def __post_init__(self) -> None:
        if self.replay_authorized is not False:
            raise TatqaP20FormalControllerError("replay must remain forbidden")
        if self.status == "implementation_or_runtime_invalid":
            if (
                self.primary_evaluated is not False
                or self.primary_value is not None
                or self.efficacy != "unknown"
                or self.failure_stage is None
                or self.failure_type_sha256 is None
            ):
                raise TatqaP20FormalControllerError("invalid disposition semantics drifted")
            _require_sha256(self.failure_type_sha256, "failure type")
        elif self.status == "valid_nonpromotion":
            if (
                self.primary_evaluated is not True
                or self.primary_value is not False
                or self.efficacy != "false"
                or self.a_hold_promoted is not False
                or self.epoch_transition_count != 0
                or self.m_view_released
                or self.m_labels_released
                or self.a_hold_score is None
            ):
                raise TatqaP20FormalControllerError("nonpromotion disposition drifted")
        elif self.status in {"valid_primary_true", "valid_primary_false"}:
            expected = self.status == "valid_primary_true"
            if (
                self.primary_evaluated is not True
                or self.primary_value is not expected
                or self.efficacy != ("true" if expected else "false")
                or self.a_hold_promoted is not True
                or self.epoch_transition_count != 1
                or self.m_view_released is not True
                or self.m_labels_released is not True
                or self.epoch_authorization is None
                or self.m_search_score is None
            ):
                raise TatqaP20FormalControllerError("completed disposition drifted")
        else:
            raise TatqaP20FormalControllerError("unknown terminal disposition")
        if self.epoch_transition_count not in (0, 1):
            raise TatqaP20FormalControllerError("epoch transition count drifted")
        if self.model_sha256 is None:
            if self.a_form_fit is not None:
                raise TatqaP20FormalControllerError(
                    "formation evidence exists without a model"
                )
        elif (
            not isinstance(self.a_form_fit, EvaluatorFitEvidence)
            or self.a_form_fit.model_sha256 != self.model_sha256
            or self.a_form_archive is None
            or self.a_form_fit.archive_sha256 != self.a_form_archive.archive_sha256
        ):
            raise TatqaP20FormalControllerError(
                "formation evidence/model binding drifted"
            )

    def payload(self) -> dict[str, object]:
        return {
            "schema": f"{VERSION}_terminal_disposition_v1",
            "status": self.status,
            "primary_count": 1,
            "primary_operator": "AND",
            "primary_evaluated": self.primary_evaluated,
            "primary_value": self.primary_value,
            "efficacy": self.efficacy,
            "A_hold_promoted": self.a_hold_promoted,
            "epoch_transition_count": self.epoch_transition_count,
            "M_search_view_released": self.m_view_released,
            "M_search_labels_released": self.m_labels_released,
            "failure_stage": self.failure_stage,
            "failure_type_sha256": self.failure_type_sha256,
            "acquisition_receipt_sha256": self.acquisition_receipt_sha256,
            "runtime_preflight_sha256": (
                None if self.preflight is None else self.preflight.receipt_sha256
            ),
            "E1_model_sha256": self.model_sha256,
            "E1_model": (
                None if self.a_form_fit is None else self.a_form_fit.model.payload()
            ),
            "A_form_fit_evidence_sha256": (
                None
                if self.a_form_fit is None
                else self.a_form_fit.fit_evidence_sha256
            ),
            "A_form_archive_sha256": (
                None if self.a_form_archive is None else self.a_form_archive.archive_sha256
            ),
            "F_search_archive_sha256": (
                None if self.f_search_archive is None else self.f_search_archive.archive_sha256
            ),
            "policy_freeze_sha256": (
                None if self.policy_freeze is None else self.policy_freeze.policy_freeze_sha256
            ),
            "A_hold_archive_sha256": (
                None if self.a_hold_archive is None else self.a_hold_archive.archive_sha256
            ),
            "A_hold_score_sha256": (
                None if self.a_hold_score is None else self.a_hold_score.score_sha256
            ),
            "epoch_authorization_sha256": (
                None
                if self.epoch_authorization is None
                else self.epoch_authorization.authorization_sha256
            ),
            "M_search_archive_sha256": (
                None if self.m_search_archive is None else self.m_search_archive.archive_sha256
            ),
            "M_search_score_sha256": (
                None if self.m_search_score is None else self.m_search_score.score_sha256
            ),
            "external_network_calls": 0,
            "api_or_online_evaluator_calls": 0,
            "retry_replay_resample_provider_switch": 0,
            "same_source_or_cohort_replay_authorized": self.replay_authorized,
        }

    @property
    def disposition_sha256(self) -> str:
        return _canonical_hash(self.payload())


class AcquisitionBoundary(Protocol):
    """Trusted capability; implementations alone may access sealed custody."""

    def claim_one_shot(self) -> str:
        """Consume the externally durable one-shot acquisition marker."""

    def load_block_view(
        self, block: str, authorization: EpochAuthorization | None
    ) -> BlockView:
        """Release one already-sealed label-free view."""

    def release_label_pack(
        self,
        block: str,
        archive_sha256: str,
        postflight_sha256: str,
    ) -> LabelPack:
        """Release labels only after both supplied receipts validate."""


class RuntimeBoundary(Protocol):
    """Injected offline runtime and executor authority."""

    def preflight(self) -> RuntimePreflight:
        ...

    def action_executor(
        self, block: str, standard_work_count: int
    ) -> AbstractContextManager[concurrent.futures.Executor]:
        ...

    def hippo_executor(
        self, block: str, concurrency_cap: int
    ) -> AbstractContextManager[concurrent.futures.Executor]:
        ...

    def run_raw(self, block: str, item: ItemView) -> Sequence[str]:
        ...

    def run_hippo(self, block: str, item: ItemView) -> Sequence[str]:
        ...

    def preparation_inference_receipt(
        self, block: str
    ) -> Mapping[str, object]:
        ...

    def postflight(self, block: str, archive: StageArchive) -> RuntimePostflight:
        ...

    def abort_all_inference(self) -> None:
        """Synchronously cancel and join every prepared model cohort."""

    def verify_all_inference_closed(self) -> None:
        """Fail unless every prepared model cohort has been joined."""

    def persist_offline_evidence(
        self,
        name: str,
        payload: Mapping[str, object],
        evidence_sha256: str,
    ) -> str:
        """Exclusively persist and reopen one exact post-label artifact."""

    def verify_offline_evidence(
        self, name: str, evidence_sha256: str
    ) -> None:
        """Reopen a previously sealed artifact before a downstream release."""


class _PeakCounter:
    def __init__(self, cap: int) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._live = 0
        self._peak = 0

    def enter(self) -> None:
        with self._lock:
            self._live += 1
            self._peak = max(self._peak, self._live)
            if self._live > self._cap:
                self._live -= 1
                raise TatqaP20FormalControllerError("HippoRAG concurrency cap exceeded")

    def leave(self) -> None:
        with self._lock:
            if self._live <= 0:
                raise TatqaP20FormalControllerError("HippoRAG concurrency counter underflow")
            self._live -= 1

    def finish(self) -> int:
        with self._lock:
            if self._live != 0 or not 1 <= self._peak <= self._cap:
                raise TatqaP20FormalControllerError("HippoRAG peak receipt is invalid")
            return self._peak


class TatqaP20FormalController:
    """Execute exactly one injected P20 lifecycle with no retry path."""

    def __init__(
        self,
        *,
        acquisition: AcquisitionBoundary,
        runtime: RuntimeBoundary,
    ) -> None:
        self._acquisition = acquisition
        self._runtime = runtime
        self._entry_lock = threading.Lock()
        self._entered = False
        self._failure_stage = "not_started"
        self._seen_items: set[str] = set()
        self._preflight: RuntimePreflight | None = None
        self._acquisition_receipt: str | None = None
        self._model_sha256: str | None = None
        self._a_form_fit: EvaluatorFitEvidence | None = None
        self._a_form_archive: StageArchive | None = None
        self._f_search_archive: StageArchive | None = None
        self._policy_freeze: PolicyFreeze | None = None
        self._a_hold_archive: StageArchive | None = None
        self._a_hold_score: StageScore | None = None
        self._epoch_authorization: EpochAuthorization | None = None
        self._m_search_archive: StageArchive | None = None
        self._m_search_score: StageScore | None = None
        self._epoch_transition_count = 0
        self._m_view_released = False
        self._m_labels_released = False

    def run(self) -> FormalDisposition:
        with self._entry_lock:
            if self._entered:
                exc = TatqaP20FormalControllerError("one-shot controller reentry")
                return self._invalid_disposition(
                    exc, stage="one_shot_reentry", include_prior_state=False
                )
            self._entered = True
        try:
            disposition = self._run_once()
            self._failure_stage = "terminal_inference_closed_verification"
            self._runtime.verify_all_inference_closed()
            return disposition
        except Exception as exc:
            original_stage = self._failure_stage
            abort_exc: Exception | None = None
            try:
                self._runtime.abort_all_inference()
            except Exception as cleanup_exc:
                abort_exc = cleanup_exc
            try:
                # Always request an independent closure proof, even when the
                # abort path itself reported an error.  Returning a terminal
                # disposition without this proof is forbidden.
                self._runtime.verify_all_inference_closed()
            except Exception as verify_exc:
                raise TatqaP20FormalControllerError(
                    "inference closure could not be proved"
                ) from verify_exc
            if abort_exc is not None:
                return self._invalid_disposition(
                    abort_exc,
                    stage="terminal_inference_abort",
                    include_prior_state=True,
                )
            return self._invalid_disposition(
                exc, stage=original_stage, include_prior_state=True
            )

    def _run_once(self) -> FormalDisposition:
        self._failure_stage = "runtime_preflight"
        preflight = self._runtime.preflight()
        if not isinstance(preflight, RuntimePreflight):
            raise TatqaP20FormalControllerError("runtime preflight type drifted")
        self._preflight = preflight

        self._failure_stage = "one_shot_acquisition_claim"
        self._acquisition_receipt = _require_sha256(
            self._acquisition.claim_one_shot(), "acquisition receipt"
        )

        self._failure_stage = "A_form_view"
        a_form_view = self._claim_view("A_form", None)
        self._failure_stage = "A_form_eager_actions"
        a_archive, a_postflight = self._execute_stage(a_form_view, model=None)
        self._a_form_archive = a_archive
        self._failure_stage = "A_form_late_labels"
        a_labels = self._release_labels(a_form_view, a_archive, a_postflight)
        self._failure_stage = "A_form_E1_fit"
        model, self._a_form_fit = self._fit_e1(
            a_form_view, a_archive, a_postflight, a_labels
        )
        self._model_sha256 = _canonical_hash(model.payload())
        self._failure_stage = "A_form_fit_durable_seal"
        self._seal_evidence(
            "A_form_fit",
            self._a_form_fit.payload(),
            self._a_form_fit.fit_evidence_sha256,
        )

        self._failure_stage = "F_search_view"
        f_view = self._claim_view("F_search", None)
        self._failure_stage = "F_search_eager_actions"
        f_archive, f_postflight = self._execute_stage(f_view, model=None)
        self._f_search_archive = f_archive
        self._failure_stage = "F_search_policy_freeze"
        self._policy_freeze = self._freeze_f_policies(
            f_view, f_archive, f_postflight, model
        )
        self._failure_stage = "F_search_policy_durable_seal"
        self._seal_evidence(
            "F_search_policy_freeze",
            self._policy_freeze.payload(),
            self._policy_freeze.policy_freeze_sha256,
        )

        self._failure_stage = "A_hold_view"
        a_hold_view = self._claim_view("A_hold", None)
        self._failure_stage = "A_hold_eager_actions"
        a_hold_archive, a_hold_postflight = self._execute_stage(
            a_hold_view, model=model
        )
        self._a_hold_archive = a_hold_archive
        self._failure_stage = "A_hold_late_labels"
        a_hold_labels = self._release_labels(
            a_hold_view, a_hold_archive, a_hold_postflight
        )
        self._failure_stage = "A_hold_exact_promotion"
        self._a_hold_score = self._score_stage(
            a_hold_view,
            a_hold_archive,
            a_hold_postflight,
            a_hold_labels,
        )
        self._failure_stage = "A_hold_score_durable_seal"
        self._seal_evidence(
            "A_hold_score",
            self._a_hold_score.payload(),
            self._a_hold_score.score_sha256,
        )
        promoted = self._a_hold_score.comparison("E0").exact_test.promoted
        if not promoted:
            return self._valid_disposition(primary=False, promoted=False)

        self._failure_stage = "epoch_transition_authorization"
        authorization = self._authorize_epoch()
        self._failure_stage = "M_search_view"
        m_view = self._claim_view("M_search", authorization)
        self._m_view_released = True
        self._failure_stage = "M_search_eager_actions"
        m_archive, m_postflight = self._execute_stage(m_view, model=model)
        self._m_search_archive = m_archive
        self._failure_stage = "M_search_late_labels"
        m_labels = self._release_labels(m_view, m_archive, m_postflight)
        self._m_labels_released = True
        self._failure_stage = "M_search_unique_AND_primary"
        self._m_search_score = self._score_stage(
            m_view, m_archive, m_postflight, m_labels
        )
        self._failure_stage = "M_search_score_durable_seal"
        self._seal_evidence(
            "M_search_score",
            self._m_search_score.payload(),
            self._m_search_score.score_sha256,
        )
        e0 = self._m_search_score.comparison("E0")
        raw = self._m_search_score.comparison("RAW")
        hippo = self._m_search_score.comparison(HIPPO_ARM)
        primary = all(
            (
                promoted,
                e0.exact_test.promoted,
                raw.aggregate_and_all_families_positive,
                hippo.aggregate_and_all_families_positive,
            )
        )
        return self._valid_disposition(primary=primary, promoted=True)

    def _seal_evidence(
        self,
        name: str,
        payload: Mapping[str, object],
        evidence_sha256: str,
    ) -> None:
        expected = _require_sha256(evidence_sha256, "offline evidence")
        observed = self._runtime.persist_offline_evidence(
            name, payload, expected
        )
        if _require_sha256(observed, "durable offline evidence") != expected:
            raise TatqaP20FormalControllerError(
                "durable offline evidence receipt drifted"
            )
        self._runtime.verify_offline_evidence(name, expected)

    def _claim_view(
        self, block: str, authorization: EpochAuthorization | None
    ) -> BlockView:
        if block == "M_search":
            if (
                authorization is None
                or authorization != self._epoch_authorization
                or self._epoch_transition_count != 1
            ):
                raise TatqaP20FormalControllerError("M_search lacks epoch authorization")
        elif authorization is not None:
            raise TatqaP20FormalControllerError("premature epoch authorization")
        view = self._acquisition.load_block_view(block, authorization)
        if not isinstance(view, BlockView) or view.block != block:
            raise TatqaP20FormalControllerError("acquisition block view drifted")
        commitments = {row.item_commitment_sha256 for row in view.items}
        if self._seen_items.intersection(commitments):
            raise TatqaP20FormalControllerError("context/item reused across blocks")
        self._seen_items.update(commitments)
        return view

    def _execute_stage(
        self,
        view: BlockView,
        *,
        model: evaluator.PairedDeltaRidgeModel | None,
    ) -> tuple[StageArchive, RuntimePostflight]:
        block = view.block
        work_arms = BLOCK_WORK_ARMS[block]
        hippo_enabled = HIPPO_ARM in work_arms
        standard_arms = tuple(arm for arm in work_arms if arm != HIPPO_ARM)
        standard_count = len(view.items) * len(standard_arms)
        submitted_count = len(view.items) * len(work_arms)
        action_context = self._runtime.action_executor(block, standard_count)

        if not hippo_enabled:
            with action_context as action_executor:
                futures = [
                    action_executor.submit(
                        self._run_standard_action, block, item, arm, model
                    )
                    for item in view.items
                    for arm in standard_arms
                ]
                submit_count = len(futures)
                terminals = tuple(future.result() for future in futures)
            inference_receipt = dict(
                self._runtime.preparation_inference_receipt(block)
            )
            archive = StageArchive(
                block=block,
                actions=terminals,
                candidate_work_actions=(),
                inference_preparation_receipt=inference_receipt,
                submitted_work_count=submitted_count,
                submitted_work_terminal_count=len(terminals),
                logical_action_count=len(view.items) * len(BLOCK_ARMS[block]),
                logical_action_terminal_count=len(terminals),
                submit_count_before_first_join=submit_count,
                eager_submit_complete_before_first_join=(
                    submit_count == submitted_count
                ),
                hippo_executor_dedicated=False,
                hippo_concurrency_cap=0,
                hippo_observed_peak=0,
            )
        else:
            hippo_context = self._runtime.hippo_executor(
                block, HIPPO_CONCURRENCY_CAP
            )
            counter = _PeakCounter(HIPPO_CONCURRENCY_CAP)
            semaphore = threading.BoundedSemaphore(HIPPO_CONCURRENCY_CAP)
            with action_context as action_executor, hippo_context as hippo_executor:
                if action_executor is hippo_executor:
                    raise TatqaP20FormalControllerError(
                        "HippoRAG executor is not dedicated"
                    )
                ordered_futures: list[
                    concurrent.futures.Future[LogicalActionResult]
                ] = []
                for item in view.items:
                    for arm in standard_arms:
                        ordered_futures.append(
                            action_executor.submit(
                                self._run_standard_action,
                                block,
                                item,
                                arm,
                                model,
                            )
                        )
                for item in view.items:
                    ordered_futures.append(
                        hippo_executor.submit(
                            self._run_hippo_action,
                            block,
                            item,
                            semaphore,
                            counter,
                        )
                    )
                submit_count = len(ordered_futures)
                terminals = tuple(future.result() for future in ordered_futures)
            wrapper_peak = counter.finish()
            if not 1 <= wrapper_peak <= HIPPO_CONCURRENCY_CAP:
                raise TatqaP20FormalControllerError(
                    "Hippo logical-wrapper concurrency accounting drifted"
                )
            inference_receipt = dict(
                self._runtime.preparation_inference_receipt(block)
            )
            actual_peak = inference_receipt.get("hippo_actual_observed_peak")
            if isinstance(actual_peak, bool) or not isinstance(actual_peak, int):
                raise TatqaP20FormalControllerError(
                    "Hippo actual inference peak is absent"
                )
            # Canonicalize terminals by item-view order and frozen arm order;
            # completion timing never affects an archive hash.
            terminal_by_key = {
                (row.item_commitment_sha256, row.logical_arm): row
                for row in terminals
            }
            terminals = tuple(
                terminal_by_key[(item.item_commitment_sha256, arm)]
                for item in view.items
                for arm in work_arms
            )
            evaluator_actions = self._derive_evaluator_actions(
                view, terminals, model
            )
            work_by_key = {
                (row.item_commitment_sha256, row.logical_arm): row
                for row in terminals
            }
            evaluator_by_key = {
                (row.item_commitment_sha256, row.logical_arm): row
                for row in evaluator_actions
            }
            logical_actions = tuple(
                (
                    evaluator_by_key[(item.item_commitment_sha256, arm)]
                    if arm in {"E0", "E1"}
                    else work_by_key[(item.item_commitment_sha256, arm)]
                )
                for item in view.items
                for arm in BLOCK_ARMS[block]
            )
            candidate_work_actions = tuple(
                work_by_key[(item.item_commitment_sha256, arm)]
                for item in view.items
                for arm in ("P0", "P1")
            )
            archive = StageArchive(
                block=block,
                actions=logical_actions,
                candidate_work_actions=candidate_work_actions,
                inference_preparation_receipt=inference_receipt,
                submitted_work_count=submitted_count,
                submitted_work_terminal_count=len(terminals),
                logical_action_count=len(view.items) * len(BLOCK_ARMS[block]),
                logical_action_terminal_count=len(logical_actions),
                submit_count_before_first_join=submit_count,
                eager_submit_complete_before_first_join=(
                    submit_count == submitted_count
                ),
                hippo_executor_dedicated=True,
                hippo_concurrency_cap=HIPPO_CONCURRENCY_CAP,
                hippo_observed_peak=actual_peak,
            )

        postflight = self._runtime.postflight(block, archive)
        if (
            not isinstance(postflight, RuntimePostflight)
            or postflight.block != block
            or postflight.archive_sha256 != archive.archive_sha256
        ):
            raise TatqaP20FormalControllerError("runtime postflight binding drifted")
        return archive, postflight

    def _typed_pair(
        self, item: ItemView
    ) -> tuple[evaluator.Action, evaluator.Action]:
        return evaluator.build_action_pair(
            item.plan,
            item.units,
            redundancy_features=item.redundancy_mapping(),
        )

    def _run_standard_action(
        self,
        block: str,
        item: ItemView,
        arm: str,
        model: evaluator.PairedDeltaRidgeModel | None,
    ) -> LogicalActionResult:
        if arm == "RAW":
            return self._baseline_result(
                block, item, arm, self._runtime.run_raw(block, item)
            )
        if arm not in {"P0", "P1"}:
            raise TatqaP20FormalControllerError("unknown standard logical arm")
        p0, p1 = self._typed_pair(item)
        selected = p0 if arm == "P0" else p1
        return LogicalActionResult(
            block=block,
            item_commitment_sha256=item.item_commitment_sha256,
            logical_arm=arm,
            selected_policy_id=selected.policy_id,
            selected_unit_ids=selected.selected_unit_ids,
            p1_minus_p0_features=p1.feature_vector,
            source_action_sha256=selected.action_sha256,
        )

    def _derive_evaluator_actions(
        self,
        view: BlockView,
        work_actions: Sequence[LogicalActionResult],
        model: evaluator.PairedDeltaRidgeModel | None,
    ) -> tuple[LogicalActionResult, ...]:
        if view.block not in {"A_hold", "M_search"}:
            if model is not None:
                raise TatqaP20FormalControllerError(
                    "formation stage unexpectedly received an evaluator model"
                )
            return ()
        if not isinstance(model, evaluator.PairedDeltaRidgeModel):
            raise TatqaP20FormalControllerError(
                "A_hold/M_search lacks the frozen E1 model"
            )
        by_key = {
            (row.item_commitment_sha256, row.logical_arm): row
            for row in work_actions
        }
        derived: list[LogicalActionResult] = []
        for item in view.items:
            item_id = item.item_commitment_sha256
            p0 = by_key[(item_id, "P0")]
            p1 = by_key[(item_id, "P1")]
            e1_source = (
                p1 if model.predict(p1.p1_minus_p0_features) > 0.0 else p0
            )
            for arm, source in (("E0", p0), ("E1", e1_source)):
                derived.append(
                    LogicalActionResult(
                        block=view.block,
                        item_commitment_sha256=item_id,
                        logical_arm=arm,
                        selected_policy_id=source.selected_policy_id,
                        selected_unit_ids=source.selected_unit_ids,
                        p1_minus_p0_features=p1.p1_minus_p0_features,
                        source_action_sha256=source.source_action_sha256,
                    )
                )
        return tuple(derived)

    def _run_hippo_action(
        self,
        block: str,
        item: ItemView,
        semaphore: threading.BoundedSemaphore,
        counter: _PeakCounter,
    ) -> LogicalActionResult:
        with semaphore:
            counter.enter()
            try:
                selected = self._runtime.run_hippo(block, item)
            finally:
                counter.leave()
        return self._baseline_result(block, item, HIPPO_ARM, selected)

    def _baseline_result(
        self,
        block: str,
        item: ItemView,
        arm: str,
        selected: Sequence[str],
    ) -> LogicalActionResult:
        top5 = tuple(selected)
        behavior_sha256 = evaluator.canonical_behavior_hash(top5)
        canonical_ids = {row.unit_id for row in item.units}
        if not set(top5).issubset(canonical_ids):
            raise TatqaP20FormalControllerError(
                "injected baseline selected a unit outside the common corpus"
            )
        source_sha256 = _canonical_hash(
            {
                "schema": f"{VERSION}_injected_baseline_action_v1",
                "block": block,
                "item_commitment_sha256": item.item_commitment_sha256,
                "logical_arm": arm,
                "ordered_top5": list(top5),
                "behavior_sha256": behavior_sha256,
            }
        )
        return LogicalActionResult(
            block=block,
            item_commitment_sha256=item.item_commitment_sha256,
            logical_arm=arm,
            selected_policy_id=arm,
            selected_unit_ids=top5,
            p1_minus_p0_features=(0,) * len(evaluator.FEATURE_ORDER),
            source_action_sha256=source_sha256,
        )

    def _release_labels(
        self,
        view: BlockView,
        archive: StageArchive,
        postflight: RuntimePostflight,
    ) -> LabelPack:
        pack = self._acquisition.release_label_pack(
            view.block, archive.archive_sha256, postflight.postflight_sha256
        )
        if not isinstance(pack, LabelPack) or pack.block != view.block:
            raise TatqaP20FormalControllerError("trusted label-pack type drifted")
        view_items = {row.item_commitment_sha256 for row in view.items}
        label_items = {row.item_commitment_sha256 for row in pack.rows}
        if view_items != label_items:
            raise TatqaP20FormalControllerError("label pack does not bind the stage view")
        corpus_by_item = {
            row.item_commitment_sha256: {unit.unit_id for unit in row.units}
            for row in view.items
        }
        if any(
            not set(row.canonical_gold_units).issubset(
                corpus_by_item[row.item_commitment_sha256]
            )
            for row in pack.rows
        ):
            raise TatqaP20FormalControllerError(
                "label pack contains a unit outside the bound item corpus"
            )
        return pack

    @staticmethod
    def _action_map(
        archive: StageArchive,
    ) -> dict[tuple[str, str], LogicalActionResult]:
        return {
            (row.item_commitment_sha256, row.logical_arm): row
            for row in (*archive.actions, *archive.candidate_work_actions)
        }

    def _fit_e1(
        self,
        view: BlockView,
        archive: StageArchive,
        postflight: RuntimePostflight,
        labels: LabelPack,
    ) -> tuple[evaluator.PairedDeltaRidgeModel, EvaluatorFitEvidence]:
        actions = self._action_map(archive)
        label_by_item = {row.item_commitment_sha256: row for row in labels.rows}
        features: list[tuple[int, ...]] = []
        utility_deltas: list[Fraction] = []
        for item in view.items:
            item_id = item.item_commitment_sha256
            p0 = actions[(item_id, "P0")]
            p1 = actions[(item_id, "P1")]
            label = label_by_item[item_id]
            features.append(p1.p1_minus_p0_features)
            utility_deltas.append(
                evaluator.item_utility(
                    p1.selected_unit_ids, label.canonical_gold_units
                )
                - evaluator.item_utility(
                    p0.selected_unit_ids, label.canonical_gold_units
                )
            )
        model = evaluator.fit_paired_delta_ridge(features, utility_deltas)
        evidence = EvaluatorFitEvidence(
            archive_sha256=archive.archive_sha256,
            postflight_sha256=postflight.postflight_sha256,
            label_pack=labels,
            feature_rows=tuple(features),
            utility_deltas=tuple(utility_deltas),
            model=model,
        )
        return model, evidence

    def _freeze_f_policies(
        self,
        view: BlockView,
        archive: StageArchive,
        postflight: RuntimePostflight,
        model: evaluator.PairedDeltaRidgeModel,
    ) -> PolicyFreeze:
        actions = self._action_map(archive)
        rows: list[PolicySelection] = []
        for item in view.items:
            item_id = item.item_commitment_sha256
            p0 = actions[(item_id, "P0")]
            p1 = actions[(item_id, "P1")]
            e1 = p1 if model.predict(p1.p1_minus_p0_features) > 0.0 else p0
            rows.append(
                PolicySelection(
                    item_commitment_sha256=item_id,
                    e0_action_result_sha256=p0.action_result_sha256,
                    e1_action_result_sha256=e1.action_result_sha256,
                    e0_behavior_sha256=p0.behavior_sha256,
                    e1_behavior_sha256=e1.behavior_sha256,
                    e1_selected_policy_id=e1.selected_policy_id,
                    same_behavior=(p0.behavior_sha256 == e1.behavior_sha256),
                )
            )
        return PolicyFreeze(
            rows=tuple(rows),
            f_search_archive_sha256=archive.archive_sha256,
            f_search_postflight_sha256=postflight.postflight_sha256,
            model=model,
        )

    def _score_stage(
        self,
        view: BlockView,
        archive: StageArchive,
        postflight: RuntimePostflight,
        labels: LabelPack,
    ) -> StageScore:
        actions = self._action_map(archive)
        label_by_item = {row.item_commitment_sha256: row for row in labels.rows}
        utilities: dict[tuple[str, str], Fraction] = {}
        for item in view.items:
            item_id = item.item_commitment_sha256
            label = label_by_item[item_id]
            for arm in BLOCK_ARMS[view.block]:
                utilities[(item_id, arm)] = evaluator.item_utility(
                    actions[(item_id, arm)].selected_unit_ids,
                    label.canonical_gold_units,
                )

        arm_totals = tuple(
            (
                arm,
                sum(
                    (utilities[(item.item_commitment_sha256, arm)] for item in view.items),
                    Fraction(0),
                ),
            )
            for arm in BLOCK_ARMS[view.block]
        )
        arm_complete_counts = tuple(
            (
                arm,
                sum(
                    utilities[(item.item_commitment_sha256, arm)] == 2
                    for item in view.items
                ),
            )
            for arm in BLOCK_ARMS[view.block]
        )
        candidate_by_key = {
            (row.item_commitment_sha256, row.logical_arm): row
            for row in archive.candidate_work_actions
        }
        candidate_expansion_counts = tuple(
            candidate_by_key[(item.item_commitment_sha256, "P1")]
            .p1_minus_p0_features[-1]
            for item in view.items
        )
        comparisons = tuple(
            self._paired_comparison(
                left_arm="E1",
                right_arm=right_arm,
                view=view,
                labels=label_by_item,
                utilities=utilities,
            )
            for right_arm in ("E0", "RAW", HIPPO_ARM)
        )
        item_rows = tuple(
            ItemScoreRow(
                item_commitment_sha256=item.item_commitment_sha256,
                family=label_by_item[item.item_commitment_sha256].family,
                canonical_gold_units=label_by_item[
                    item.item_commitment_sha256
                ].canonical_gold_units,
                arm_utilities=tuple(
                    (
                        arm,
                        utilities[(item.item_commitment_sha256, arm)],
                    )
                    for arm in BLOCK_ARMS[view.block]
                ),
            )
            for item in view.items
        )
        return StageScore(
            block=view.block,
            archive_sha256=archive.archive_sha256,
            postflight_sha256=postflight.postflight_sha256,
            label_pack=labels,
            arm_totals=arm_totals,
            arm_complete_counts=arm_complete_counts,
            candidate_expansion_item_count=sum(
                value > 0 for value in candidate_expansion_counts
            ),
            candidate_expansion_unit_count=sum(candidate_expansion_counts),
            item_rows=item_rows,
            comparisons=comparisons,
        )

    @staticmethod
    def _paired_comparison(
        *,
        left_arm: str,
        right_arm: str,
        view: BlockView,
        labels: Mapping[str, LabelRow],
        utilities: Mapping[tuple[str, str], Fraction],
    ) -> PairedComparison:
        deltas = tuple(
            utilities[(item.item_commitment_sha256, left_arm)]
            - utilities[(item.item_commitment_sha256, right_arm)]
            for item in view.items
        )
        family_nets = tuple(
            (
                family,
                sum(
                    (
                        delta
                        for item, delta in zip(view.items, deltas)
                        if labels[item.item_commitment_sha256].family == family
                    ),
                    Fraction(0),
                ),
            )
            for family in FAMILY_ORDER
        )
        return PairedComparison(
            left_arm=left_arm,
            right_arm=right_arm,
            deltas=deltas,
            family_nets=family_nets,
            exact_test=evaluator.exact_magnitude_preserving_sign_flip(deltas),
        )

    def _authorize_epoch(self) -> EpochAuthorization:
        if (
            self._epoch_transition_count != 0
            or self._epoch_authorization is not None
            or self._a_hold_score is None
            or self._policy_freeze is None
            or not self._a_hold_score.comparison("E0").exact_test.promoted
        ):
            raise TatqaP20FormalControllerError("epoch transition is not uniquely authorized")
        authorization = EpochAuthorization(
            a_hold_score_sha256=self._a_hold_score.score_sha256,
            policy_freeze_sha256=self._policy_freeze.policy_freeze_sha256,
        )
        self._epoch_transition_count = 1
        self._epoch_authorization = authorization
        return authorization

    def _valid_disposition(
        self, *, primary: bool, promoted: bool
    ) -> FormalDisposition:
        status = (
            "valid_nonpromotion"
            if not promoted
            else ("valid_primary_true" if primary else "valid_primary_false")
        )
        return FormalDisposition(
            status=status,
            primary_evaluated=True,
            primary_value=primary,
            efficacy="true" if primary else "false",
            a_hold_promoted=promoted,
            epoch_transition_count=self._epoch_transition_count,
            m_view_released=self._m_view_released,
            m_labels_released=self._m_labels_released,
            replay_authorized=False,
            acquisition_receipt_sha256=self._acquisition_receipt,
            preflight=self._preflight,
            model_sha256=self._model_sha256,
            a_form_fit=self._a_form_fit,
            a_form_archive=self._a_form_archive,
            f_search_archive=self._f_search_archive,
            policy_freeze=self._policy_freeze,
            a_hold_archive=self._a_hold_archive,
            a_hold_score=self._a_hold_score,
            epoch_authorization=self._epoch_authorization,
            m_search_archive=self._m_search_archive,
            m_search_score=self._m_search_score,
        )

    def _invalid_disposition(
        self,
        exc: BaseException,
        *,
        stage: str,
        include_prior_state: bool,
    ) -> FormalDisposition:
        known_a_hold_promotion = bool(
            include_prior_state
            and self._a_hold_score is not None
            and self._a_hold_score.comparison("E0").exact_test.promoted
        )
        return FormalDisposition(
            status="implementation_or_runtime_invalid",
            primary_evaluated=False,
            primary_value=None,
            efficacy="unknown",
            a_hold_promoted=known_a_hold_promotion,
            epoch_transition_count=(
                self._epoch_transition_count if include_prior_state else 0
            ),
            m_view_released=(self._m_view_released if include_prior_state else False),
            m_labels_released=(
                self._m_labels_released if include_prior_state else False
            ),
            replay_authorized=False,
            failure_stage=stage,
            failure_type_sha256=_exception_type_sha256(exc),
            acquisition_receipt_sha256=(
                self._acquisition_receipt if include_prior_state else None
            ),
            preflight=self._preflight if include_prior_state else None,
            model_sha256=self._model_sha256 if include_prior_state else None,
            a_form_fit=self._a_form_fit if include_prior_state else None,
            a_form_archive=self._a_form_archive if include_prior_state else None,
            f_search_archive=(
                self._f_search_archive if include_prior_state else None
            ),
            policy_freeze=self._policy_freeze if include_prior_state else None,
            a_hold_archive=self._a_hold_archive if include_prior_state else None,
            a_hold_score=self._a_hold_score if include_prior_state else None,
            epoch_authorization=(
                self._epoch_authorization if include_prior_state else None
            ),
            m_search_archive=(
                self._m_search_archive if include_prior_state else None
            ),
            m_search_score=self._m_search_score if include_prior_state else None,
        )


__all__ = [
    "AcquisitionBoundary",
    "BLOCK_ARMS",
    "BLOCK_WORK_ARMS",
    "BLOCK_FAMILY_COUNTS",
    "BLOCK_ITEM_COUNTS",
    "BlockView",
    "EpochAuthorization",
    "EvaluatorFitEvidence",
    "FAMILY_ORDER",
    "FormalDisposition",
    "HIPPO_ARM",
    "HIPPO_CONCURRENCY_CAP",
    "ItemView",
    "ItemScoreRow",
    "LabelPack",
    "LabelRow",
    "LogicalActionResult",
    "PairedComparison",
    "PolicyFreeze",
    "PolicySelection",
    "RuntimeBoundary",
    "RuntimePostflight",
    "RuntimePreflight",
    "StageArchive",
    "StageScore",
    "TatqaP20FormalController",
    "TatqaP20FormalControllerError",
]
