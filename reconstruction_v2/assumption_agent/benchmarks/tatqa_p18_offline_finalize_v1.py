"""Independent, local-only recomputation of a terminal TAT-QA P18 study.

The finalizer deliberately does not import the production controller, custody
adapter, or formal-study entrypoint.  It consumes only the already-terminal
canonical JSON disposition (and, optionally, its durable control directory),
recomputes every scored quantity from the persisted actions and late labels,
and writes one exclusive, self-hashed report.  There is no source-data loader,
network capability, evaluator API, retry, replay, or resampling path here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks import tatqa_p18_typed_evaluator_core_v1 as core


VERSION = "tatqa_p18_offline_finalize_v1"
FORMAL_VERSION = "tatqa_p18_formal_study_v1"
CONTROLLER_VERSION = "tatqa_p18_formal_controller_v1"
ADAPTER_VERSION = "tatqa_p18_formal_adapters_v1"
HIPPO_ARM = "official_HippoRAG_retrieve_only"
TYPED_PLAN_TRANSPORT_SCHEMA = (
    "tatqa_p18_formal_runtime_v1_typed_plan_transport_receipt_v1"
)
HIPPO_TRANSPORT_SCHEMA = "tatqa_p18_formal_runtime_v1_hippo_transport_receipt_v1"
SYSTEMD_UNIT_CLOSURE_SCHEMA = (
    "tatqa_p18_formal_runtime_v1_systemd_unit_closure_v1"
)
SYSTEMD_START_POLICY_SCHEMA = (
    "tatqa_p18_formal_runtime_v1_systemd_start_policy_v1"
)
FILESYSTEM_ISOLATION = (
    "systemd_InaccessiblePaths_official_source_and_acquisition_v1"
)
NETWORK_PROPERTIES = ["IPAddressDeny=any", "RestrictAddressFamilies=AF_UNIX"]
FAMILIES = ("TABLE", "TEXT", "TABLE_TEXT")
ITEM_COUNTS = {"A_form": 48, "F_search": 36, "A_hold": 30, "M_search": 30}
FAMILY_COUNTS = {"A_form": 16, "A_hold": 10, "M_search": 10}
ARMS = {
    "A_form": ("P0", "P1"),
    "F_search": ("P0", "P1"),
    "A_hold": ("E0", "E1", "RAW", HIPPO_ARM),
    "M_search": ("E0", "E1", "RAW", HIPPO_ARM),
}
WORK_ARMS = {
    "A_form": ("P0", "P1"),
    "F_search": ("P0", "P1"),
    "A_hold": ("P0", "P1", "RAW", HIPPO_ARM),
    "M_search": ("P0", "P1", "RAW", HIPPO_ARM),
}
EVIDENCE_ARTIFACT = {
    "A_form_fit": "A_form_fit",
    "F_search_policy_freeze": "policy_freeze",
    "A_hold_score": "A_hold_score",
    "M_search_score": "M_search_score",
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_UNIT_ID = re.compile(r"(?:T:(?:0|[1-9][0-9]*)|P:[1-9][0-9]*)\Z")


class TatqaP18OfflineFinalizeError(RuntimeError):
    """The terminal artifact is incomplete, noncanonical, or does not replay."""


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
        raise TatqaP18OfflineFinalizeError("value is not canonical JSON") from exc


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("nonfinite JSON constant")


def _read_regular(path: Path, *, field: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise TatqaP18OfflineFinalizeError(f"{field} is unavailable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise TatqaP18OfflineFinalizeError(f"{field} is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                raise TatqaP18OfflineFinalizeError(f"{field} changed during open")
            raw = handle.read()
    except OSError as exc:
        raise TatqaP18OfflineFinalizeError(f"{field} cannot be read safely") from exc
    if len(raw) != before.st_size:
        raise TatqaP18OfflineFinalizeError(f"{field} changed during read")
    return raw


def _strict_json_file(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, field=field)
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TatqaP18OfflineFinalizeError(f"{field} is not strict JSON") from exc
    if not isinstance(value, dict) or _canonical_bytes(value) != raw:
        raise TatqaP18OfflineFinalizeError(f"{field} is not canonical JSON")
    return value, raw


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TatqaP18OfflineFinalizeError(f"{field} is not an object")
    return dict(value)


def _keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    if set(value) != expected:
        raise TatqaP18OfflineFinalizeError(f"{field} key registry drifted")


def _sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TatqaP18OfflineFinalizeError(f"{field} is not a lowercase SHA-256")
    return value


def _fraction(value: object, *, field: str) -> Fraction:
    row = _mapping(value, field=field)
    _keys(row, {"numerator", "denominator"}, field=field)
    numerator = row["numerator"]
    denominator = row["denominator"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise TatqaP18OfflineFinalizeError(f"{field} is not an exact fraction")
    result = Fraction(numerator, denominator)
    if result.numerator != numerator or result.denominator != denominator:
        raise TatqaP18OfflineFinalizeError(f"{field} is not reduced")
    return result


def _fraction_payload(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _verify_self_hash(
    value: Mapping[str, Any], *, field: str, schema: str | None = None
) -> str:
    claimed = _sha(value.get(field), field=field)
    body = dict(value)
    del body[field]
    if schema is not None and body.get("schema") != schema:
        raise TatqaP18OfflineFinalizeError(f"{field} schema drifted")
    if _semantic_hash(body) != claimed:
        raise TatqaP18OfflineFinalizeError(f"{field} does not bind its payload")
    return claimed


def _action(value: object, *, block: str, expected_arm: str) -> tuple[dict[str, Any], str]:
    row = _mapping(value, field=f"{block} {expected_arm} action")
    _keys(
        row,
        {
            "P1_minus_P0_features",
            "behavior_sha256",
            "block",
            "fixed_feature_order",
            "item_commitment_sha256",
            "logical_arm",
            "ordered_top5",
            "schema",
            "selected_policy_id",
            "source_action_sha256",
        },
        field=f"{block} action",
    )
    if (
        row["schema"] != f"{CONTROLLER_VERSION}_logical_action_result_v1"
        or row["block"] != block
        or row["logical_arm"] != expected_arm
        or row["fixed_feature_order"] != list(core.FEATURE_ORDER)
    ):
        raise TatqaP18OfflineFinalizeError(f"{block} action identity drifted")
    _sha(row["item_commitment_sha256"], field=f"{block} action item")
    _sha(row["source_action_sha256"], field=f"{block} source action")
    top5 = row["ordered_top5"]
    if (
        not isinstance(top5, list)
        or len(top5) != core.TOP_K
        or len(set(top5)) != core.TOP_K
        or any(not isinstance(unit, str) or _UNIT_ID.fullmatch(unit) is None for unit in top5)
    ):
        raise TatqaP18OfflineFinalizeError(f"{block} ordered top-five drifted")
    try:
        features = core.feature_vector(row["P1_minus_P0_features"])
        behavior = core.canonical_behavior_hash(top5)
    except Exception as exc:
        raise TatqaP18OfflineFinalizeError(f"{block} action core contract drifted") from exc
    if row["P1_minus_P0_features"] != list(features) or row["behavior_sha256"] != behavior:
        raise TatqaP18OfflineFinalizeError(f"{block} action behavior/features drifted")
    policy = row["selected_policy_id"]
    if expected_arm in {"P0", "E0"} and policy != core.P0_POLICY_ID:
        raise TatqaP18OfflineFinalizeError(f"{block} P0/E0 policy drifted")
    if expected_arm == "P1" and policy != core.P1_POLICY_ID:
        raise TatqaP18OfflineFinalizeError(f"{block} P1 policy drifted")
    if expected_arm == "E1" and policy not in {core.P0_POLICY_ID, core.P1_POLICY_ID}:
        raise TatqaP18OfflineFinalizeError(f"{block} E1 policy drifted")
    if expected_arm in {"RAW", HIPPO_ARM} and (
        policy != expected_arm or any(features)
    ):
        raise TatqaP18OfflineFinalizeError(f"{block} baseline policy drifted")
    if expected_arm in {"RAW", HIPPO_ARM}:
        source_payload = {
            "schema": f"{CONTROLLER_VERSION}_injected_baseline_action_v1",
            "block": block,
            "item_commitment_sha256": row["item_commitment_sha256"],
            "logical_arm": expected_arm,
            "ordered_top5": top5,
            "behavior_sha256": behavior,
        }
        if row["source_action_sha256"] != _semantic_hash(source_payload):
            raise TatqaP18OfflineFinalizeError(f"{block} baseline source action drifted")
    return row, _semantic_hash(row)


def _parse_model(value: object) -> tuple[core.PairedDeltaRidgeModel, dict[str, Any]]:
    payload = _mapping(value, field="E1 model")
    _keys(
        payload,
        {
            "coefficient_float64_hex",
            "feature_order",
            "intercept_float64_hex",
            "intercept_penalized",
            "population_mean_float64_hex",
            "population_std_float64_hex",
            "ridge_lambda",
            "scaler",
            "solver",
            "zero_variance_maps_to_zero",
        },
        field="E1 model",
    )
    try:
        if (
            payload["feature_order"] != list(core.FEATURE_ORDER)
            or payload["scaler"]
            != "A_form_population_mean_and_population_standard_deviation_v1"
            or payload["zero_variance_maps_to_zero"] is not True
            or payload["ridge_lambda"] != 1
            or payload["intercept_penalized"] is not False
        ):
            raise ValueError("model metadata")
        model = core.PairedDeltaRidgeModel(
            population_mean=tuple(float.fromhex(row) for row in payload["population_mean_float64_hex"]),
            population_std=tuple(float.fromhex(row) for row in payload["population_std_float64_hex"]),
            intercept=float.fromhex(payload["intercept_float64_hex"]),
            coefficients=tuple(float.fromhex(row) for row in payload["coefficient_float64_hex"]),
            solver=payload["solver"],
        )
    except (KeyError, TypeError, ValueError, core.TatqaP18TypedEvaluatorError) as exc:
        raise TatqaP18OfflineFinalizeError("E1 model payload drifted") from exc
    if model.payload() != payload:
        raise TatqaP18OfflineFinalizeError("E1 model is not canonical")
    return model, payload


def _positive_int(value: object, *, field: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TatqaP18OfflineFinalizeError(f"{field} is not a valid positive integer")
    return value


def _systemd_unit_closure(
    value: object, *, unit_name_sha256: str
) -> dict[str, Any]:
    receipt = _mapping(value, field="systemd unit closure")
    _keys(
        receipt,
        {
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
        },
        field="systemd unit closure",
    )
    for field in (
        "control_group_sha256",
        "systemctl_reset_failed_stderr_sha256",
        "systemctl_reset_failed_stdout_sha256",
        "systemctl_show_stderr_sha256",
        "systemctl_show_stdout_sha256",
        "unit_name_sha256",
    ):
        _sha(receipt[field], field=f"systemd closure {field}")
    if (
        receipt["schema"] != SYSTEMD_UNIT_CLOSURE_SCHEMA
        or receipt["unit_name_sha256"] != unit_name_sha256
        or receipt["load_state"] != "not-found"
        or receipt["active_state"] != "inactive"
        or receipt["sub_state"] != "dead"
        or type(receipt["main_pid"]) is not int
        or receipt["main_pid"] != 0
        or type(receipt["control_group_process_count"]) is not int
        or receipt["control_group_process_count"] != 0
        or type(receipt["control_group_thread_count"]) is not int
        or receipt["control_group_thread_count"] != 0
        or type(receipt["systemctl_show_returncode"]) is not int
        or receipt["systemctl_show_returncode"] != 0
        or type(receipt["systemctl_reset_failed_returncode"]) is not int
        or receipt["systemctl_reset_failed_returncode"] not in {0, 1}
    ):
        raise TatqaP18OfflineFinalizeError("systemd unit closure drifted")
    return receipt


def _systemd_start_policy(
    value: object, *, unit_name_sha256: str
) -> dict[str, Any]:
    receipt = _mapping(value, field="systemd start policy")
    _keys(
        receipt,
        {
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
        },
        field="systemd start policy",
    )
    for field in (
        "control_group_sha256",
        "systemctl_show_stderr_sha256",
        "systemctl_show_stdout_sha256",
        "unit_name_sha256",
    ):
        _sha(receipt[field], field=f"systemd start policy {field}")
    if (
        receipt["schema"] != SYSTEMD_START_POLICY_SCHEMA
        or receipt["unit_name_sha256"] != unit_name_sha256
        or receipt["load_state"] != "loaded"
        or receipt["active_state"] != "active"
        or receipt["sub_state"] != "running"
        or type(receipt["main_pid"]) is not int
        or receipt["main_pid"] <= 1
        or type(receipt["tasks_max"]) is not int
        or receipt["tasks_max"] != 3
        or receipt["kill_mode"] != "control-group"
        or type(receipt["systemctl_show_returncode"]) is not int
        or receipt["systemctl_show_returncode"] != 0
    ):
        raise TatqaP18OfflineFinalizeError("systemd start policy drifted")
    return receipt


def _qwen_transport_receipt(
    value: object, *, block: str, item_count: int
) -> dict[str, Any]:
    receipt = _mapping(value, field=f"{block} Qwen transport receipt")
    _keys(
        receipt,
        {
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
        },
        field=f"{block} Qwen transport receipt",
    )
    started = _positive_int(
        receipt["model_execution_started_monotonic_ns"],
        field=f"{block} Qwen interval start",
    )
    finished = _positive_int(
        receipt["model_execution_finished_monotonic_ns"],
        field=f"{block} Qwen interval finish",
    )
    pid = _positive_int(receipt["worker_pid"], field=f"{block} Qwen PID", minimum=2)
    context = _positive_int(
        receipt["model_context_tokens"], field=f"{block} Qwen model context"
    )
    for field in ("input_sha256", "output_sha256", "stderr_sha256", "stdout_sha256"):
        _sha(receipt[field], field=f"{block} Qwen {field}")
    unit_name_sha256 = _sha(
        receipt["systemd_unit_name_sha256"],
        field=f"{block} Qwen systemd unit name",
    )
    _systemd_unit_closure(
        receipt["systemd_unit_closure"],
        unit_name_sha256=unit_name_sha256,
    )
    if (
        receipt["schema"] != TYPED_PLAN_TRANSPORT_SCHEMA
        or receipt["block"] != block
        or type(receipt["item_count"]) is not int
        or receipt["item_count"] != item_count
        or type(receipt["batch_size"]) is not int
        or receipt["batch_size"] != 4
        or receipt["physical_GPU"] != "1"
        or receipt["filesystem_isolation"] != FILESYSTEM_ISOLATION
        or receipt["network_properties"] != NETWORK_PROPERTIES
        or context < 16_640
        or finished <= started
    ):
        raise TatqaP18OfflineFinalizeError(f"{block} Qwen transport contract drifted")
    return {
        "payload": receipt,
        "sha256": _semantic_hash(receipt),
        "pid": pid,
        "started": started,
        "finished": finished,
    }


def _hippo_transport_receipt(
    value: object, *, block: str, item_commitments: set[str]
) -> dict[str, Any]:
    receipt = _mapping(value, field=f"{block} Hippo transport receipt")
    _keys(
        receipt,
        {
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
            "maximum_worker_process_threads",
            "systemd_start_policy",
            "systemd_start_policy_sha256",
            "systemd_tasks_max",
            "systemd_unit_closure",
            "systemd_unit_name_sha256",
            "thread_monitor_process_reservation",
            "visible_GPU",
            "worker_pid",
        },
        field=f"{block} Hippo transport receipt",
    )
    item = receipt["item_commitment_sha256"]
    _sha(item, field=f"{block} Hippo item commitment")
    started = _positive_int(
        receipt["model_execution_started_monotonic_ns"],
        field=f"{block} Hippo interval start",
    )
    finished = _positive_int(
        receipt["model_execution_finished_monotonic_ns"],
        field=f"{block} Hippo interval finish",
    )
    pid = _positive_int(receipt["worker_pid"], field=f"{block} Hippo PID", minimum=2)
    threads = tuple(
        _positive_int(receipt[field], field=f"{block} Hippo {field}")
        for field in (
            "configured_torch_intraop_threads",
            "configured_torch_interop_threads",
            "observed_process_thread_peak",
        )
    )
    for field in (
        "input_file_sha256",
        "input_semantic_sha256",
        "output_file_sha256",
        "stderr_sha256",
        "stdout_sha256",
    ):
        _sha(receipt[field], field=f"{block} Hippo {field}")
    unit_name_sha256 = _sha(
        receipt["systemd_unit_name_sha256"],
        field=f"{block} Hippo systemd unit name",
    )
    start_policy_sha256 = _sha(
        receipt["systemd_start_policy_sha256"],
        field=f"{block} Hippo systemd start policy",
    )
    start_policy = _systemd_start_policy(
        receipt["systemd_start_policy"],
        unit_name_sha256=unit_name_sha256,
    )
    closure = _systemd_unit_closure(
        receipt["systemd_unit_closure"],
        unit_name_sha256=unit_name_sha256,
    )
    if (
        receipt["schema"] != HIPPO_TRANSPORT_SCHEMA
        or receipt["block"] != block
        or item not in item_commitments
        or type(receipt["CPU_threads"]) is not int
        or receipt["CPU_threads"] != 2
        or receipt["visible_GPU"] != ""
        or receipt["filesystem_isolation"] != FILESYSTEM_ISOLATION
        or receipt["network_properties"] != NETWORK_PROPERTIES
        or type(receipt["systemd_tasks_max"]) is not int
        or receipt["systemd_tasks_max"] != 3
        or type(receipt["thread_monitor_process_reservation"]) is not int
        or receipt["thread_monitor_process_reservation"] != 1
        or type(receipt["maximum_worker_process_threads"]) is not int
        or receipt["maximum_worker_process_threads"] != 2
        or receipt["thread_monitor_process_reservation"]
        + receipt["maximum_worker_process_threads"]
        != receipt["systemd_tasks_max"]
        or start_policy["main_pid"] != pid
        or start_policy["control_group_sha256"]
        != closure["control_group_sha256"]
        or _semantic_hash(start_policy) != start_policy_sha256
        or finished <= started
        or any(row > 2 for row in threads)
    ):
        raise TatqaP18OfflineFinalizeError(f"{block} Hippo transport contract drifted")
    return {
        "payload": receipt,
        "sha256": _semantic_hash(receipt),
        "item": item,
        "pid": pid,
        "started": started,
        "finished": finished,
        "configured_torch_intraop_threads": threads[0],
        "configured_torch_interop_threads": threads[1],
        "observed_process_thread_peak": threads[2],
    }


def _archive(value: object, *, block: str, model: core.PairedDeltaRidgeModel | None) -> dict[str, Any]:
    archive = _mapping(value, field=f"{block} archive")
    expected_keys = {
        "actual_inference_preparation",
        "api_or_online_evaluator_calls",
        "block",
        "candidate_work_additional_effect_gate_count",
        "effect_gate_count",
        "effect_gate_scope",
        "eager_submit_complete_before_first_join",
        "external_network_calls",
        "hippo_concurrency_cap",
        "hippo_executor_dedicated",
        "hippo_observed_peak",
        "logical_action_count",
        "logical_action_result_sha256s",
        "logical_action_results",
        "logical_action_submission_accounting_only_not_model_inference",
        "logical_action_terminal_count",
        "logical_behavior_sha256s",
        "predeclared_logical_arms",
        "retry_replay_resample_provider_switch",
        "schema",
        "shared_candidate_work_action_result_sha256s",
        "shared_candidate_work_action_results",
        "shared_candidate_work_only_not_effect_gate",
        "submit_count_before_first_join",
        "submitted_work_action_result_sha256s",
        "submitted_work_action_results",
        "submitted_work_arms",
        "submitted_work_count",
        "submitted_work_terminal_count",
    }
    _keys(archive, expected_keys, field=f"{block} archive")
    item_count = ITEM_COUNTS[block]
    logical_count = item_count * len(ARMS[block])
    submitted_count = item_count * len(WORK_ARMS[block])
    if (
        archive["schema"] != f"{CONTROLLER_VERSION}_stage_archive_v1"
        or archive["block"] != block
        or archive["submitted_work_count"] != submitted_count
        or archive["submitted_work_terminal_count"] != submitted_count
        or archive["logical_action_count"] != logical_count
        or archive["logical_action_terminal_count"] != logical_count
        or archive["submit_count_before_first_join"] != submitted_count
        or archive["eager_submit_complete_before_first_join"] is not True
        or archive["submitted_work_arms"] != list(WORK_ARMS[block])
        or archive["predeclared_logical_arms"] != list(ARMS[block])
        or archive["logical_action_submission_accounting_only_not_model_inference"] is not True
        or archive["shared_candidate_work_only_not_effect_gate"] is not True
        or archive["effect_gate_count"] != 1
        or archive["effect_gate_scope"] != "single_predeclared_E1_effect_rule"
        or archive["candidate_work_additional_effect_gate_count"] != 0
        or archive["external_network_calls"] != 0
        or archive["api_or_online_evaluator_calls"] != 0
        or archive["retry_replay_resample_provider_switch"] != 0
    ):
        raise TatqaP18OfflineFinalizeError(f"{block} archive accounting drifted")
    has_hippo = block in {"A_hold", "M_search"}
    if (
        archive["hippo_executor_dedicated"] is not has_hippo
        or archive["hippo_concurrency_cap"] != (8 if has_hippo else 0)
        or (
            has_hippo
            and (
                isinstance(archive["hippo_observed_peak"], bool)
                or not isinstance(archive["hippo_observed_peak"], int)
                or not 1 <= archive["hippo_observed_peak"] <= 8
            )
        )
        or (not has_hippo and archive["hippo_observed_peak"] != 0)
    ):
        raise TatqaP18OfflineFinalizeError(f"{block} Hippo accounting drifted")

    raw_logical = archive["logical_action_results"]
    if not isinstance(raw_logical, list) or len(raw_logical) != logical_count:
        raise TatqaP18OfflineFinalizeError(f"{block} logical actions are incomplete")
    logical: list[dict[str, Any]] = []
    logical_hashes: list[str] = []
    for index, raw in enumerate(raw_logical):
        arm = ARMS[block][index % len(ARMS[block])]
        row, action_hash = _action(raw, block=block, expected_arm=arm)
        logical.append(row)
        logical_hashes.append(action_hash)
    item_order = [
        logical[index * len(ARMS[block])]["item_commitment_sha256"]
        for index in range(item_count)
    ]
    if len(set(item_order)) != item_count or any(
        logical[index * len(ARMS[block]) + offset]["item_commitment_sha256"] != item
        for index, item in enumerate(item_order)
        for offset in range(len(ARMS[block]))
    ):
        raise TatqaP18OfflineFinalizeError(f"{block} logical item registry drifted")
    if (
        archive["logical_action_result_sha256s"] != logical_hashes
        or archive["logical_behavior_sha256s"]
        != [row["behavior_sha256"] for row in logical]
    ):
        raise TatqaP18OfflineFinalizeError(f"{block} logical action hash registry drifted")

    candidate_raw = archive["shared_candidate_work_action_results"]
    candidate_count = 2 * item_count if has_hippo else 0
    if not isinstance(candidate_raw, list) or len(candidate_raw) != candidate_count:
        raise TatqaP18OfflineFinalizeError(f"{block} candidate action coverage drifted")
    candidate: list[dict[str, Any]] = []
    candidate_hashes: list[str] = []
    for index, raw in enumerate(candidate_raw):
        arm = ("P0", "P1")[index % 2]
        row, action_hash = _action(raw, block=block, expected_arm=arm)
        candidate.append(row)
        candidate_hashes.append(action_hash)
    if archive["shared_candidate_work_action_result_sha256s"] != candidate_hashes:
        raise TatqaP18OfflineFinalizeError(f"{block} candidate action hashes drifted")

    by_key = {
        (row["item_commitment_sha256"], row["logical_arm"]): row
        for row in (*logical, *candidate)
    }
    if len(by_key) != len(logical) + len(candidate):
        raise TatqaP18OfflineFinalizeError(f"{block} duplicate action key")
    if has_hippo:
        if model is None:
            raise TatqaP18OfflineFinalizeError(f"{block} lacks replay model")
        for item in item_order:
            p0, p1 = by_key[(item, "P0")], by_key[(item, "P1")]
            e0, e1 = by_key[(item, "E0")], by_key[(item, "E1")]
            chosen = p1 if model.predict(p1["P1_minus_P0_features"]) > 0.0 else p0
            if (
                e0["ordered_top5"] != p0["ordered_top5"]
                or e0["source_action_sha256"] != p0["source_action_sha256"]
                or e0["P1_minus_P0_features"] != p1["P1_minus_P0_features"]
                or e1["ordered_top5"] != chosen["ordered_top5"]
                or e1["source_action_sha256"] != chosen["source_action_sha256"]
                or e1["P1_minus_P0_features"] != p1["P1_minus_P0_features"]
            ):
                raise TatqaP18OfflineFinalizeError(f"{block} evaluator derivation drifted")

    submitted_raw = archive["submitted_work_action_results"]
    if not isinstance(submitted_raw, list) or len(submitted_raw) != submitted_count:
        raise TatqaP18OfflineFinalizeError(f"{block} submitted work is incomplete")
    expected_submitted = [by_key[(item, arm)] for item in item_order for arm in WORK_ARMS[block]]
    if submitted_raw != expected_submitted or archive["submitted_work_action_result_sha256s"] != [
        _semantic_hash(row) for row in expected_submitted
    ]:
        raise TatqaP18OfflineFinalizeError(f"{block} submitted work registry drifted")

    inference = _mapping(archive["actual_inference_preparation"], field=f"{block} inference")
    inference_keys = {
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
    _keys(inference, inference_keys, field=f"{block} inference")
    _verify_self_hash(
        inference,
        field="preparation_inference_receipt_sha256",
        schema=f"{ADAPTER_VERSION}_preparation_inference_receipt_v1",
    )
    expected_hippo = item_count if has_hippo else 0
    qwen_transport = _qwen_transport_receipt(
        inference["qwen_transport_receipt"], block=block, item_count=item_count
    )
    raw_hippo_receipts = inference["hippo_transport_receipts"]
    if not isinstance(raw_hippo_receipts, list):
        raise TatqaP18OfflineFinalizeError(f"{block} full Hippo receipts are absent")
    validated_hippo = [
        _hippo_transport_receipt(row, block=block, item_commitments=set(item_order))
        for row in raw_hippo_receipts
    ]
    hippo_transport_hashes = inference["hippo_transport_receipt_sha256s"]
    hippo_worker_hashes = inference["hippo_worker_receipt_sha256s"]
    hippo_pids = inference["hippo_worker_pids"]
    overlap_witnesses = [
        row["item"]
        for row in validated_hippo
        if max(qwen_transport["started"], row["started"])
        < min(qwen_transport["finished"], row["finished"])
    ]
    declared_witnesses = inference["qwen_hippo_overlap_witness_item_commitments"]
    if (
        inference.get("block") != block
        or inference.get("retry_replay_resample_provider_switch") != 0
        or inference["actual_model_future_expected_count"] != 1 + expected_hippo
        or inference["actual_model_future_submit_count_before_first_join"] != 1 + expected_hippo
        or inference["all_actual_model_futures_submitted_before_first_join"] is not True
        or inference["qwen_batch_item_count"] != item_count
        or inference["qwen_batch_submitted_count"] != 1
        or inference["qwen_batch_terminal_count"] != 1
        or inference["qwen_executor_dedicated"] is not True
        or inference["minilm_raw_compiled_item_count"] != item_count
        or inference["qwen_worker_pid"] != qwen_transport["pid"]
        or inference["qwen_transport_receipt_sha256"] != qwen_transport["sha256"]
        or inference["qwen_transport_receipt"] != qwen_transport["payload"]
        or not isinstance(inference["qwen_worker_receipt_sha256"], str)
        or _SHA256.fullmatch(inference["qwen_worker_receipt_sha256"]) is None
        or len(validated_hippo) != expected_hippo
        or len({row["item"] for row in validated_hippo}) != expected_hippo
        or [row["item"] for row in validated_hippo]
        != sorted(row["item"] for row in validated_hippo)
        or not isinstance(hippo_transport_hashes, list)
        or [row["sha256"] for row in validated_hippo] != hippo_transport_hashes
        or not isinstance(hippo_worker_hashes, list)
        or len(hippo_worker_hashes) != expected_hippo
        or any(not isinstance(row, str) or _SHA256.fullmatch(row) is None for row in hippo_worker_hashes)
        or not isinstance(hippo_pids, list)
        or [row["pid"] for row in validated_hippo] != hippo_pids
        or not isinstance(declared_witnesses, list)
        or declared_witnesses != overlap_witnesses
    ):
        raise TatqaP18OfflineFinalizeError(f"{block} inference receipt drifted")
    if has_hippo:
        if (
            inference["hippo_actual_concurrency_cap"] != 8
            or inference["hippo_actual_observed_peak"] != archive["hippo_observed_peak"]
            or inference["hippo_future_submitted_count"] != expected_hippo
            or inference["hippo_future_terminal_count"] != expected_hippo
            or inference["hippo_future_consumed_count"] != expected_hippo
            or inference["hippo_executor_dedicated"] is not True
            or inference["qwen_hippo_independent_executors"] is not True
            or inference["qwen_hippo_overlap_observed"] is not True
            or not overlap_witnesses
        ):
            raise TatqaP18OfflineFinalizeError(f"{block} Hippo inference receipt drifted")
    elif any(
        (
            inference["hippo_actual_concurrency_cap"] != 0,
            inference["hippo_actual_observed_peak"] != 0,
            inference["hippo_future_submitted_count"] != 0,
            inference["hippo_future_terminal_count"] != 0,
            inference["hippo_future_consumed_count"] != 0,
            inference["hippo_executor_dedicated"] is not False,
            inference["qwen_hippo_independent_executors"] is not False,
            inference["qwen_hippo_overlap_observed"] is not False,
            bool(overlap_witnesses),
            declared_witnesses != [],
        )
    ):
        raise TatqaP18OfflineFinalizeError(f"{block} non-Hippo inference drifted")
    return {
        "payload": archive,
        "sha256": _semantic_hash(archive),
        "item_order": item_order,
        "actions": by_key,
        "logical_action_count": logical_count,
        "submitted_work_count": submitted_count,
        "qwen_transport_receipt": qwen_transport["payload"],
        "hippo_transport_receipts": [row["payload"] for row in validated_hippo],
        "transport_receipts": [
            qwen_transport["payload"],
            *(row["payload"] for row in validated_hippo),
        ],
        "transport_receipt_sha256s": [
            qwen_transport["sha256"],
            *(row["sha256"] for row in validated_hippo),
        ],
        "worker_pids": [
            qwen_transport["pid"],
            *(row["pid"] for row in validated_hippo),
        ],
        "overlap_witness_item_commitments": overlap_witnesses,
    }


def _label_pack(value: object, *, block: str) -> dict[str, Any]:
    pack = _mapping(value, field=f"{block} label pack")
    _keys(pack, {"block", "rows", "schema"}, field=f"{block} label pack")
    rows = pack["rows"]
    if (
        pack["schema"] != f"{CONTROLLER_VERSION}_trusted_label_pack_commitment_v1"
        or pack["block"] != block
        or not isinstance(rows, list)
        or len(rows) != ITEM_COUNTS[block]
    ):
        raise TatqaP18OfflineFinalizeError(f"{block} label pack drifted")
    parsed: list[dict[str, Any]] = []
    for raw in rows:
        row = _mapping(raw, field=f"{block} label row")
        _keys(
            row,
            {"canonical_gold_units", "family", "item_commitment_sha256"},
            field=f"{block} label row",
        )
        gold = row["canonical_gold_units"]
        if (
            row["family"] not in FAMILIES
            or not isinstance(gold, list)
            or not 1 <= len(gold) <= core.TOP_K
            or len(set(gold)) != len(gold)
            or any(not isinstance(unit, str) or _UNIT_ID.fullmatch(unit) is None for unit in gold)
        ):
            raise TatqaP18OfflineFinalizeError(f"{block} label row drifted")
        _sha(row["item_commitment_sha256"], field=f"{block} label item")
        parsed.append(row)
    if (
        len({row["item_commitment_sha256"] for row in parsed}) != len(parsed)
        or Counter(row["family"] for row in parsed)
        != Counter({family: FAMILY_COUNTS[block] for family in FAMILIES})
    ):
        raise TatqaP18OfflineFinalizeError(f"{block} label registry drifted")
    return {"payload": pack, "sha256": _semantic_hash(pack), "rows": parsed}


def _fit(value: object, archive: Mapping[str, Any]) -> dict[str, Any]:
    fit = _mapping(value, field="A_form fit")
    _keys(
        fit,
        {
            "E1_model",
            "E1_model_sha256",
            "archive_sha256",
            "feature_rows",
            "fixed_feature_order",
            "label_pack",
            "label_pack_sha256",
            "postflight_sha256",
            "schema",
            "scoring",
            "utility_deltas",
        },
        field="A_form fit",
    )
    labels = _label_pack(fit["label_pack"], block="A_form")
    model, model_payload = _parse_model(fit["E1_model"])
    rows = fit["feature_rows"]
    deltas_raw = fit["utility_deltas"]
    if (
        fit["schema"] != f"{CONTROLLER_VERSION}_A_form_evaluator_fit_evidence_v1"
        or fit["scoring"] != "local_offline_exact_only"
        or fit["fixed_feature_order"] != list(core.FEATURE_ORDER)
        or fit["archive_sha256"] != archive["sha256"]
        or fit["label_pack_sha256"] != labels["sha256"]
        or not isinstance(rows, list)
        or not isinstance(deltas_raw, list)
        or len(rows) != ITEM_COUNTS["A_form"]
        or len(deltas_raw) != ITEM_COUNTS["A_form"]
    ):
        raise TatqaP18OfflineFinalizeError("A_form fit identity drifted")
    _sha(fit["postflight_sha256"], field="A_form postflight")
    features = []
    try:
        for row in rows:
            normalized = core.feature_vector(row)
            if row != list(normalized):
                raise ValueError("feature encoding")
            features.append(normalized)
    except Exception as exc:
        raise TatqaP18OfflineFinalizeError("A_form feature rows drifted") from exc
    deltas = [_fraction(row, field="A_form utility delta") for row in deltas_raw]
    if labels["rows"] and [row["item_commitment_sha256"] for row in labels["rows"]] != archive["item_order"]:
        raise TatqaP18OfflineFinalizeError("A_form label/action order drifted")
    recalculated: list[Fraction] = []
    for index, item in enumerate(archive["item_order"]):
        p0 = archive["actions"][(item, "P0")]
        p1 = archive["actions"][(item, "P1")]
        gold = labels["rows"][index]["canonical_gold_units"]
        if features[index] != tuple(p1["P1_minus_P0_features"]):
            raise TatqaP18OfflineFinalizeError("A_form feature/action binding drifted")
        recalculated.append(
            core.item_utility(p1["ordered_top5"], gold)
            - core.item_utility(p0["ordered_top5"], gold)
        )
    if deltas != recalculated:
        raise TatqaP18OfflineFinalizeError("A_form exact utility deltas drifted")
    replayed = core.fit_paired_delta_ridge(features, deltas)
    if (
        replayed.payload() != model_payload
        or fit["E1_model_sha256"] != _semantic_hash(model_payload)
    ):
        raise TatqaP18OfflineFinalizeError("A_form ridge replay drifted")
    return {
        "payload": fit,
        "sha256": _semantic_hash(fit),
        "postflight_sha256": fit["postflight_sha256"],
        "model": model,
        "model_payload": model_payload,
        "model_sha256": _semantic_hash(model_payload),
        "label_pack_sha256": labels["sha256"],
    }


def _policy_freeze(value: object, archive: Mapping[str, Any], fit: Mapping[str, Any]) -> dict[str, Any]:
    freeze = _mapping(value, field="policy freeze")
    _keys(
        freeze,
        {
            "E1_model",
            "E1_model_sha256",
            "F_search_archive_sha256",
            "F_search_postflight_sha256",
            "behavior_hashes_always_reported",
            "label_pack_created_or_released",
            "rows",
            "schema",
        },
        field="policy freeze",
    )
    rows = freeze["rows"]
    if (
        freeze["schema"] != f"{CONTROLLER_VERSION}_F_search_policy_freeze_v1"
        or freeze["F_search_archive_sha256"] != archive["sha256"]
        or freeze["E1_model"] != fit["model_payload"]
        or freeze["E1_model_sha256"] != fit["model_sha256"]
        or freeze["label_pack_created_or_released"] is not False
        or freeze["behavior_hashes_always_reported"] is not True
        or not isinstance(rows, list)
        or len(rows) != ITEM_COUNTS["F_search"]
    ):
        raise TatqaP18OfflineFinalizeError("policy freeze identity drifted")
    _sha(freeze["F_search_postflight_sha256"], field="F_search postflight")
    row_keys = {
        "E0_action_result_sha256",
        "E0_behavior_sha256",
        "E1_action_result_sha256",
        "E1_behavior_sha256",
        "E1_selected_policy_id",
        "item_commitment_sha256",
        "same_behavior",
    }
    for item, raw in zip(archive["item_order"], rows, strict=True):
        row = _mapping(raw, field="policy row")
        _keys(row, row_keys, field="policy row")
        p0 = archive["actions"][(item, "P0")]
        p1 = archive["actions"][(item, "P1")]
        chosen = p1 if fit["model"].predict(p1["P1_minus_P0_features"]) > 0.0 else p0
        expected = {
            "item_commitment_sha256": item,
            "E0_action_result_sha256": _semantic_hash(p0),
            "E1_action_result_sha256": _semantic_hash(chosen),
            "E0_behavior_sha256": p0["behavior_sha256"],
            "E1_behavior_sha256": chosen["behavior_sha256"],
            "E1_selected_policy_id": chosen["selected_policy_id"],
            "same_behavior": p0["behavior_sha256"] == chosen["behavior_sha256"],
        }
        if row != expected:
            raise TatqaP18OfflineFinalizeError("policy row replay drifted")
    return {
        "payload": freeze,
        "sha256": _semantic_hash(freeze),
        "postflight_sha256": freeze["F_search_postflight_sha256"],
    }


def _comparison_payload(
    *, left: str, right: str, deltas: Sequence[Fraction], families: Sequence[str]
) -> dict[str, Any]:
    family_nets = {
        family: sum(
            (delta for delta, observed in zip(deltas, families, strict=True) if observed == family),
            Fraction(0),
        )
        for family in FAMILIES
    }
    exact = core.exact_magnitude_preserving_sign_flip(tuple(deltas))
    delta_payload = [_fraction_payload(row) for row in deltas]
    return {
        "left_arm": left,
        "right_arm": right,
        "net_U": _fraction_payload(sum(deltas, Fraction(0))),
        "gain_count": sum(row > 0 for row in deltas),
        "harm_count": sum(row < 0 for row in deltas),
        "tie_count": sum(row == 0 for row in deltas),
        "family_nets": {
            family: _fraction_payload(family_nets[family]) for family in FAMILIES
        },
        "exact_test": exact.payload(),
        "paired_deltas": delta_payload,
        "paired_delta_sha256": _semantic_hash(delta_payload),
    }


def _score(value: object, *, block: str, archive: Mapping[str, Any]) -> dict[str, Any]:
    score = _mapping(value, field=f"{block} score")
    _keys(
        score,
        {
            "archive_sha256",
            "arm_complete_counts",
            "arm_totals",
            "block",
            "candidate_expansion_outside_P0_top5",
            "comparisons",
            "item_exact_utility_rows",
            "label_pack",
            "label_pack_sha256",
            "postflight_sha256",
            "schema",
            "scoring",
        },
        field=f"{block} score",
    )
    labels = _label_pack(score["label_pack"], block=block)
    item_rows = score["item_exact_utility_rows"]
    if (
        score["schema"] != f"{CONTROLLER_VERSION}_{block}_offline_score_v1"
        or score["block"] != block
        or score["archive_sha256"] != archive["sha256"]
        or score["label_pack_sha256"] != labels["sha256"]
        or score["scoring"] != "local_offline_exact_only"
        or [row["item_commitment_sha256"] for row in labels["rows"]] != archive["item_order"]
        or not isinstance(item_rows, list)
        or len(item_rows) != ITEM_COUNTS[block]
    ):
        raise TatqaP18OfflineFinalizeError(f"{block} score identity drifted")
    _sha(score["postflight_sha256"], field=f"{block} postflight")
    expected_rows: list[dict[str, Any]] = []
    utilities: dict[str, list[Fraction]] = {arm: [] for arm in ARMS[block]}
    families: list[str] = []
    for item, label in zip(archive["item_order"], labels["rows"], strict=True):
        families.append(label["family"])
        arm_values: dict[str, dict[str, int]] = {}
        for arm in ARMS[block]:
            utility = core.item_utility(
                archive["actions"][(item, arm)]["ordered_top5"],
                label["canonical_gold_units"],
            )
            utilities[arm].append(utility)
            arm_values[arm] = _fraction_payload(utility)
        expected_rows.append(
            {
                "item_commitment_sha256": item,
                "family": label["family"],
                "canonical_gold_units": label["canonical_gold_units"],
                "arm_utilities": arm_values,
            }
        )
    if item_rows != expected_rows:
        raise TatqaP18OfflineFinalizeError(f"{block} item utility replay drifted")
    totals = {
        arm: _fraction_payload(sum(utilities[arm], Fraction(0))) for arm in ARMS[block]
    }
    complete = {arm: sum(value == 2 for value in utilities[arm]) for arm in ARMS[block]}
    if score["arm_totals"] != totals or score["arm_complete_counts"] != complete:
        raise TatqaP18OfflineFinalizeError(f"{block} aggregate replay drifted")
    candidate_counts = [
        archive["actions"][(item, "P1")]["P1_minus_P0_features"][-1]
        for item in archive["item_order"]
    ]
    expansion = {
        "item_count": sum(value > 0 for value in candidate_counts),
        "unit_count": sum(candidate_counts),
    }
    if score["candidate_expansion_outside_P0_top5"] != expansion:
        raise TatqaP18OfflineFinalizeError(f"{block} candidate expansion drifted")
    comparisons = [
        _comparison_payload(
            left="E1",
            right=right,
            deltas=tuple(
                left - baseline
                for left, baseline in zip(utilities["E1"], utilities[right], strict=True)
            ),
            families=families,
        )
        for right in ("E0", "RAW", HIPPO_ARM)
    ]
    if score["comparisons"] != comparisons:
        raise TatqaP18OfflineFinalizeError(f"{block} paired comparison replay drifted")
    return {
        "payload": score,
        "sha256": _semantic_hash(score),
        "postflight_sha256": score["postflight_sha256"],
        "label_pack_sha256": labels["sha256"],
        "arm_totals": totals,
        "arm_complete_counts": complete,
        "comparisons": {row["right_arm"]: row for row in comparisons},
    }


def _controller_claims(value: object) -> dict[str, Any]:
    row = _mapping(value, field="controller disposition")
    _keys(
        row,
        {
            "A_form_archive_sha256",
            "A_form_fit_evidence_sha256",
            "A_hold_archive_sha256",
            "A_hold_promoted",
            "A_hold_score_sha256",
            "E1_model",
            "E1_model_sha256",
            "F_search_archive_sha256",
            "M_search_archive_sha256",
            "M_search_labels_released",
            "M_search_score_sha256",
            "M_search_view_released",
            "acquisition_receipt_sha256",
            "api_or_online_evaluator_calls",
            "efficacy",
            "epoch_authorization_sha256",
            "epoch_transition_count",
            "external_network_calls",
            "failure_stage",
            "failure_type_sha256",
            "policy_freeze_sha256",
            "primary_count",
            "primary_evaluated",
            "primary_operator",
            "primary_value",
            "retry_replay_resample_provider_switch",
            "runtime_preflight_sha256",
            "same_source_or_cohort_replay_authorized",
            "schema",
            "status",
        },
        field="controller disposition",
    )
    if (
        row["schema"] != f"{CONTROLLER_VERSION}_terminal_disposition_v1"
        or row["primary_count"] != 1
        or row["primary_operator"] != "AND"
        or row["external_network_calls"] != 0
        or row["api_or_online_evaluator_calls"] != 0
        or row["retry_replay_resample_provider_switch"] != 0
        or row["same_source_or_cohort_replay_authorized"] is not False
    ):
        raise TatqaP18OfflineFinalizeError("controller disposition contract drifted")
    return row


def _formal(value: Mapping[str, Any]) -> dict[str, Any]:
    _keys(
        value,
        {
            "api_or_online_evaluator_calls",
            "controller_disposition",
            "controller_disposition_sha256",
            "external_network_calls",
            "final_disposition_sha256",
            "offline_artifacts",
            "replay_retry_resample_provider_switch",
            "runtime_fingerprint_self_sha256",
            "schema",
            "status",
            "systemd_network_preflight",
            "version",
        },
        field="formal disposition",
    )
    claimed = _sha(value["final_disposition_sha256"], field="final disposition")
    body = dict(value)
    del body["final_disposition_sha256"]
    controller = _controller_claims(value["controller_disposition"])
    network = _mapping(value["systemd_network_preflight"], field="network preflight")
    _keys(
        network,
        {"network_properties", "returncode", "stderr_sha256", "stdout_sha256"},
        field="network preflight",
    )
    if (
        value["schema"] != f"{FORMAL_VERSION}_durable_final_disposition_v1"
        or value["version"] != FORMAL_VERSION
        or _semantic_hash(body) != claimed
        or value["controller_disposition_sha256"] != _semantic_hash(controller)
        or value["status"] != controller["status"]
        or value["external_network_calls"] != 0
        or value["api_or_online_evaluator_calls"] != 0
        or value["replay_retry_resample_provider_switch"] != 0
        or network["network_properties"]
        != ["IPAddressDeny=any", "RestrictAddressFamilies=AF_UNIX"]
        or network["returncode"] != 0
    ):
        raise TatqaP18OfflineFinalizeError("formal disposition binding drifted")
    _sha(network["stdout_sha256"], field="network preflight stdout")
    _sha(network["stderr_sha256"], field="network preflight stderr")
    _sha(value["runtime_fingerprint_self_sha256"], field="runtime fingerprint")
    _sha(controller["acquisition_receipt_sha256"], field="acquisition receipt")
    return controller


def _artifact_registry(value: object) -> dict[str, Any]:
    artifacts = _mapping(value, field="offline artifacts")
    _keys(
        artifacts,
        {
            "A_form_archive",
            "A_form_fit",
            "A_hold_archive",
            "A_hold_score",
            "E1_model",
            "F_search_archive",
            "M_search_archive",
            "M_search_score",
            "epoch_authorization",
            "policy_freeze",
            "runtime_preflight",
        },
        field="offline artifacts",
    )
    return artifacts


def _verify_preflight(value: object, expected_sha: object) -> str:
    payload = _mapping(value, field="runtime preflight")
    if (
        payload.get("schema") != f"{CONTROLLER_VERSION}_runtime_preflight_v1"
        or payload.get("qualified") is not True
        or payload.get("public_synthetic_distinct_rankings") is not True
        or payload.get("external_network_calls") != 0
        or payload.get("api_or_online_evaluator_calls") != 0
        or payload.get("retry_replay_resample_provider_switch") != 0
    ):
        raise TatqaP18OfflineFinalizeError("runtime preflight drifted")
    p0 = _sha(payload.get("public_synthetic_p0_behavior_sha256"), field="canary P0")
    p1 = _sha(payload.get("public_synthetic_p1_behavior_sha256"), field="canary P1")
    result = _semantic_hash(payload)
    if p0 == p1 or result != expected_sha:
        raise TatqaP18OfflineFinalizeError("runtime preflight hash drifted")
    return result


def _epoch(value: object, hold_sha: str, policy_sha: str) -> dict[str, Any]:
    row = _mapping(value, field="epoch authorization")
    expected = {
        "schema": f"{CONTROLLER_VERSION}_epoch_transition_authorization_v1",
        "A_hold_score_sha256": hold_sha,
        "policy_freeze_sha256": policy_sha,
        "previous_evaluator": "E0",
        "active_evaluator": "E1",
        "E0_counterfactual_retained": True,
        "M_search_authorized": True,
        "transition_index": 1,
        "rollback_authorized": False,
    }
    if row != expected:
        raise TatqaP18OfflineFinalizeError("epoch authorization replay drifted")
    return {"payload": row, "sha256": _semantic_hash(row)}


def _positive_all_families(comparison: Mapping[str, Any]) -> bool:
    return _fraction(comparison["net_U"], field="comparison net") > 0 and all(
        _fraction(comparison["family_nets"][family], field=f"{family} net") > 0
        for family in FAMILIES
    )


def _promoted(comparison: Mapping[str, Any]) -> bool:
    return comparison["exact_test"]["promoted"] is True


def _verify_control_root(
    control_root: Path,
    formal_path: Path,
    formal_raw: bytes,
    formal: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    stages: Mapping[str, Mapping[str, Any]],
    fit: Mapping[str, Any],
    policy: Mapping[str, Any],
    scores: Mapping[str, Mapping[str, Any]],
    *,
    full: bool,
) -> dict[str, Any]:
    root = control_root.absolute()
    if root.is_symlink() or not root.is_dir():
        raise TatqaP18OfflineFinalizeError("control root is not a regular directory")
    control_final, control_raw = _strict_json_file(
        root / "formal.disposition.json", field="control final disposition"
    )
    if control_raw != formal_raw or control_final != formal:
        raise TatqaP18OfflineFinalizeError("control-root final disposition drifted")
    try:
        same_file = formal_path.samefile(root / "formal.disposition.json")
    except OSError:
        same_file = False
    if not same_file and formal_raw != control_raw:
        raise TatqaP18OfflineFinalizeError("input/control final file mismatch")

    expected_evidence = ["A_form_fit", "F_search_policy_freeze", "A_hold_score"]
    if full:
        expected_evidence.append("M_search_score")
    runtime_hashes: set[str] = set()
    canary_hashes: set[str] = set()
    evidence_receipts: dict[str, str] = {}
    evidence_values = {
        "A_form_fit": fit,
        "F_search_policy_freeze": policy,
        "A_hold_score": scores["A_hold"],
        **({"M_search_score": scores["M_search"]} if full else {}),
    }
    for name in expected_evidence:
        envelope, _raw = _strict_json_file(
            root / "evidence" / f"{name}.json", field=f"{name} evidence"
        )
        _keys(
            envelope,
            {
                "durable_evidence_receipt_sha256",
                "evidence_sha256",
                "name",
                "payload",
                "production_canary_sha256",
                "runtime_fingerprint_sha256",
                "schema",
            },
            field=f"{name} evidence",
        )
        receipt = _verify_self_hash(envelope, field="durable_evidence_receipt_sha256")
        expected_payload = evidence_values[name]["payload"]
        if (
            envelope["schema"] != f"{ADAPTER_VERSION}_durable_offline_evidence_v1"
            or envelope["name"] != name
            or envelope["payload"] != expected_payload
            or envelope["evidence_sha256"] != evidence_values[name]["sha256"]
            or _semantic_hash(envelope["payload"]) != envelope["evidence_sha256"]
        ):
            raise TatqaP18OfflineFinalizeError(f"{name} evidence chain drifted")
        runtime_hashes.add(_sha(envelope["runtime_fingerprint_sha256"], field="evidence runtime"))
        canary_hashes.add(_sha(envelope["production_canary_sha256"], field="evidence canary"))
        evidence_receipts[name] = receipt

    expected_stages = ["A_form", "F_search", "A_hold"] + (["M_search"] if full else [])
    stage_receipts: dict[str, dict[str, str]] = {}
    for block in expected_stages:
        stage_root = root / "stages" / block
        preparation, _ = _strict_json_file(
            stage_root / "block.preparation.json", field=f"{block} preparation"
        )
        preparation_keys = {
            "actual_model_future_expected_count",
            "actual_model_future_submit_count_before_first_join",
            "all_actual_model_futures_submitted_before_first_join",
            "block",
            "block_view_sha256",
            "hippo_actual_inference_cap",
            "hippo_actual_submitted_count",
            "item_count",
            "items",
            "minilm_raw_compiled_item_count",
            "preparation_receipt_sha256",
            "production_canary_sha256",
            "qwen_hippo_dedicated_inference_executors",
            "qwen_hippo_overlap_observed",
            "retry_replay_resample_provider_switch",
            "runtime_fingerprint_sha256",
            "schema",
            "typed_plan_input_sha256",
            "typed_plan_output_sha256",
            "typed_plan_transport_receipt_sha256",
            "typed_plan_worker_receipt_sha256",
            "typed_plan_worker_pid",
        }
        _keys(preparation, preparation_keys, field=f"{block} preparation")
        prep_sha = _verify_self_hash(
            preparation,
            field="preparation_receipt_sha256",
            schema=f"{ADAPTER_VERSION}_block_preparation_receipt_v1",
        )
        hippo_count = ITEM_COUNTS[block] if block in {"A_hold", "M_search"} else 0
        prep_items = preparation["items"]
        if (
            preparation.get("block") != block
            or preparation.get("item_count") != ITEM_COUNTS[block]
            or preparation["actual_model_future_expected_count"] != 1 + hippo_count
            or preparation["actual_model_future_submit_count_before_first_join"] != 1 + hippo_count
            or preparation["all_actual_model_futures_submitted_before_first_join"] is not True
            or preparation["hippo_actual_inference_cap"] != (8 if hippo_count else 0)
            or preparation["hippo_actual_submitted_count"] != hippo_count
            or preparation["minilm_raw_compiled_item_count"] != ITEM_COUNTS[block]
            or preparation["qwen_hippo_dedicated_inference_executors"] is not bool(hippo_count)
            or preparation["qwen_hippo_overlap_observed"] is not bool(hippo_count)
            or preparation["retry_replay_resample_provider_switch"] != 0
            or not isinstance(prep_items, list)
            or len(prep_items) != ITEM_COUNTS[block]
        ):
            raise TatqaP18OfflineFinalizeError(f"{block} preparation drifted")
        prep_item_keys = {
            "item_commitment_sha256",
            "prompt_receipt_sha256",
            "raw_behavior_sha256",
            "tensor_sha256",
        }
        for expected_item, raw_item in zip(stages[block]["item_order"], prep_items, strict=True):
            prep_item = _mapping(raw_item, field=f"{block} preparation item")
            _keys(prep_item, prep_item_keys, field=f"{block} preparation item")
            if prep_item["item_commitment_sha256"] != expected_item:
                raise TatqaP18OfflineFinalizeError(f"{block} preparation item order drifted")
            for field in prep_item_keys:
                _sha(prep_item[field], field=f"{block} preparation {field}")
        for field in (
            "block_view_sha256",
            "production_canary_sha256",
            "runtime_fingerprint_sha256",
            "typed_plan_input_sha256",
            "typed_plan_output_sha256",
            "typed_plan_transport_receipt_sha256",
            "typed_plan_worker_receipt_sha256",
        ):
            _sha(preparation[field], field=f"{block} preparation {field}")
        if (
            isinstance(preparation["typed_plan_worker_pid"], bool)
            or not isinstance(preparation["typed_plan_worker_pid"], int)
            or preparation["typed_plan_worker_pid"] <= 1
        ):
            raise TatqaP18OfflineFinalizeError(f"{block} preparation worker PID drifted")
        inference, _ = _strict_json_file(
            stage_root / "preparation.inference.json", field=f"{block} inference"
        )
        inference_sha = _verify_self_hash(
            inference,
            field="preparation_inference_receipt_sha256",
            schema=f"{ADAPTER_VERSION}_preparation_inference_receipt_v1",
        )
        if inference != stages[block]["payload"]["actual_inference_preparation"]:
            raise TatqaP18OfflineFinalizeError(f"{block} inference/archive drifted")
        archive_envelope, _ = _strict_json_file(
            stage_root / "action.archive.json", field=f"{block} action archive"
        )
        _keys(
            archive_envelope,
            {
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
            },
            field=f"{block} action archive",
        )
        archive_receipt = _verify_self_hash(
            archive_envelope, field="durable_archive_receipt_sha256"
        )
        postflight_envelope, _ = _strict_json_file(
            stage_root / "runtime.postflight.json", field=f"{block} postflight"
        )
        _keys(
            postflight_envelope,
            {
                "block",
                "durable_postflight_receipt_sha256",
                "inference_executors_closed_after_terminal_validation",
                "postflight",
                "postflight_sha256",
                "preparation_inference_receipt_sha256",
                "production_canary_sha256",
                "runtime_fingerprint_sha256",
                "schema",
                "transport_receipt_aggregate_sha256",
                "transport_receipts",
                "transport_receipt_sha256s",
                "worker_pids",
            },
            field=f"{block} postflight",
        )
        postflight_receipt = _verify_self_hash(
            postflight_envelope, field="durable_postflight_receipt_sha256"
        )
        postflight_payload = _mapping(
            postflight_envelope.get("postflight"), field=f"{block} postflight payload"
        )
        postflight_sha = _semantic_hash(postflight_payload)
        expected_postflight_payload = {
            "schema": f"{CONTROLLER_VERSION}_runtime_postflight_v1",
            "block": block,
            "archive_sha256": stages[block]["sha256"],
            "runtime_ok": True,
            "external_network_calls": 0,
            "api_or_online_evaluator_calls": 0,
            "retry_replay_resample_provider_switch": 0,
            "controller_or_worker_source_reads": 0,
            "controller_or_worker_label_reads": 0,
            "maximum_cpu_threads_per_hippo_process": (2 if hippo_count else 0),
        }
        expected_postflight = (
            fit["postflight_sha256"]
            if block == "A_form"
            else policy["postflight_sha256"]
            if block == "F_search"
            else scores[block]["postflight_sha256"]
        )
        shared_fields = (
            "production_canary_sha256",
            "runtime_fingerprint_sha256",
            "transport_receipt_aggregate_sha256",
            "transport_receipts",
            "transport_receipt_sha256s",
            "worker_pids",
        )
        if (
            archive_envelope.get("schema") != f"{ADAPTER_VERSION}_durable_action_archive_v1"
            or archive_envelope.get("block") != block
            or archive_envelope.get("archive") != stages[block]["payload"]
            or archive_envelope.get("archive_sha256") != stages[block]["sha256"]
            or archive_envelope.get("block_preparation_receipt_sha256") != prep_sha
            or archive_envelope.get("preparation_inference_receipt_sha256") != inference_sha
            or preparation["production_canary_sha256"] != archive_envelope.get("production_canary_sha256")
            or preparation["runtime_fingerprint_sha256"] != archive_envelope.get("runtime_fingerprint_sha256")
            or preparation["typed_plan_transport_receipt_sha256"]
            != inference["qwen_transport_receipt_sha256"]
            or preparation["typed_plan_input_sha256"]
            != inference["qwen_transport_receipt"]["input_sha256"]
            or preparation["typed_plan_output_sha256"]
            != inference["qwen_transport_receipt"]["output_sha256"]
            or preparation["typed_plan_worker_receipt_sha256"]
            != inference["qwen_worker_receipt_sha256"]
            or preparation["typed_plan_worker_pid"] != inference["qwen_worker_pid"]
            or archive_envelope.get("inference_executors_closed_after_terminal_validation") is not True
            or postflight_envelope.get("inference_executors_closed_after_terminal_validation") is not True
            or postflight_envelope.get("schema") != f"{ADAPTER_VERSION}_durable_runtime_postflight_v1"
            or postflight_envelope.get("block") != block
            or postflight_envelope.get("postflight_sha256") != postflight_sha
            or postflight_payload != expected_postflight_payload
            or postflight_sha != expected_postflight
            or postflight_payload.get("archive_sha256") != stages[block]["sha256"]
            or postflight_envelope.get("preparation_inference_receipt_sha256") != inference_sha
            or any(archive_envelope.get(field) != postflight_envelope.get(field) for field in shared_fields)
        ):
            raise TatqaP18OfflineFinalizeError(f"{block} durable stage chain drifted")
        transport_receipts = archive_envelope["transport_receipts"]
        transport_hashes = archive_envelope["transport_receipt_sha256s"]
        worker_pids = archive_envelope["worker_pids"]
        hippo_worker_hashes = archive_envelope["hippo_worker_receipt_sha256s"]
        if (
            not isinstance(transport_hashes, list)
            or len(transport_hashes) != 1 + hippo_count
            or any(not isinstance(row, str) or _SHA256.fullmatch(row) is None for row in transport_hashes)
            or not isinstance(worker_pids, list)
            or len(worker_pids) != len(transport_hashes)
            or any(isinstance(row, bool) or not isinstance(row, int) or row <= 1 for row in worker_pids)
            or not isinstance(hippo_worker_hashes, list)
            or len(hippo_worker_hashes) != hippo_count
            or any(not isinstance(row, str) or _SHA256.fullmatch(row) is None for row in hippo_worker_hashes)
            or transport_hashes
            != [
                inference["qwen_transport_receipt_sha256"],
                *inference["hippo_transport_receipt_sha256s"],
            ]
            or worker_pids
            != [inference["qwen_worker_pid"], *inference["hippo_worker_pids"]]
            or hippo_worker_hashes != inference["hippo_worker_receipt_sha256s"]
            or not isinstance(transport_receipts, list)
            or transport_receipts != stages[block]["transport_receipts"]
            or [_semantic_hash(row) for row in transport_receipts] != transport_hashes
            or archive_envelope["transport_receipt_aggregate_sha256"]
            != _semantic_hash(
                {
                    "transport_receipts": transport_receipts,
                    "transport_receipt_sha256s": transport_hashes,
                    "worker_pids": worker_pids,
                }
            )
        ):
            raise TatqaP18OfflineFinalizeError(f"{block} durable transport chain drifted")
        runtime_hashes.add(_sha(archive_envelope["runtime_fingerprint_sha256"], field="stage runtime"))
        canary_hashes.add(_sha(archive_envelope["production_canary_sha256"], field="stage canary"))
        stage_receipts[block] = {
            "preparation_receipt_sha256": prep_sha,
            "inference_receipt_sha256": inference_sha,
            "archive_receipt_sha256": archive_receipt,
            "postflight_receipt_sha256": postflight_receipt,
        }
    if runtime_hashes != {formal["runtime_fingerprint_self_sha256"]} or len(canary_hashes) != 1:
        raise TatqaP18OfflineFinalizeError("durable runtime/canary chain drifted")
    return {
        "verified": True,
        "evidence_receipts": evidence_receipts,
        "stage_receipts": stage_receipts,
        "runtime_fingerprint_self_sha256": next(iter(runtime_hashes)),
        "production_canary_self_sha256": next(iter(canary_hashes)),
    }


def recompute_final_disposition(
    formal_disposition_path: str | Path,
    *,
    control_root: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute the complete frozen terminal result without writing output."""

    formal_path = Path(formal_disposition_path).absolute()
    formal, formal_raw = _strict_json_file(formal_path, field="formal disposition")
    controller = _formal(formal)
    status = controller["status"]
    if status not in {"valid_nonpromotion", "valid_primary_true", "valid_primary_false"}:
        raise TatqaP18OfflineFinalizeError(
            "implementation-invalid disposition is not independently scoreable"
        )
    artifacts = _artifact_registry(formal["offline_artifacts"])
    _verify_preflight(artifacts["runtime_preflight"], controller["runtime_preflight_sha256"])
    a_form = _archive(artifacts["A_form_archive"], block="A_form", model=None)
    fit = _fit(artifacts["A_form_fit"], a_form)
    if artifacts["E1_model"] != fit["model_payload"]:
        raise TatqaP18OfflineFinalizeError("duplicated E1 model drifted")
    f_search = _archive(artifacts["F_search_archive"], block="F_search", model=None)
    policy = _policy_freeze(artifacts["policy_freeze"], f_search, fit)
    a_hold = _archive(artifacts["A_hold_archive"], block="A_hold", model=fit["model"])
    hold_score = _score(artifacts["A_hold_score"], block="A_hold", archive=a_hold)
    hold_promoted = _promoted(hold_score["comparisons"]["E0"])
    full = status in {"valid_primary_true", "valid_primary_false"}
    stages: dict[str, Mapping[str, Any]] = {
        "A_form": a_form,
        "F_search": f_search,
        "A_hold": a_hold,
    }
    scores: dict[str, Mapping[str, Any]] = {"A_hold": hold_score}
    epoch: dict[str, Any] | None = None
    primary = False
    if full:
        if not hold_promoted:
            raise TatqaP18OfflineFinalizeError("M_search exists without A_hold promotion")
        epoch = _epoch(artifacts["epoch_authorization"], hold_score["sha256"], policy["sha256"])
        m_search = _archive(artifacts["M_search_archive"], block="M_search", model=fit["model"])
        m_score = _score(artifacts["M_search_score"], block="M_search", archive=m_search)
        stages["M_search"] = m_search
        scores["M_search"] = m_score
        primary = all(
            (
                hold_promoted,
                _promoted(m_score["comparisons"]["E0"]),
                _positive_all_families(m_score["comparisons"]["RAW"]),
                _positive_all_families(m_score["comparisons"][HIPPO_ARM]),
            )
        )
    elif (
        hold_promoted
        or artifacts["epoch_authorization"] is not None
        or artifacts["M_search_archive"] is not None
        or artifacts["M_search_score"] is not None
    ):
        raise TatqaP18OfflineFinalizeError("nonpromotion terminal contains M_search artifacts")

    expected_status = "valid_nonpromotion" if not hold_promoted else (
        "valid_primary_true" if primary else "valid_primary_false"
    )
    expected_claims = {
        "A_form_fit_evidence_sha256": fit["sha256"],
        "A_form_archive_sha256": a_form["sha256"],
        "F_search_archive_sha256": f_search["sha256"],
        "policy_freeze_sha256": policy["sha256"],
        "A_hold_archive_sha256": a_hold["sha256"],
        "A_hold_score_sha256": hold_score["sha256"],
        "E1_model_sha256": fit["model_sha256"],
        "E1_model": fit["model_payload"],
        "epoch_authorization_sha256": None if epoch is None else epoch["sha256"],
        "M_search_archive_sha256": None if not full else stages["M_search"]["sha256"],
        "M_search_score_sha256": None if not full else scores["M_search"]["sha256"],
        "status": expected_status,
        "primary_evaluated": True,
        "primary_value": primary,
        "efficacy": "true" if primary else "false",
        "A_hold_promoted": hold_promoted,
        "epoch_transition_count": 1 if full else 0,
        "M_search_view_released": full,
        "M_search_labels_released": full,
        "failure_stage": None,
        "failure_type_sha256": None,
    }
    if any(controller.get(key) != value for key, value in expected_claims.items()):
        raise TatqaP18OfflineFinalizeError("controller/recomputed terminal claims drifted")

    control_audit = {"verified": False}
    if control_root is not None:
        control_audit = _verify_control_root(
            Path(control_root),
            formal_path,
            formal_raw,
            formal,
            artifacts,
            stages,
            fit,
            policy,
            scores,
            full=full,
        )
    return {
        "schema": f"{VERSION}_report_v1",
        "version": VERSION,
        "verification_status": "verified_exact_offline_recomputation",
        "formal_disposition_file_sha256": hashlib.sha256(formal_raw).hexdigest(),
        "formal_disposition_self_sha256": formal["final_disposition_sha256"],
        "controller_disposition_sha256": formal["controller_disposition_sha256"],
        "recomputed_status": expected_status,
        "recomputed_A_hold_promoted": hold_promoted,
        "recomputed_primary_value": primary,
        "single_joint_primary": {
            "operator": "AND",
            "A_hold_E1_over_E0_promoted": hold_promoted,
            "M_search_E1_over_E0_promoted": (
                False if not full else _promoted(scores["M_search"]["comparisons"]["E0"])
            ),
            "M_search_E1_over_RAW_aggregate_and_all_families_positive": (
                False if not full else _positive_all_families(scores["M_search"]["comparisons"]["RAW"])
            ),
            "M_search_E1_over_official_HippoRAG_aggregate_and_all_families_positive": (
                False if not full else _positive_all_families(scores["M_search"]["comparisons"][HIPPO_ARM])
            ),
            "value": primary,
        },
        "recomputed_hashes": {
            "A_form_archive_sha256": a_form["sha256"],
            "A_form_fit_evidence_sha256": fit["sha256"],
            "E1_model_sha256": fit["model_sha256"],
            "F_search_archive_sha256": f_search["sha256"],
            "policy_freeze_sha256": policy["sha256"],
            "A_hold_archive_sha256": a_hold["sha256"],
            "A_hold_score_sha256": hold_score["sha256"],
            "epoch_authorization_sha256": None if epoch is None else epoch["sha256"],
            "M_search_archive_sha256": None if not full else stages["M_search"]["sha256"],
            "M_search_score_sha256": None if not full else scores["M_search"]["sha256"],
        },
        "stage_action_accounting": {
            block: {
                "logical_action_count": stages[block]["logical_action_count"],
                "submitted_work_count": stages[block]["submitted_work_count"],
            }
            for block in stages
        },
        "runtime_receipt_recomputation": {
            block: {
                "transport_receipt_count": len(stages[block]["transport_receipts"]),
                "qwen_transport_receipt_sha256": stages[block][
                    "transport_receipt_sha256s"
                ][0],
                "hippo_transport_receipt_count": len(
                    stages[block]["hippo_transport_receipts"]
                ),
                "overlap_witness_item_commitments": stages[block][
                    "overlap_witness_item_commitments"
                ],
                "maximum_configured_torch_intraop_threads": max(
                    (
                        row["configured_torch_intraop_threads"]
                        for row in stages[block]["hippo_transport_receipts"]
                    ),
                    default=0,
                ),
                "maximum_configured_torch_interop_threads": max(
                    (
                        row["configured_torch_interop_threads"]
                        for row in stages[block]["hippo_transport_receipts"]
                    ),
                    default=0,
                ),
                "maximum_observed_process_thread_peak": max(
                    (
                        row["observed_process_thread_peak"]
                        for row in stages[block]["hippo_transport_receipts"]
                    ),
                    default=0,
                ),
                "full_transport_receipt_aggregate_sha256": _semantic_hash(
                    {
                        "transport_receipts": stages[block]["transport_receipts"],
                        "transport_receipt_sha256s": stages[block][
                            "transport_receipt_sha256s"
                        ],
                        "worker_pids": stages[block]["worker_pids"],
                    }
                ),
            }
            for block in stages
        },
        "score_recomputation": {
            block: {
                "arm_totals": scores[block]["arm_totals"],
                "arm_complete_counts": scores[block]["arm_complete_counts"],
                "comparison_sha256s": {
                    right: _semantic_hash(comparison)
                    for right, comparison in scores[block]["comparisons"].items()
                },
            }
            for block in scores
        },
        "control_root_audit": control_audit,
        "external_network_calls": 0,
        "api_or_online_evaluator_calls": 0,
        "retry_replay_resample_provider_switch": 0,
        "formal_source_files_opened": 0,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = _canonical_bytes(value)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except OSError as exc:
        raise TatqaP18OfflineFinalizeError("exclusive offline report write failed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if _read_regular(path, field="offline final report") != raw:
        raise TatqaP18OfflineFinalizeError("offline final report reopen drifted")


def finalize_offline(
    formal_disposition_path: str | Path,
    output_path: str | Path,
    *,
    control_root: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute once and exclusively persist a canonical self-hashed report."""

    body = recompute_final_disposition(
        formal_disposition_path, control_root=control_root
    )
    report = {**body, "self_sha256": _semantic_hash(body)}
    output = Path(output_path).absolute()
    if output.exists() or output.is_symlink():
        raise TatqaP18OfflineFinalizeError("offline final report already exists")
    _write_exclusive(output, report)
    reopened, _raw = _strict_json_file(output, field="offline final report")
    if reopened != report or _verify_self_hash(reopened, field="self_sha256") != report["self_sha256"]:
        raise TatqaP18OfflineFinalizeError("offline final report self-check drifted")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("formal_disposition")
    parser.add_argument("output")
    parser.add_argument("--control-root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    finalize_offline(
        arguments.formal_disposition,
        arguments.output,
        control_root=arguments.control_root,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TatqaP18OfflineFinalizeError",
    "finalize_offline",
    "main",
    "recompute_final_disposition",
]
