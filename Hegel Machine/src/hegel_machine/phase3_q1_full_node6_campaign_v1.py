"""Pure wire and state transitions for the full-node6 diagnostic campaign."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from typing import Any, Final, Mapping, NoReturn


CONTRACT_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q1-full-node6-campaign-contract/3"
)
CONTRACT_ID: Final = "hegel-phase3a-q1-full-node6-campaign-successor-v3"
CAPSULE_SCHEMA_VERSION: Final = "hegel-phase3a-q1-full-node6-capsule/1"
PENDING_SCHEMA_VERSION: Final = "hegel-phase3a-q1-full-node6-pending/1"
LEDGER_SCHEMA_VERSION: Final = "hegel-phase3a-q1-full-node6-ledger-record/2"
OUTCOME_SCHEMA_VERSION: Final = "hegel-phase3a-q1-full-node6-outcome/1"

PREFLIGHT_ADMITTED: Final = "FULL_NODE6_PREFLIGHT_ADMITTED_NOT_CONSUMED"
PENDING_NOT_CONSUMED: Final = "PENDING_FULL_NODE6_NOT_CONSUMED"
CONSUMED: Final = "FULL_NODE6_AUTHORIZATION_CONSUMED"
SATURATED: Final = "PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY"
GUARD_HIT: Final = "PREFLIGHT_CAPACITY_GUARD_HIT"
RESOURCE_INFEASIBLE: Final = "PREFLIGHT_RESOURCE_ENVELOPE_INFEASIBLE"
FAILED_TERMINAL: Final = "FULL_NODE6_DIAGNOSTIC_FAILED_TERMINAL_NO_RESUME"
OUTCOME_UNKNOWN_TERMINAL: Final = (
    "FULL_NODE6_CONSUMED_OUTCOME_UNKNOWN_TERMINAL_NO_RESUME"
)

OWNER_AUTHORIZATION_TEXT: Final = (
    "授权 Q1 full-node6 capacity-preflight diagnostic "
    "successor-v3 distinct-outer actual；确认 successor-v2 measurement outcome "
    "UNKNOWN、terminal no-resume，且 prior unknown conservatively counts as "
    "measurement；另行授权 exact v3 capsule-only additional fresh "
    "remeasurement；禁止复用 successor-v2 work product"
)
OWNER_AUTHORIZATION_BODY: Final = {
    "schema_version": "hegel-phase3a-q1-full-node6-owner-authorization/3",
    "authority_role": "PROJECT_OWNER",
    "authorization_text": OWNER_AUTHORIZATION_TEXT,
    "superseded_v2_resume_allowed": False,
    "distinct_fresh_successor_v3_logical_campaign_only": True,
    "future_exact_user_authorization_required": True,
    "prior_measurement_outcome_unknown_acknowledged": True,
    "prior_unknown_conservatively_counts_as_measurement": True,
    "new_measurement_budget_scope": "V3_CAPSULE_ONLY",
    "new_measurement_is_explicitly_additional": True,
    "v2_work_product_reuse_allowed": False,
    "logical_campaign_count": 1,
    "maximum_supervisor_popen_count": 1,
    "maximum_quotient_measurement_count_per_endpoint_signature": 1,
    "same_capsule_preconsumption_pending_replay_allowed": True,
    "pending_replay_is_campaign_execution": False,
    "pending_replay_consumes_authorization": False,
    "measurement_outcome_unknown_terminal_no_resume": True,
    "q1_start_allowed": False,
    "network_allowed": False,
    "adoption_allowed": False,
    "promotion_allowed": False,
    "current_pointer_write_allowed": False,
}

DOMAINS: Final = {
    "authorization": b"HEGEL/Q1/FULL_NODE6/AUTHORIZATION/SUCCESSOR_V3/V1\x00",
    "capsule": b"HEGEL/Q1/FULL_NODE6/CAPSULE/V1\x00",
    "contract": b"HEGEL/Q1/FULL_NODE6/CONTRACT/SUCCESSOR_V3/V1\x00",
    "ledger": b"HEGEL/Q1/FULL_NODE6/LEDGER/V1\x00",
    "outcome": b"HEGEL/Q1/FULL_NODE6/OUTCOME/V1\x00",
    "pending": b"HEGEL/Q1/FULL_NODE6/PENDING/V1\x00",
}

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class FullNode6CampaignValidationError(ValueError):
    pass


def _fail(detail: str) -> NoReturn:
    raise FullNode6CampaignValidationError(detail)


def canonical_json_bytes_v1(value: object) -> bytes:
    def finite(item: object) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            _fail("non-finite JSON is forbidden")
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                _fail("JSON object keys must be strings")
            for child in item.values():
                finite(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                finite(child)

    finite(value)
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        _fail(f"canonical JSON rejected: {error}")


def domain_digest_v1(domain: bytes, value: object) -> str:
    if type(domain) is not bytes or not domain:
        _fail("digest domain must be nonempty bytes")
    return "sha256:" + sha256(domain + canonical_json_bytes_v1(value)).hexdigest()


OWNER_AUTHORIZATION_DIGEST: Final = domain_digest_v1(
    DOMAINS["authorization"], OWNER_AUTHORIZATION_BODY
)
OWNER_AUTHORIZATION: Final = {
    **OWNER_AUTHORIZATION_BODY,
    "authorization_digest": OWNER_AUTHORIZATION_DIGEST,
}


def _object(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} must be an object")
    return deepcopy(value)


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        _fail(f"{label} must be tagged sha256")
    return value


def _commit(value: object, label: str) -> str:
    if type(value) is not str or _COMMIT.fullmatch(value) is None:
        _fail(f"{label} must be a full lowercase commit")
    return value


def validate_contract_v1(value: object) -> dict[str, Any]:
    contract = _object(value, "contract")
    required = {
        "schema_version",
        "contract_id",
        "status",
        "contract_digest",
        "authorization",
        "source_lineage",
        "predecessor",
        "superseded_campaign",
        "target",
        "fixed_paths",
        "preconsumption_gate_order",
        "execution_policy",
        "state_machine",
        "authority",
    }
    if set(contract) != required:
        _fail("contract field registry differs")
    observed = _digest(contract["contract_digest"], "contract_digest")
    unsigned = deepcopy(contract)
    unsigned.pop("contract_digest")
    if (
        contract["schema_version"] != CONTRACT_SCHEMA_VERSION
        or contract["contract_id"] != CONTRACT_ID
        or contract["status"] != "IMPLEMENTED_NOT_EXECUTED"
        or contract["authorization"] != OWNER_AUTHORIZATION
        or domain_digest_v1(DOMAINS["contract"], unsigned) != observed
    ):
        _fail("contract header, authorization or digest differs")
    lineage = _object(contract["source_lineage"], "source_lineage")
    _commit(lineage.get("required_ancestor_commit"), "required_ancestor_commit")
    _commit(lineage.get("required_parent_commit"), "required_parent_commit")
    if (
        lineage.get("project_root")
        != "/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine"
        or lineage.get("required_ancestor_commit")
        != "33dc9cbb7d753f83650169d6d6b9caa508927764"
        or lineage.get("required_parent_commit")
        != "1321eb3be3c28f823fba04b947bb96ee16d35f6c"
        or lineage.get("clean_head_required") is not True
        or lineage.get("additive_successor_only") is not True
        or type(lineage.get("additive_paths_in_exact_order")) is not list
        or not lineage["additive_paths_in_exact_order"]
        or "docs/Hegel_Machine_Current_State_Reconciliation_v4.md"
        not in lineage["additive_paths_in_exact_order"]
    ):
        _fail("source lineage differs")
    predecessor = _object(contract["predecessor"], "predecessor")
    for name in (
        "capsule_digest",
        "terminal_ledger_record_digest",
        "artifact_sha256",
    ):
        _digest(predecessor.get(name), f"predecessor.{name}")
    if (
        predecessor.get("terminal_status")
        != "ATTEMPT2_TARGET_D_V2_QUALIFIED_Q1_NOT_RUN"
        or predecessor.get("qualification_count") != 20
        or predecessor.get("qualification_mask") != 0xFFFFF
        or predecessor.get("q1_state") != "NOT_RUN"
    ):
        _fail("predecessor terminal boundary differs")
    superseded = _object(
        contract["superseded_campaign"],
        "superseded_campaign",
    )
    for name in (
        "capsule_digest",
        "consumed_ledger_record_digest",
        "consumed_ledger_file_sha256",
        "terminal_ledger_record_digest",
        "terminal_ledger_file_sha256",
        "outcome_digest",
        "trajectory_event_log_file_sha256",
        "trajectory_terminal_file_sha256",
    ):
        _digest(superseded.get(name), f"superseded_campaign.{name}")
    _commit(superseded.get("source_commit"), "superseded_campaign.source_commit")
    if (
        superseded.get("terminal_status") != FAILED_TERMINAL
        or superseded.get("capsule_digest")
        != "sha256:8eec9c3afaaea443553cd886c84879f465e0b15f406c7f88a6bfef512867cdde"
        or superseded.get("source_commit")
        != "1321eb3be3c28f823fba04b947bb96ee16d35f6c"
        or superseded.get("consumed_ledger_record_digest")
        != "sha256:4643ee4c0740f5f374eb32d6dfaf9c24e0338d47979666a246c1be920e7dc6e4"
        or superseded.get("consumed_ledger_file_sha256")
        != "sha256:d43522bc1df643dc1e56cf0a30826b44e5f758e14dfeab9d58cdf07d59e724a2"
        or superseded.get("terminal_ledger_record_digest")
        != "sha256:fabe961525bfc101907b099ca11a8c77b404e2e27253f8f6a615af68dc7b5e81"
        or superseded.get("terminal_ledger_file_sha256")
        != "sha256:f85afc4790dd25da050ba91f36044429030cbae5aaab5f4275ca96603fa661bd"
        or superseded.get("outcome_digest")
        != "sha256:b5bfd879546b15bfe9a19646ab87e8fd5a886dcd4ed99122d4d69644dc43f7cd"
        or superseded.get("trajectory_event_log_file_sha256")
        != "sha256:bd2959d92d39c4c983c6797f3e0dfdf1f1e23be256a0d61a6cfaf967aafe40cd"
        or superseded.get("trajectory_terminal_file_sha256")
        != "sha256:d5f9a867dbc65fb673e41195498334f96551bc78ecd4a2665b413b4ad089fbf8"
        or superseded.get("error_code") != "FAIL_FULL_NODE6_SUPERVISOR_RESOURCE"
        or superseded.get("error_detail") != "PYTHON_ENDPOINT isolation differs"
        or superseded.get("trajectory_event_count") != 27
        or superseded.get("measurement_outcome") != "UNKNOWN"
        or superseded.get("measurement_endpoint_start_count") != 1
        or superseded.get("python_endpoint_start_count") != 1
        or superseded.get("rust_endpoint_start_count") != 0
        or superseded.get("trusted_host_replay_start_count") != 0
        or superseded.get("q1_state") != "NOT_RUN"
        or superseded.get("q1_gate_count") != 0
        or superseded.get("q1_gate_mask") != 0
        or superseded.get("retry_or_resume_allowed") is not False
    ):
        _fail("superseded campaign terminal boundary differs")
    target = _object(contract["target"], "target")
    if (
        target.get("action_id") != "full-node6-capacity-preflight-v1"
        or target.get("technical_contract_relative_path")
        != "config/phase3_q1_full_node6_capacity_preflight_v1.json"
        or target.get("artifact_relative_path")
        != "artifacts/phase3_q1_full_node6_capacity_preflight_v3.json"
        or target.get("allowed_success_status") != SATURATED
        or target.get("maximum_supervisor_popen_count") != 1
        or target.get("maximum_measurements_per_endpoint_signature") != 1
        or target.get("fresh_quotient_rerun_for_projection_forbidden") is not True
    ):
        _fail("target execution boundary differs")
    fixed = _object(contract["fixed_paths"], "fixed_paths")
    if set(fixed) != {
        "project_root",
        "cargo_cache_source",
        "campaign_lock_path",
        "ledger_root",
        "evidence_root",
        "trajectory_root",
        "run_root",
        "probe_root",
        "artifact_path",
        "stdout_path",
        "stderr_path",
    } or any(type(item) is not str or not item for item in fixed.values()):
        _fail("fresh successor-v3 paths differ")
    if contract["preconsumption_gate_order"] != [
        "AUTHORIZATION_TEXT",
        "PATH_AND_FORMAT",
        "SOURCE_LINEAGE_AND_OBJECT_CLOSURE",
        "PREDECESSOR_TERMINAL_AND_ARTIFACT",
        "SUPERSEDED_CAMPAIGN_TERMINAL",
        "TECHNICAL_CONTRACT_AND_FROZEN_SEMANTICS",
        "EXACT_COMMIT_SNAPSHOT_MATERIALIZATION",
        "PYTHON_RUST_OFFLINE_DEPENDENCIES",
        "PINNED_LOCAL_IMAGES_AND_DAEMON",
        "CPU_TOPOLOGY_AND_EXACT_AFFINITY",
        "RUNTIME_ISOLATION_LIVE_GAUNTLET",
        "MEMORY_AND_STORAGE_RESERVATION",
        "CURRENT_FUTURE_FORMAL_AND_PROBE_ROOTS_ABSENT",
    ]:
        _fail("preconsumption gate order differs")
    execution = _object(contract["execution_policy"], "execution_policy")
    if (
        execution.get("logical_campaign_count") != 1
        or execution.get("maximum_supervisor_popen_count") != 1
        or execution.get("supervisor_timeout_seconds") != 1212000
        or execution.get("pending_exit_code") != 75
        or execution.get("all_gates_before_consumption_and_popen") is not True
        or execution.get("pending_requires_zero_ledger_write_and_zero_popen")
        is not True
        or execution.get("same_capsule_pending_reentry_allowed") is not True
        or execution.get("runtime_isolation_live_gauntlet_required_receipt")
        != {
            "probe_command": "/bin/true",
            "quotient_engine_invocation_count": 0,
            "signature_selection_count": 0,
            "production_endpoint_start_count": 0,
            "supervisor_popen_count": 0,
        }
        or execution.get("postconsumption_terminal_no_resume") is not True
        or execution.get("network_allowed") is not False
        or execution.get("q1_state") != "NOT_RUN"
    ):
        _fail("execution policy differs")
    authority = _object(contract["authority"], "authority")
    if authority != {
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_formal_roots": None,
        "q1_output_slots": [None] * 8,
        "q1_receipt": None,
        "q2_state": "NOT_RUN",
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
        "m3_formal_roots": None,
        "outside_certificate_issued": False,
        "active_transition_allowed": False,
        "adoption_allowed": False,
        "promotion_allowed": False,
        "current_pointer_write_allowed": False,
        "external_scientific_claim_allowed": False,
    }:
        _fail("authority boundary differs")
    return contract


def build_capsule_v1(
    contract: Mapping[str, Any],
    *,
    source_commit: str,
    source_tree: str,
    source_closure_digest: str,
    static_gate_receipts: Mapping[str, object],
) -> dict[str, Any]:
    validated = validate_contract_v1(contract)
    _commit(source_commit, "source_commit")
    _commit(source_tree, "source_tree")
    _digest(source_closure_digest, "source_closure_digest")
    if type(static_gate_receipts) is not dict or not static_gate_receipts:
        _fail("static_gate_receipts must be one nonempty object")
    body: dict[str, Any] = {
        "schema_version": CAPSULE_SCHEMA_VERSION,
        "status": PREFLIGHT_ADMITTED,
        "contract_digest": validated["contract_digest"],
        "authorization": deepcopy(OWNER_AUTHORIZATION),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "source_closure_digest": source_closure_digest,
        "static_gate_receipts": deepcopy(dict(static_gate_receipts)),
        "maximum_supervisor_popen_count": 1,
        "maximum_measurements_per_endpoint_signature": 1,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
        "capsule_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("capsule_digest")
    body["capsule_digest"] = domain_digest_v1(DOMAINS["capsule"], unsigned)
    return body


def validate_capsule_v1(
    value: object,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    capsule = _object(value, "capsule")
    validated = validate_contract_v1(contract)
    observed = _digest(capsule.get("capsule_digest"), "capsule_digest")
    unsigned = deepcopy(capsule)
    unsigned.pop("capsule_digest", None)
    if (
        capsule.get("schema_version") != CAPSULE_SCHEMA_VERSION
        or capsule.get("status") != PREFLIGHT_ADMITTED
        or capsule.get("contract_digest") != validated["contract_digest"]
        or capsule.get("authorization") != OWNER_AUTHORIZATION
        or capsule.get("maximum_supervisor_popen_count") != 1
        or capsule.get("maximum_measurements_per_endpoint_signature") != 1
        or capsule.get("q1_state") != "NOT_RUN"
        or capsule.get("q1_gate_count") != 0
        or capsule.get("q1_gate_mask") != 0
        or capsule.get("q1_output_slots") != [None] * 8
        or domain_digest_v1(DOMAINS["capsule"], unsigned) != observed
    ):
        _fail("capsule differs")
    _commit(capsule.get("source_commit"), "capsule.source_commit")
    _commit(capsule.get("source_tree"), "capsule.source_tree")
    _digest(capsule.get("source_closure_digest"), "capsule.source_closure_digest")
    return capsule


def build_pending_v1(
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
    *,
    phase: str,
    condition: str,
    detail: str,
) -> dict[str, Any]:
    validated = validate_contract_v1(contract)
    cap = validate_capsule_v1(capsule, validated)
    if any(type(value) is not str or not value for value in (phase, condition, detail)):
        _fail("pending phase, condition and detail must be nonempty")
    body = {
        "schema_version": PENDING_SCHEMA_VERSION,
        "status": PENDING_NOT_CONSUMED,
        "contract_digest": validated["contract_digest"],
        "capsule_digest": cap["capsule_digest"],
        "authorization_digest": OWNER_AUTHORIZATION_DIGEST,
        "phase": phase,
        "condition": condition,
        "detail": detail,
        "campaign_execution_count": 0,
        "authorization_consumption_count": 0,
        "supervisor_popen_count": 0,
        "retry_transition_count": 0,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
        "pending_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("pending_digest")
    body["pending_digest"] = domain_digest_v1(DOMAINS["pending"], unsigned)
    return body


def build_ledger_record_v1(
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
    *,
    sequence: int,
    record_type: str,
    status: str,
    previous_record_digest_or_null: str | None,
    body: Mapping[str, object],
) -> dict[str, Any]:
    validated = validate_contract_v1(contract)
    cap = validate_capsule_v1(capsule, validated)
    if type(sequence) is not int or sequence not in (1, 2):
        _fail("ledger sequence differs")
    if (sequence, record_type) not in ((1, "CONSUMED"), (2, "TERMINAL")):
        _fail("ledger record type differs")
    if sequence == 1 and previous_record_digest_or_null is not None:
        _fail("consumed ledger predecessor must be null")
    if sequence == 2:
        _digest(previous_record_digest_or_null, "previous_record_digest_or_null")
    value = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "contract_digest": validated["contract_digest"],
        "capsule_digest": cap["capsule_digest"],
        "authorization_digest": OWNER_AUTHORIZATION_DIGEST,
        "sequence": sequence,
        "record_type": record_type,
        "status": status,
        "previous_record_digest_or_null": previous_record_digest_or_null,
        "body": deepcopy(dict(body)),
        "record_digest": None,
    }
    unsigned = deepcopy(value)
    unsigned.pop("record_digest")
    value["record_digest"] = domain_digest_v1(DOMAINS["ledger"], unsigned)
    return value


@dataclass(frozen=True, slots=True)
class CampaignDecisionV1:
    status: str
    success: bool
    retry_or_resume_allowed: bool


def decide_supervisor_return_v1(
    *,
    return_code: int,
    terminal_status_or_null: str | None,
    replay_passed: bool,
) -> CampaignDecisionV1:
    if type(return_code) is not int or type(replay_passed) is not bool:
        _fail("supervisor decision inputs differ")
    if (
        return_code == 0
        and terminal_status_or_null == SATURATED
        and replay_passed
    ):
        return CampaignDecisionV1(SATURATED, True, False)
    if (
        return_code == 0
        and terminal_status_or_null == GUARD_HIT
        and replay_passed
    ):
        return CampaignDecisionV1(GUARD_HIT, False, False)
    if (
        return_code == 3
        and terminal_status_or_null == RESOURCE_INFEASIBLE
        and replay_passed
    ):
        return CampaignDecisionV1(RESOURCE_INFEASIBLE, False, False)
    return CampaignDecisionV1(FAILED_TERMINAL, False, False)


__all__ = [
    "CAPSULE_SCHEMA_VERSION",
    "CONSUMED",
    "CONTRACT_ID",
    "CONTRACT_SCHEMA_VERSION",
    "CampaignDecisionV1",
    "FAILED_TERMINAL",
    "FullNode6CampaignValidationError",
    "GUARD_HIT",
    "LEDGER_SCHEMA_VERSION",
    "OUTCOME_SCHEMA_VERSION",
    "OUTCOME_UNKNOWN_TERMINAL",
    "OWNER_AUTHORIZATION",
    "OWNER_AUTHORIZATION_DIGEST",
    "OWNER_AUTHORIZATION_TEXT",
    "PENDING_NOT_CONSUMED",
    "PREFLIGHT_ADMITTED",
    "RESOURCE_INFEASIBLE",
    "SATURATED",
    "build_capsule_v1",
    "build_ledger_record_v1",
    "build_pending_v1",
    "canonical_json_bytes_v1",
    "decide_supervisor_return_v1",
    "domain_digest_v1",
    "validate_capsule_v1",
    "validate_contract_v1",
]
