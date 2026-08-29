"""Pure wire/state kernel for the formal Q0.5b attempt-2 Target-D campaign.

The module performs no I/O.  The companion tool owns every live gate, the
no-replace ledger and the single supervisor ``Popen``.  Keeping the state
transition here makes the important boundary explicit: PENDING is possible
only before consumption; every state at or after consumption is terminal.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
from typing import Any, Final, Mapping, NoReturn


CONTRACT_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-attempt2-target-d-campaign-contract/2"
)
CONTRACT_ID: Final = "hegel-phase3a-q05b-attempt2-target-d-campaign-v2"
PENDING_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-attempt2-target-d-pending/2"
)
CAPSULE_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-attempt2-target-d-capsule/2"
)
LEDGER_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-attempt2-target-d-ledger-record/2"
)
OUTCOME_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-attempt2-target-d-outcome/2"
)
DECISION_AUDIT_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-attempt2-target-d-decision-audit/2"
)

PENDING_NOT_CONSUMED: Final = "PENDING_ATTEMPT2_TARGET_D_V2_NOT_CONSUMED"
PREFLIGHT_ADMITTED: Final = "ATTEMPT2_TARGET_D_V2_PREFLIGHT_ADMITTED"
CONSUMED: Final = "ATTEMPT2_TARGET_D_V2_AUTHORIZATION_CONSUMED"
QUALIFIED: Final = "ATTEMPT2_TARGET_D_V2_QUALIFIED_Q1_NOT_RUN"
FAILED_TERMINAL: Final = "ATTEMPT2_TARGET_D_V2_FAILED_TERMINAL_NO_RETRY"
OUTCOME_UNKNOWN_TERMINAL: Final = (
    "ATTEMPT2_TARGET_D_V2_CONSUMED_OUTCOME_UNKNOWN_TERMINAL_NO_RESUME"
)

OWNER_AUTHORIZATION_TEXT: Final = "授权 Q0.5b attempt-2 actual"
OWNER_AUTHORIZATION_BODY: Final = {
    "schema_version": "hegel-phase3a-q05b-attempt2-owner-authorization/2",
    "authority_role": "PROJECT_OWNER",
    "authorization_text": OWNER_AUTHORIZATION_TEXT,
    "authorized_attempt_ordinal": 2,
    "logical_attempt_count": 1,
    "maximum_actual_supervisor_popen_count": 1,
    "preconsumption_pending_reentry_is_retry": False,
    "postconsumption_retry_or_resume_allowed": False,
    "shadow_success_is_formal_actual_evidence": False,
}

DOMAINS: Final = {
    "authorization": b"HEGEL/Q05B/ATTEMPT2/TARGET_D/AUTHORIZATION/V2\x00",
    "capsule": b"HEGEL/Q05B/ATTEMPT2/TARGET_D/CAPSULE/V2\x00",
    "contract": b"HEGEL/Q05B/ATTEMPT2/TARGET_D/CONTRACT/V2\x00",
    "decision_audit": b"HEGEL/Q05B/ATTEMPT2/TARGET_D/DECISION_AUDIT/V2\x00",
    "ledger": b"HEGEL/Q05B/ATTEMPT2/TARGET_D/LEDGER/V2\x00",
    "outcome": b"HEGEL/Q05B/ATTEMPT2/TARGET_D/OUTCOME/V2\x00",
    "pending": b"HEGEL/Q05B/ATTEMPT2/TARGET_D/PENDING/V2\x00",
}

_TAGGED_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FULL40 = re.compile(r"[0-9a-f]{40}\Z")


class Attempt2TargetDCampaignValidationError(ValueError):
    """Stable validation failure for the additive formal campaign."""


def _fail(detail: str) -> NoReturn:
    raise Attempt2TargetDCampaignValidationError(detail)


def canonical_json_bytes_v2(value: object) -> bytes:
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


def domain_digest_v2(domain: bytes, value: object) -> str:
    return "sha256:" + sha256(domain + canonical_json_bytes_v2(value)).hexdigest()


OWNER_AUTHORIZATION_DIGEST: Final = domain_digest_v2(
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
    if type(value) is not str or _TAGGED_SHA256.fullmatch(value) is None:
        _fail(f"{label} must be tagged sha256")
    return value


def _full40(value: object, label: str) -> str:
    if type(value) is not str or _FULL40.fullmatch(value) is None:
        _fail(f"{label} must be full lowercase commit id")
    return value


def validate_contract_v2(value: object) -> dict[str, Any]:
    contract = _object(value, "contract")
    required = {
        "schema_version",
        "contract_id",
        "status",
        "contract_digest",
        "authorization",
        "source_lineage",
        "meta_v11_bridge",
        "target_d_v2",
        "historical_v2r20_evidence",
        "fixed_paths",
        "preconsumption_gate_order",
        "execution_policy",
        "state_machine",
        "forbidden_scope",
    }
    if set(contract) != required:
        _fail("contract field registry differs")
    if (
        contract["schema_version"] != CONTRACT_SCHEMA_VERSION
        or contract["contract_id"] != CONTRACT_ID
        or contract["status"] != "IMPLEMENTED_NOT_EXECUTED"
        or contract["authorization"] != OWNER_AUTHORIZATION
    ):
        _fail("contract header or authorization differs")
    observed = _digest(contract["contract_digest"], "contract_digest")
    unsigned = deepcopy(contract)
    unsigned.pop("contract_digest")
    if domain_digest_v2(DOMAINS["contract"], unsigned) != observed:
        _fail("contract digest differs")
    lineage = _object(contract["source_lineage"], "source_lineage")
    _full40(lineage.get("required_ancestor_commit"), "required ancestor")
    _full40(lineage.get("required_ancestor_parent"), "required ancestor parent")
    _full40(lineage.get("target_d_commit"), "Target-D commit")
    if (
        lineage.get("current_head_must_be_clean") is not True
        or lineage.get("attempt2_slice_is_additive_only") is not True
        or lineage.get("required_ancestor_is_exact_post_meta_commit") is not True
        or lineage.get("runtime_clean_full_head_and_tree_bound_in_capsule") is not True
    ):
        _fail("source lineage policy differs")
    target = _object(contract["target_d_v2"], "target_d_v2")
    if (
        target.get("artifact_max_canonical_bytes") != 2_147_483_648
        or target.get("expected_artifact_status")
        != "Q05B_QUALIFICATION_20_OF_20_Q1_NOT_RUN"
        or target.get("formal_actual_completed") is not False
    ):
        _fail("Target-D authority boundary differs")
    execution = _object(contract["execution_policy"], "execution_policy")
    if (
        execution.get("logical_attempt_count") != 1
        or execution.get("maximum_actual_supervisor_popen_count") != 1
        or execution.get("pending_exit_code") != 75
        or execution.get("all_gates_before_consumption_and_popen") is not True
        or execution.get("post_popen_terminal_no_retry") is not True
    ):
        _fail("execution policy differs")
    state = _object(contract["state_machine"], "state_machine")
    if (
        state.get("preconsumption_pending_is_stateless") is not True
        or state.get(
            "stable_campaign_lock_precedes_cleanup_replay_and_state_creation"
        )
        is not True
        or state.get("campaign_lock_is_held_through_terminal_publication")
        is not True
        or state.get("concurrent_run_waits_then_replays_existing_ledger")
        is not True
    ):
        _fail("PENDING or campaign-lock state-machine policy differs")
    forbidden = _object(contract["forbidden_scope"], "forbidden_scope")
    if not forbidden or any(value is not True for value in forbidden.values()):
        _fail("forbidden scope differs")
    return contract


def build_pending_v2(
    contract: Mapping[str, Any],
    *,
    phase: str,
    condition: str,
    detail: str,
    source_commit_or_null: str | None,
    meta_chain_manifest_or_null: str | None,
) -> dict[str, Any]:
    validated = validate_contract_v2(contract)
    if not phase or not condition or not detail:
        _fail("PENDING phase, condition and detail must be nonempty")
    body = {
        "schema_version": PENDING_SCHEMA_VERSION,
        "status": PENDING_NOT_CONSUMED,
        "contract_digest": validated["contract_digest"],
        "authorization_digest": OWNER_AUTHORIZATION_DIGEST,
        "attempt_ordinal": 2,
        "phase": phase,
        "condition": condition,
        "detail": detail[:4096],
        "source_commit_or_null": source_commit_or_null,
        "meta_chain_manifest_or_null": meta_chain_manifest_or_null,
        "reentry": "FULL_PREFLIGHT_REPLAY_SAME_AUTHORIZATION_AND_SOURCE_LINEAGE",
        "campaign_execution_count": 0,
        "authorization_consumption_count": 0,
        "ledger_creation_or_write_count": 0,
        "supervisor_popen_count": 0,
        "retry_transition_count": 0,
        "pending_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("pending_digest")
    body["pending_digest"] = domain_digest_v2(DOMAINS["pending"], unsigned)
    return body


def validate_pending_v2(value: object, contract: Mapping[str, Any]) -> dict[str, Any]:
    pending = _object(value, "pending")
    rebuilt = deepcopy(pending)
    observed = _digest(rebuilt.pop("pending_digest", None), "pending_digest")
    if (
        pending.get("schema_version") != PENDING_SCHEMA_VERSION
        or pending.get("status") != PENDING_NOT_CONSUMED
        or pending.get("contract_digest") != validate_contract_v2(contract)["contract_digest"]
        or pending.get("authorization_digest") != OWNER_AUTHORIZATION_DIGEST
        or any(
            pending.get(field) != 0
            for field in (
                "campaign_execution_count",
                "authorization_consumption_count",
                "ledger_creation_or_write_count",
                "supervisor_popen_count",
                "retry_transition_count",
            )
        )
        or domain_digest_v2(DOMAINS["pending"], rebuilt) != observed
    ):
        _fail("PENDING wire differs")
    return pending


def build_capsule_v2(
    contract: Mapping[str, Any],
    *,
    source_commit: str,
    source_tree: str,
    meta_v11_receipt_digest: str,
    gate_receipt_digests: Mapping[str, str],
) -> dict[str, Any]:
    validated = validate_contract_v2(contract)
    _full40(source_commit, "source_commit")
    _full40(source_tree, "source_tree")
    _digest(meta_v11_receipt_digest, "meta_v11_receipt_digest")
    gates = dict(gate_receipt_digests)
    if list(gates) != list(validated["preconsumption_gate_order"]):
        _fail("gate receipt order differs")
    for name, digest in gates.items():
        _digest(digest, f"gate receipt {name}")
    body = {
        "schema_version": CAPSULE_SCHEMA_VERSION,
        "status": PREFLIGHT_ADMITTED,
        "contract_id": CONTRACT_ID,
        "contract_digest": validated["contract_digest"],
        "authorization": deepcopy(OWNER_AUTHORIZATION),
        "source_commit": source_commit,
        "source_tree": source_tree,
        "meta_v11_receipt_digest": meta_v11_receipt_digest,
        "gate_receipt_digests": gates,
        "fixed_paths": deepcopy(validated["fixed_paths"]),
        "maximum_actual_supervisor_popen_count": 1,
        "capsule_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("capsule_digest")
    body["capsule_digest"] = domain_digest_v2(DOMAINS["capsule"], unsigned)
    return body


def validate_capsule_v2(value: object, contract: Mapping[str, Any]) -> dict[str, Any]:
    capsule = _object(value, "capsule")
    observed = _digest(capsule.get("capsule_digest"), "capsule_digest")
    unsigned = deepcopy(capsule)
    unsigned.pop("capsule_digest", None)
    if (
        capsule.get("schema_version") != CAPSULE_SCHEMA_VERSION
        or capsule.get("status") != PREFLIGHT_ADMITTED
        or capsule.get("authorization") != OWNER_AUTHORIZATION
        or capsule.get("contract_digest") != validate_contract_v2(contract)["contract_digest"]
        or domain_digest_v2(DOMAINS["capsule"], unsigned) != observed
    ):
        _fail("capsule differs")
    return capsule


def build_ledger_record_v2(
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
    *,
    sequence: int,
    record_type: str,
    status: str,
    previous_record_digest_or_null: str | None,
    body: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_contract_v2(contract)
    cap = validate_capsule_v2(capsule, validated)
    if type(sequence) is not int or sequence not in (1, 2):
        _fail("ledger sequence differs")
    expected_types = {1: "CONSUMED", 2: "TERMINAL"}
    if record_type != expected_types[sequence]:
        _fail("ledger record type differs")
    if sequence == 1 and previous_record_digest_or_null is not None:
        _fail("genesis consumption must not have a previous digest")
    if sequence > 1:
        _digest(previous_record_digest_or_null, "previous ledger digest")
    record = {
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
    unsigned = deepcopy(record)
    unsigned.pop("record_digest")
    record["record_digest"] = domain_digest_v2(DOMAINS["ledger"], unsigned)
    return record


@dataclass(frozen=True)
class ActualDecisionV2:
    decision: str
    audit: dict[str, Any]


def decide_actual_return_v2(
    *,
    return_code: int,
    artifact_replay_passed: bool,
    artifact_status_or_null: str | None,
    artifact_sha256_or_null: str | None,
) -> ActualDecisionV2:
    if type(return_code) is not int:
        _fail("return code differs")
    artifact_valid = (
        artifact_replay_passed is True
        and artifact_status_or_null == "Q05B_QUALIFICATION_20_OF_20_Q1_NOT_RUN"
        and type(artifact_sha256_or_null) is str
        and re.fullmatch(r"[0-9a-f]{64}", artifact_sha256_or_null) is not None
    )
    success = return_code == 0 and artifact_valid
    decision = QUALIFIED if success else FAILED_TERMINAL
    audit_body = {
        "schema_version": DECISION_AUDIT_SCHEMA_VERSION,
        "decision": decision,
        "predicate_results": [
            {"predicate": "SUPERVISOR_RETURN_CODE_ZERO", "passed": return_code == 0},
            {"predicate": "TARGET_D_ARTIFACT_REPLAY_PASS", "passed": artifact_replay_passed is True},
            {
                "predicate": "TARGET_D_ARTIFACT_STATUS_EXACT",
                "passed": artifact_status_or_null
                == "Q05B_QUALIFICATION_20_OF_20_Q1_NOT_RUN",
            },
            {"predicate": "FORMAL_ACTUAL_NOT_SHADOW", "passed": True},
        ],
        "return_code": return_code,
        "artifact_sha256_or_null": artifact_sha256_or_null,
        "decision_audit_digest": None,
    }
    unsigned = deepcopy(audit_body)
    unsigned.pop("decision_audit_digest")
    audit_body["decision_audit_digest"] = domain_digest_v2(
        DOMAINS["decision_audit"], unsigned
    )
    return ActualDecisionV2(decision=decision, audit=audit_body)


def build_outcome_v2(
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
    *,
    decision: ActualDecisionV2,
    supervisor_return_code: int,
    stdout_sha256: str,
    stderr_sha256: str,
    artifact_sha256_or_null: str | None,
    supervisor_error_or_null: str | None = None,
) -> dict[str, Any]:
    validated = validate_contract_v2(contract)
    cap = validate_capsule_v2(capsule, validated)
    if decision.decision != decision.audit.get("decision"):
        _fail("decision and audit must come from one core return")
    body = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "status": decision.decision,
        "contract_digest": validated["contract_digest"],
        "capsule_digest": cap["capsule_digest"],
        "attempt_ordinal": 2,
        "actual_supervisor_popen_count": 1,
        "supervisor_return_code": supervisor_return_code,
        "stdout_sha256": stdout_sha256,
        "stderr_sha256": stderr_sha256,
        "artifact_sha256_or_null": artifact_sha256_or_null,
        "supervisor_error_or_null": supervisor_error_or_null,
        "decision": decision.decision,
        "decision_audit": deepcopy(decision.audit),
        "q1_state": "NOT_RUN",
        "retry_or_resume_allowed": False,
        "outcome_digest": None,
    }
    unsigned = deepcopy(body)
    unsigned.pop("outcome_digest")
    body["outcome_digest"] = domain_digest_v2(DOMAINS["outcome"], unsigned)
    return body


__all__ = [
    "Attempt2TargetDCampaignValidationError",
    "ActualDecisionV2",
    "CONSUMED",
    "FAILED_TERMINAL",
    "OUTCOME_UNKNOWN_TERMINAL",
    "OWNER_AUTHORIZATION",
    "OWNER_AUTHORIZATION_DIGEST",
    "OWNER_AUTHORIZATION_TEXT",
    "PENDING_NOT_CONSUMED",
    "PREFLIGHT_ADMITTED",
    "QUALIFIED",
    "build_capsule_v2",
    "build_ledger_record_v2",
    "build_outcome_v2",
    "build_pending_v2",
    "canonical_json_bytes_v2",
    "decide_actual_return_v2",
    "domain_digest_v2",
    "validate_capsule_v2",
    "validate_contract_v2",
    "validate_pending_v2",
]
