"""Pure prospective owner authorization for future Phase-2B C1 construction.

This module validates one closed, canonical, content-addressed policy supplied
as bytes or an already decoded object.  It deliberately has no filesystem,
Docker, network, key-generation, campaign-ID allocation, scoring, or signature
authentication path.  A successful result authorizes construction and records
conditional scope only; it cannot make a campaign eligible or executable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from types import MappingProxyType
from typing import Final, Mapping, NoReturn


PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_SCHEMA: Final = (
    "hegel-machine-phase2b-c1-owner-actual-unsealed-960-authorization/1"
)
PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_DOCUMENT_ID: Final = (
    "hegel-phase2b-c1-owner-authorization-from-zero-actual-unsealed-960-v1"
)
PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_STATUS: Final = (
    "OWNER_AUTHORIZED_PROSPECTIVE_CONDITIONAL_CONSTRUCTION_AND_EXECUTION"
)
PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_CLAIM_LEVEL: Final = (
    "PROSPECTIVE_CONDITIONAL_AUTHORIZATION_NOT_IMPLEMENTATION_NOT_EXECUTION_NOT_EVIDENCE"
)
PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION/BODY/V1\x00"
)
PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_PREFIX: Final = (
    "phase2b_c1_owner_actual_unsealed_960_authorization_v1_"
)
PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_ID: Final = (
    "phase2b_c1_owner_actual_unsealed_960_authorization_v1_"
    "dd2cd6a98b429fe01f172daf16ea70404c91356e40e9e42daed4cb5416a5c50d"
)

FAIL_AUTHORIZATION_INPUT: Final = "FAIL_PHASE2B_C1_OWNER_AUTHORIZATION_V1_INPUT"
FAIL_AUTHORIZATION_JSON: Final = "FAIL_PHASE2B_C1_OWNER_AUTHORIZATION_V1_JSON"
FAIL_AUTHORIZATION_SCHEMA: Final = "FAIL_PHASE2B_C1_OWNER_AUTHORIZATION_V1_SCHEMA"
FAIL_AUTHORIZATION_CANONICAL: Final = (
    "FAIL_PHASE2B_C1_OWNER_AUTHORIZATION_V1_CANONICAL"
)
FAIL_AUTHORIZATION_CONTENT_ID: Final = (
    "FAIL_PHASE2B_C1_OWNER_AUTHORIZATION_V1_CONTENT_ID"
)
FAIL_AUTHORIZATION_CLAIM_BOUNDARY: Final = (
    "FAIL_PHASE2B_C1_OWNER_AUTHORIZATION_V1_CLAIM_BOUNDARY"
)

_MAXIMUM_AUTHORIZATION_BYTES: Final = 128 * 1024
_AUTHORIZATION_FIELDS: Final = frozenset({"authorization_body", "authorization_id"})


class Phase2BC1OwnerActualUnsealed960AuthorizationErrorV1(ValueError):
    """Stable fail-closed error for the prospective authorization boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BC1OwnerActualUnsealed960AuthorizationErrorV1(code, detail)


_EXACT_CONTRACT_BINDINGS: Final = {
    "protocol_id": (
        "phase2b_protocol_"
        "62ad411b5b5a0f912626c54e4bd822a8c585a8f612f5fce6040cce500a11756a"
    ),
    "exact_freeze_version": "hegel-freeze-p2b-p3-v1.0.1",
    "exact_freeze_id": (
        "phase2b_exact_freeze_"
        "ffa1fd4fed0b5c2c018803aa9f730b8c85c144efe7e4aa324256681d1c742cbe"
    ),
    "authority_class": "OWNER_ACCEPTED_CONTAINER_TECHNICAL_ACTORS_V1",
    "authority_requirement_id": (
        "phase2b_c1_owner_accepted_container_authority_requirement_v1_"
        "53631d563f8da8f26c7202b7dd0681be5423fe2ca5c15e9a9f5deaa2920d9940"
    ),
    "actor_profile_content_id": (
        "phase2b_c1_container_actor_profile_v2_"
        "b2f081771f3de8d2a531b08b58109d95c51815904e9612e66f1bbfe69babf906"
    ),
    "actor_profile_canonical_sha256": (
        "8ca1850f026dabb7a4b69add0f89b33a3fcfb789223b2fd95ef0379608adbeed"
    ),
    "logical_stage_wire_schema_id": (
        "phase2b_c1_logical_stage_wire_schema_v1_"
        "1a625fbc369b057b9fc295fec02391d0421fc071ca41905b15901d4365833b5d"
    ),
    "logical_stage_wire_policy_id": (
        "phase2b_c1_logical_stage_wire_policy_v2_"
        "c88653f2caa92eae1dfb7456217c4ca05c339e8f23e0540a06e435552dc59573"
    ),
    "stage_event_contract_manifest_id": (
        "phase2b_c1_stage_event_contract_manifest_v1_"
        "c0929d05bdae56205e08700ab760e88a4668c686e73fe4be7a00ce6d69f9baf7"
    ),
}

_AUTHORIZED_SCOPE: Final = [
    "INDEPENDENT_HOLDOUT_GENERATOR_FROM_ZERO_CONSTRUCTION",
    "EMBEDDING_NEAREST_PROTOTYPE_BASELINE_REGISTRATION_CAMPAIGN",
    "FROZEN_LLM_SEMANTIC_ONLY_BASELINE_REGISTRATION_CAMPAIGN",
    "FLAT_LEARNED_TYPED_BASELINE_REGISTRATION_CAMPAIGN",
    "READINESS_AND_CAMPAIGN_MACHINERY_CONSTRUCTION",
    "ONE_FUTURE_FRESH_ACTUAL_UNSEALED_960_CAMPAIGN_AFTER_PREREQUISITES_AND_INDEPENDENT_LAUNCH_OVERLAY",
]

_BASELINE_REGISTRATIONS: Final = [
    {
        "baseline_id": "embedding_nearest_prototype",
        "baseline_spec_id": (
            "phase2b_baseline_spec_"
            "d14fa83a6843b92d1a502bba97b982323d1de8bf0abe22763d8c1cb67c97cfa4"
        ),
        "revision_policy": "exact_40_hex_commit_required",
    },
    {
        "baseline_id": "frozen_llm_semantic_only",
        "baseline_spec_id": (
            "phase2b_baseline_spec_"
            "c07005a2e32423f5e11a2160367b089d8f69208289239bfc18821fb351f13fa8"
        ),
        "revision_policy": "exact_40_hex_commit_required",
    },
    {
        "baseline_id": "flat_learned_typed",
        "baseline_spec_id": (
            "phase2b_baseline_spec_"
            "0e7b098030bf7e5a36f2deb0befd40fbabc55ce538a78765be2aa33d6b7af6d2"
        ),
        "revision_policy": "repository_dependency_lock_required",
    },
]

_PROSPECTIVE_AUTHORIZATION_CLAIMS: Final = {
    "construction": True,
    "independent_holdout_generator_construction_authorized": True,
    "three_baseline_registration_campaigns_authorized": True,
    "readiness_machinery_construction_authorized": True,
    "campaign_machinery_construction_authorized": True,
    "actual_unsealed_960_campaign_conditionally_authorized": True,
    "maximum_fresh_campaigns_authorized": 1,
}

_PREFLIGHT_SAFETY_BOUNDARY: Final = {
    "authorization_is_self_executing": False,
    "campaign_id": None,
    "attempt_id": None,
    "campaign_id_allocation_authorized": False,
    "actual_campaign_eligible": False,
    "execution_authorized": False,
    "formal_holdout_generation_authorized": False,
    "shadow_only": True,
    "prerequisites_verified": False,
    "independent_launch_overlay_verified": False,
    "legacy_logical_stage_fields_unchanged": True,
    "legacy_registry_policy_authorization_literal": (
        "NOT_AUTHORIZED_BY_FROZEN_ACTOR_PROFILE_V2_OR_EXECUTION_REGISTRY_V1"
    ),
    "legacy_actual_campaign_eligible": False,
    "legacy_actual_execution_claimed": False,
    "legacy_execution_authorized": False,
}

_CONTAINER_ACTOR_THREAT_MODEL: Final = {
    "same_admin_controller": True,
    "owner_accepted_threat_model": True,
    "organizational_independence": False,
    "independent_human_actors": False,
    "technical_role_independence_profile_required": True,
    "technical_role_independence_live_verified": False,
    "remote_attestation": False,
    "hardware_key_nonexportability": False,
    "malicious_admin_resistance": False,
}

_AUTHORIZATION_ARTIFACT_PERMANENT_NONCLAIMS: Final = {
    "implementation_exists": False,
    "generator_exists": False,
    "baseline_registrations_exist": False,
    "owner_signature_authenticated": False,
    "docker_invoked": False,
    "filesystem_accessed": False,
    "network_accessed": False,
    "keys_generated": False,
    "scoring_performed": False,
    "holdout_generated": False,
    "holdout_formally_sealed": False,
    "actual_execution_claimed": False,
    "formal_gate_evaluation_performed": False,
    "effect_evidence": False,
    "formal_phase2b_exit_claim": False,
    "c1_exit_evidence": False,
    "active_promotion_authorized": False,
    "organizational_independence": False,
    "independent_human_actors": False,
    "remote_attestation": False,
    "hardware_key_nonexportability": False,
    "malicious_admin_resistance": False,
    "cross_host_global_one_shot_verified": False,
}

_FUTURE_EXECUTION_PREREQUISITES: Final = [
    "AUTHORIZATION_CONTENT_ID_VERIFIED_AND_NOT_REVOKED",
    "EXECUTION_BRANCH_EXACT_BINDINGS_REPLAYED",
    "INDEPENDENT_HOLDOUT_GENERATOR_IMPLEMENTED_AND_QUALIFIED",
    "THREE_BASELINE_REGISTRATIONS_FROZEN_BEFORE_GENERATION",
    "VALIDATION_VERSION_PASSED",
    "WIRE_BUILDER_AND_COVERT_CHANNEL_AUDIT_PASSED",
    "SIX_ROLE_LIVE_DOCKER_QUALIFICATION_VERIFIED",
    "DURABLE_CUSTODY_LEDGER_CAS_AND_ATTEMPT_DENOMINATOR_VERIFIED",
    "EXECUTION_FREEZE_AND_COMMITMENTS_CROSS_BOUND",
    "FRESH_CAMPAIGN_AND_ATTEMPT_IDS_ALLOCATED_BY_SEPARATE_LAUNCH_OVERLAY",
    "INDEPENDENT_LAUNCH_OVERLAY_AUTHORIZED_AND_VERIFIED",
]

_REVOCATION_AND_VERSIONING: Final = {
    "authorization_epoch": 1,
    "supersedes_authorization_id": None,
    "revocation_status_at_issue": "ACTIVE",
    "historical_artifact_immutable": True,
    "retroactive_invalidation_default": False,
    "revocation_requires_separate_owner_artifact": True,
    "launch_overlay_must_check_latest_revocation_status": True,
    "scope_or_binding_change_requires_new_major_version": True,
    "revocation_cannot_create_execution_evidence": True,
}

_EXPECTED_BODY: Final = {
    "schema_version": PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_SCHEMA,
    "document_id": PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_DOCUMENT_ID,
    "status": PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_STATUS,
    "claim_level": PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_CLAIM_LEVEL,
    "authority_class": "OWNER_ACCEPTED_CONTAINER_TECHNICAL_ACTORS_V1",
    "authorization_semantics": (
        "OWNER_PROSPECTIVE_SCOPE_AUTHORIZATION_NOT_IMPLEMENTATION_NOT_EXECUTION_NOT_EVIDENCE"
    ),
    "canonicalization": (
        "SORTED_KEYS_ASCII_JSON_NO_WHITESPACE_NO_NEWLINE_ALLOW_NAN_FALSE"
    ),
    "exact_contract_bindings": _EXACT_CONTRACT_BINDINGS,
    "authorized_scope_in_order": _AUTHORIZED_SCOPE,
    "baseline_registrations_in_order": _BASELINE_REGISTRATIONS,
    "prospective_authorization_claims": _PROSPECTIVE_AUTHORIZATION_CLAIMS,
    "preflight_safety_boundary": _PREFLIGHT_SAFETY_BOUNDARY,
    "container_actor_threat_model": _CONTAINER_ACTOR_THREAT_MODEL,
    "authorization_artifact_permanent_nonclaims": (
        _AUTHORIZATION_ARTIFACT_PERMANENT_NONCLAIMS
    ),
    "future_execution_prerequisites_in_order": _FUTURE_EXECUTION_PREREQUISITES,
    "revocation_and_versioning": _REVOCATION_AND_VERSIONING,
}


def _exact_json_equal(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return set(value) == set(expected) and all(
            _exact_json_equal(value[key], expected[key]) for key in expected
        )
    if type(expected) is list:
        return len(value) == len(expected) and all(
            _exact_json_equal(item, expected_item)
            for item, expected_item in zip(value, expected, strict=True)
        )
    return value == expected


def _detached_json_copy(value: object) -> object:
    if type(value) is dict:
        return {key: _detached_json_copy(item) for key, item in value.items()}
    if type(value) is list:
        return [_detached_json_copy(item) for item in value]
    return value


def _deep_freeze(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_deep_thaw(item) for item in value]
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        _fail(FAIL_AUTHORIZATION_CANONICAL, f"authorization is not canonical JSON: {exc}")


def _reject_float(_: str) -> NoReturn:
    _fail(FAIL_AUTHORIZATION_JSON, "floating-point values are forbidden")


def _reject_constant(_: str) -> NoReturn:
    _fail(FAIL_AUTHORIZATION_JSON, "non-finite numeric constants are forbidden")


def _pairs_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            _fail(FAIL_AUTHORIZATION_JSON, "duplicate or non-string object key")
        result[key] = value
    return result


def canonical_phase2b_c1_owner_actual_unsealed_960_authorization_body_bytes_v1(
    body: object,
) -> bytes:
    """Return canonical bytes only for the one exact V1 authorization body."""

    if not _exact_json_equal(body, _EXPECTED_BODY):
        _fail(FAIL_AUTHORIZATION_SCHEMA, "authorization body differs from exact V1")
    return _canonical_json_bytes(body)


def phase2b_c1_owner_actual_unsealed_960_authorization_id_v1(body: object) -> str:
    """Content-address the exact policy body; this allocates no campaign ID."""

    canonical = canonical_phase2b_c1_owner_actual_unsealed_960_authorization_body_bytes_v1(
        body
    )
    return PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_PREFIX + hashlib.sha256(
        PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_DOMAIN + canonical
    ).hexdigest()


def _validate_authorization(value: object) -> tuple[Mapping[str, object], str]:
    if type(value) is not dict or set(value) != _AUTHORIZATION_FIELDS:
        _fail(FAIL_AUTHORIZATION_SCHEMA, "authorization wrapper field set differs")
    body = value["authorization_body"]
    canonical_phase2b_c1_owner_actual_unsealed_960_authorization_body_bytes_v1(body)
    authorization_id = value["authorization_id"]
    if (
        type(authorization_id) is not str
        or re.fullmatch(
            re.escape(PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_PREFIX)
            + r"[0-9a-f]{64}",
            authorization_id,
        )
        is None
    ):
        _fail(FAIL_AUTHORIZATION_CONTENT_ID, "authorization ID syntax differs")
    expected_id = phase2b_c1_owner_actual_unsealed_960_authorization_id_v1(body)
    if (
        authorization_id != expected_id
        or authorization_id
        != PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_ID
    ):
        _fail(FAIL_AUTHORIZATION_CONTENT_ID, "authorization content ID differs")
    frozen = _deep_freeze(_detached_json_copy(body))
    assert isinstance(frozen, Mapping)
    return frozen, authorization_id


class Phase2BC1OwnerActualUnsealed960AuthorizationDispositionV1(str, Enum):
    PROSPECTIVE_CONDITIONAL_AUTHORIZATION_VERIFIED_NOT_EXECUTION = (
        "PROSPECTIVE_CONDITIONAL_AUTHORIZATION_VERIFIED_NOT_EXECUTION"
    )
    REJECTED = "REJECTED"


class Phase2BC1OwnerActualUnsealed960AuthorizationReasonV1(str, Enum):
    EXACT_PROSPECTIVE_AUTHORIZATION_CONTENT_VERIFIED = (
        "EXACT_PROSPECTIVE_AUTHORIZATION_CONTENT_VERIFIED"
    )
    WRONG_INPUT_TYPE = "WRONG_INPUT_TYPE"
    AUTHORIZATION_REJECTED = "AUTHORIZATION_REJECTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_RESULT_TOKEN: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class Phase2BC1OwnerActualUnsealed960AuthorizationVerificationV1:
    disposition: Phase2BC1OwnerActualUnsealed960AuthorizationDispositionV1
    reason: Phase2BC1OwnerActualUnsealed960AuthorizationReasonV1
    authorization_id: str | None
    claim_level: str
    policy: Mapping[str, object] | None
    authorization_content_verified: bool
    construction: bool
    actual_unsealed_960_campaign_conditionally_authorized: bool
    authorization_is_self_executing: bool
    campaign_id: None
    attempt_id: None
    campaign_id_allocation_authorized: bool
    actual_campaign_eligible: bool
    execution_authorized: bool
    formal_holdout_generation_authorized: bool
    shadow_only: bool
    legacy_logical_stage_fields_unchanged: bool
    owner_signature_authenticated: bool
    docker_invoked: bool
    filesystem_accessed: bool
    network_accessed: bool
    keys_generated: bool
    scoring_performed: bool
    actual_execution_claimed: bool
    c1_exit_evidence: bool

    def __new__(
        cls, token: object = None
    ) -> "Phase2BC1OwnerActualUnsealed960AuthorizationVerificationV1":
        if token is not _RESULT_TOKEN:
            raise TypeError("authorization verification results are privately issued")
        return object.__new__(cls)


def _issue_result(
    *,
    disposition: Phase2BC1OwnerActualUnsealed960AuthorizationDispositionV1,
    reason: Phase2BC1OwnerActualUnsealed960AuthorizationReasonV1,
    authorization_id: str | None,
    policy: Mapping[str, object] | None,
    accepted: bool,
) -> Phase2BC1OwnerActualUnsealed960AuthorizationVerificationV1:
    result = Phase2BC1OwnerActualUnsealed960AuthorizationVerificationV1(_RESULT_TOKEN)
    preflight = policy["preflight_safety_boundary"] if accepted and policy else None
    nonclaims = (
        policy["authorization_artifact_permanent_nonclaims"]
        if accepted and policy
        else None
    )
    prospective = policy["prospective_authorization_claims"] if accepted and policy else None
    values = {
        "disposition": disposition,
        "reason": reason,
        "authorization_id": authorization_id,
        "claim_level": PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_CLAIM_LEVEL,
        "policy": policy,
        "authorization_content_verified": accepted,
        "construction": bool(accepted and prospective["construction"]),
        "actual_unsealed_960_campaign_conditionally_authorized": bool(
            accepted
            and prospective["actual_unsealed_960_campaign_conditionally_authorized"]
        ),
        "authorization_is_self_executing": False,
        "campaign_id": None,
        "attempt_id": None,
        "campaign_id_allocation_authorized": False,
        "actual_campaign_eligible": False,
        "execution_authorized": False,
        "formal_holdout_generation_authorized": False,
        "shadow_only": True,
        "legacy_logical_stage_fields_unchanged": bool(
            accepted and preflight["legacy_logical_stage_fields_unchanged"]
        ),
        "owner_signature_authenticated": False,
        "docker_invoked": False,
        "filesystem_accessed": False,
        "network_accessed": False,
        "keys_generated": False,
        "scoring_performed": False,
        "actual_execution_claimed": False,
        "c1_exit_evidence": False,
    }
    if accepted and any(nonclaims.values()):
        _fail(FAIL_AUTHORIZATION_CLAIM_BOUNDARY, "permanent nonclaim changed polarity")
    for name, field_value in values.items():
        object.__setattr__(result, name, field_value)
    return result


def verify_phase2b_c1_owner_actual_unsealed_960_authorization_v1(
    value: object,
) -> Phase2BC1OwnerActualUnsealed960AuthorizationVerificationV1:
    """Verify an already decoded authorization without side effects."""

    if type(value) is not dict:
        return _issue_result(
            disposition=Phase2BC1OwnerActualUnsealed960AuthorizationDispositionV1.REJECTED,
            reason=Phase2BC1OwnerActualUnsealed960AuthorizationReasonV1.WRONG_INPUT_TYPE,
            authorization_id=None,
            policy=None,
            accepted=False,
        )
    try:
        policy, authorization_id = _validate_authorization(value)
    except Phase2BC1OwnerActualUnsealed960AuthorizationErrorV1:
        return _issue_result(
            disposition=Phase2BC1OwnerActualUnsealed960AuthorizationDispositionV1.REJECTED,
            reason=Phase2BC1OwnerActualUnsealed960AuthorizationReasonV1.AUTHORIZATION_REJECTED,
            authorization_id=None,
            policy=None,
            accepted=False,
        )
    except Exception:
        return _issue_result(
            disposition=Phase2BC1OwnerActualUnsealed960AuthorizationDispositionV1.REJECTED,
            reason=Phase2BC1OwnerActualUnsealed960AuthorizationReasonV1.INTERNAL_ERROR,
            authorization_id=None,
            policy=None,
            accepted=False,
        )
    return _issue_result(
        disposition=(
            Phase2BC1OwnerActualUnsealed960AuthorizationDispositionV1.
            PROSPECTIVE_CONDITIONAL_AUTHORIZATION_VERIFIED_NOT_EXECUTION
        ),
        reason=(
            Phase2BC1OwnerActualUnsealed960AuthorizationReasonV1.
            EXACT_PROSPECTIVE_AUTHORIZATION_CONTENT_VERIFIED
        ),
        authorization_id=authorization_id,
        policy=policy,
        accepted=True,
    )


def load_phase2b_c1_owner_actual_unsealed_960_authorization_v1(
    payload: bytes,
) -> Phase2BC1OwnerActualUnsealed960AuthorizationVerificationV1:
    """Parse exact canonical bytes and return a private verification result."""

    if type(payload) is not bytes or not 1 <= len(payload) <= _MAXIMUM_AUTHORIZATION_BYTES:
        return _issue_result(
            disposition=Phase2BC1OwnerActualUnsealed960AuthorizationDispositionV1.REJECTED,
            reason=Phase2BC1OwnerActualUnsealed960AuthorizationReasonV1.WRONG_INPUT_TYPE,
            authorization_id=None,
            policy=None,
            accepted=False,
        )
    try:
        text = payload.decode("ascii")
        decoded = json.loads(
            text,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
            object_pairs_hook=_pairs_without_duplicates,
        )
        if type(decoded) is not dict or payload != _canonical_json_bytes(decoded):
            _fail(FAIL_AUTHORIZATION_CANONICAL, "source bytes are not exact canonical JSON")
        policy, authorization_id = _validate_authorization(decoded)
    except Phase2BC1OwnerActualUnsealed960AuthorizationErrorV1:
        return _issue_result(
            disposition=Phase2BC1OwnerActualUnsealed960AuthorizationDispositionV1.REJECTED,
            reason=Phase2BC1OwnerActualUnsealed960AuthorizationReasonV1.AUTHORIZATION_REJECTED,
            authorization_id=None,
            policy=None,
            accepted=False,
        )
    except (UnicodeError, json.JSONDecodeError):
        return _issue_result(
            disposition=Phase2BC1OwnerActualUnsealed960AuthorizationDispositionV1.REJECTED,
            reason=Phase2BC1OwnerActualUnsealed960AuthorizationReasonV1.AUTHORIZATION_REJECTED,
            authorization_id=None,
            policy=None,
            accepted=False,
        )
    except Exception:
        return _issue_result(
            disposition=Phase2BC1OwnerActualUnsealed960AuthorizationDispositionV1.REJECTED,
            reason=Phase2BC1OwnerActualUnsealed960AuthorizationReasonV1.INTERNAL_ERROR,
            authorization_id=None,
            policy=None,
            accepted=False,
        )
    return _issue_result(
        disposition=(
            Phase2BC1OwnerActualUnsealed960AuthorizationDispositionV1.
            PROSPECTIVE_CONDITIONAL_AUTHORIZATION_VERIFIED_NOT_EXECUTION
        ),
        reason=(
            Phase2BC1OwnerActualUnsealed960AuthorizationReasonV1.
            EXACT_PROSPECTIVE_AUTHORIZATION_CONTENT_VERIFIED
        ),
        authorization_id=authorization_id,
        policy=policy,
        accepted=True,
    )


__all__ = [
    "FAIL_AUTHORIZATION_CANONICAL",
    "FAIL_AUTHORIZATION_CLAIM_BOUNDARY",
    "FAIL_AUTHORIZATION_CONTENT_ID",
    "FAIL_AUTHORIZATION_INPUT",
    "FAIL_AUTHORIZATION_JSON",
    "FAIL_AUTHORIZATION_SCHEMA",
    "PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_CLAIM_LEVEL",
    "PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_DOCUMENT_ID",
    "PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_DOMAIN",
    "PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_ID",
    "PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_PREFIX",
    "PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_SCHEMA",
    "PHASE2B_C1_OWNER_ACTUAL_UNSEALED_960_AUTHORIZATION_V1_STATUS",
    "Phase2BC1OwnerActualUnsealed960AuthorizationDispositionV1",
    "Phase2BC1OwnerActualUnsealed960AuthorizationErrorV1",
    "Phase2BC1OwnerActualUnsealed960AuthorizationReasonV1",
    "Phase2BC1OwnerActualUnsealed960AuthorizationVerificationV1",
    "canonical_phase2b_c1_owner_actual_unsealed_960_authorization_body_bytes_v1",
    "load_phase2b_c1_owner_actual_unsealed_960_authorization_v1",
    "phase2b_c1_owner_actual_unsealed_960_authorization_id_v1",
    "verify_phase2b_c1_owner_actual_unsealed_960_authorization_v1",
]
