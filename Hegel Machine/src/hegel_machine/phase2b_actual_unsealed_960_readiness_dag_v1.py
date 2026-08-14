"""Pure Phase-2B actual-unsealed-960 readiness ordering companion.

This module verifies a closed, canonical, content-addressed readiness DAG.  It
corrects ordering only: formal trusted-wire and namespace-aware covert-auditor
implementation qualification must precede generator, baseline, and statistical
validation work.  It has no filesystem, Docker, network, key-generation,
campaign allocation, holdout generation, audit execution, or scoring path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from types import MappingProxyType
from typing import Final, Mapping, NoReturn


PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_SCHEMA: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-readiness-dag/1"
)
PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_DOCUMENT_ID: Final = (
    "hegel-phase2b-actual-unsealed-960-readiness-dag-v1"
)
PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_STATUS: Final = (
    "READINESS_ORDERING_ERRATUM_FROZEN_NOT_EXECUTED"
)
PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_CLAIM_LEVEL: Final = (
    "READINESS_ORDERING_ERRATUM_ONLY_NOT_GATE_EVIDENCE_NOT_EXECUTION"
)
PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_DOMAIN: Final = (
    b"HEGEL/PHASE2B/ACTUAL_UNSEALED_960/READINESS_DAG/BODY/V1\x00"
)
PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_PREFIX: Final = (
    "phase2b_actual_unsealed_960_readiness_dag_v1_"
)
PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_ID: Final = (
    "phase2b_actual_unsealed_960_readiness_dag_v1_"
    "775b86cb499db3a991686634335ec444f7e44c7caea92b1c2b948036b585ea84"
)

OWNER_AUTHORIZATION_ID_V1: Final = (
    "phase2b_c1_owner_actual_unsealed_960_authorization_v1_"
    "dd2cd6a98b429fe01f172daf16ea70404c91356e40e9e42daed4cb5416a5c50d"
)
PROTOCOL_ID: Final = (
    "phase2b_protocol_"
    "62ad411b5b5a0f912626c54e4bd822a8c585a8f612f5fce6040cce500a11756a"
)
EXACT_FREEZE_VERSION: Final = "hegel-freeze-p2b-p3-v1.0.1"
EXACT_FREEZE_ID: Final = (
    "phase2b_exact_freeze_"
    "ffa1fd4fed0b5c2c018803aa9f730b8c85c144efe7e4aa324256681d1c742cbe"
)

FAIL_READINESS_INPUT: Final = "FAIL_PHASE2B_READINESS_DAG_V1_INPUT"
FAIL_READINESS_JSON: Final = "FAIL_PHASE2B_READINESS_DAG_V1_JSON"
FAIL_READINESS_CANONICAL: Final = "FAIL_PHASE2B_READINESS_DAG_V1_CANONICAL"
FAIL_READINESS_SCHEMA: Final = "FAIL_PHASE2B_READINESS_DAG_V1_SCHEMA"
FAIL_READINESS_CONTENT_ID: Final = "FAIL_PHASE2B_READINESS_DAG_V1_CONTENT_ID"
FAIL_READINESS_CLAIM_BOUNDARY: Final = (
    "FAIL_PHASE2B_READINESS_DAG_V1_CLAIM_BOUNDARY"
)

_MAXIMUM_PAYLOAD_BYTES: Final = 128 * 1024
_WRAPPER_FIELDS: Final = frozenset({"readiness_dag_body", "readiness_dag_id"})
_BODY_FIELDS: Final = frozenset(
    {
        "schema_version",
        "document_id",
        "status",
        "claim_level",
        "canonicalization",
        "exact_contract_bindings",
        "ordering_erratum",
        "readiness_dag",
        "safety_boundary",
        "wire_covert_qualification_contract",
    }
)

_NODE_IDS: Final = (
    "R01_AUTHORIZATION_VERIFIED",
    "R02_EXECUTION_BRANCH_BINDINGS_REPLAYED",
    "R03_FORMAL_TRUSTED_WIRE_IMPLEMENTATION_QUALIFIED",
    "R04_FORMAL_COVERT_AUDITOR_IMPLEMENTATION_QUALIFIED",
    "R05_SYNTHETIC_DEV_WIRE_COVERT_QUALIFICATION_VERIFIED",
    "R06_GENERATOR_IMPLEMENTATION_QUALIFIED",
    "R07_BASELINE_REGISTRATIONS_FROZEN",
    "R08_STATISTICAL_VALIDATION_PASSED",
    "R09_SIX_ROLE_LIVE_DOCKER_QUALIFIED",
    "R10_DURABLE_CUSTODY_AND_LEDGER_VERIFIED",
    "R11_EXECUTION_FREEZE_COMMITMENTS_BOUND",
    "R12_INDEPENDENT_LAUNCH_OVERLAY_VERIFIED",
    "R13_FRESH_IDS_ALLOCATED_BY_VERIFIED_OVERLAY",
    "R14_ACTUAL_UNSEALED_960_COHORT_GENERATED",
    "R15_ACTUAL_CORPUS_COVERT_AUDIT_PASSED",
    "R16_INFERENCE_ADMISSION_UNLOCKED",
)

_EXPECTED_DEPENDENCIES: Final = (
    (),
    (_NODE_IDS[0],),
    (_NODE_IDS[1],),
    (_NODE_IDS[2],),
    (_NODE_IDS[3],),
    (_NODE_IDS[4],),
    (_NODE_IDS[4],),
    (_NODE_IDS[5], _NODE_IDS[6]),
    (_NODE_IDS[7],),
    (_NODE_IDS[8],),
    (_NODE_IDS[9],),
    (_NODE_IDS[10],),
    (_NODE_IDS[11],),
    (_NODE_IDS[12],),
    (_NODE_IDS[13],),
    (_NODE_IDS[14],),
)

_EXACT_BINDINGS: Final = {
    "owner_authorization_id": OWNER_AUTHORIZATION_ID_V1,
    "protocol_id": PROTOCOL_ID,
    "exact_freeze_version": EXACT_FREEZE_VERSION,
    "exact_freeze_id": EXACT_FREEZE_ID,
}

_EXACT_ORDERING_ERRATUM: Final = {
    "correction_kind": "STRICTER_ADDITIVE_LAUNCH_READINESS_ORDERING_COMPANION",
    "correction_reason": (
        "EXACT_FREEZE_PRIORITY_3_PRECEDES_PHASE2B_STATISTICAL_COMPLETION_PRIORITY_5"
    ),
    "old_list_problem": (
        "GENERATOR_BASELINE_AND_VALIDATION_PRECEDED_WIRE_COVERT_QUALIFICATION"
    ),
    "old_list_order_must_not_be_used_for_launch_readiness": True,
    "authorization_v1_historical_artifact_immutable": True,
    "supersedes_owner_authorization": False,
    "scope_changed": False,
    "conditions_relaxed": False,
    "prerequisite_evidence_created": False,
    "strict_ordering_constraints_in_order": [
        "FORMAL_WIRE_COVERT_QUALIFICATION_BEFORE_GENERATOR_BASELINES_AND_STATISTICAL_VALIDATION",
        "LAUNCH_OVERLAY_AUTHORIZATION_AND_VERIFICATION_BEFORE_FRESH_ID_ALLOCATION",
        "ACTUAL_CORPUS_COVERT_AUDIT_AFTER_GENERATION_AND_BEFORE_INFERENCE_ADMISSION",
    ],
}

_SAFETY_FIELDS_FALSE: Final = (
    "actual_campaign_eligible",
    "actual_960_capacity_qualified",
    "actual_corpus_audit_executed",
    "actual_execution_claimed",
    "actual_holdout_generated",
    "baseline_registrations_exist",
    "campaign_id_allocation_authorized",
    "docker_invoked",
    "execution_authorized",
    "formal_corpus_audit_executed",
    "formal_covert_channel_audit_passed",
    "formal_holdout_generation_authorized",
    "formal_trusted_wire_implementation_qualified",
    "generator_exists",
    "independent_launch_overlay_verified",
    "inference_admission_unlocked",
    "inference_performed",
    "keys_generated",
    "network_accessed",
    "prerequisites_verified",
    "scoring_performed",
    "synthetic_dev_qualification_executed",
    "verifier_filesystem_accessed",
)


class Phase2BActualUnsealed960ReadinessDagErrorV1(ValueError):
    """Stable fail-closed error for the V1 readiness ordering boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BActualUnsealed960ReadinessDagErrorV1(code, detail)


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
        _fail(FAIL_READINESS_CANONICAL, f"readiness DAG is not canonical JSON: {exc}")


def _reject_float(_: str) -> NoReturn:
    _fail(FAIL_READINESS_JSON, "floating-point values are forbidden")


def _reject_constant(_: str) -> NoReturn:
    _fail(FAIL_READINESS_JSON, "non-finite numeric constants are forbidden")


def _pairs_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            _fail(FAIL_READINESS_JSON, "duplicate or non-string object key")
        result[key] = value
    return result


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


def _strict_body_shape(body: object) -> dict[str, object]:
    if type(body) is not dict or set(body) != _BODY_FIELDS:
        _fail(FAIL_READINESS_SCHEMA, "readiness DAG body field set differs")
    if body["schema_version"] != PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_SCHEMA:
        _fail(FAIL_READINESS_SCHEMA, "schema version differs")
    if body["document_id"] != PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_DOCUMENT_ID:
        _fail(FAIL_READINESS_SCHEMA, "document ID differs")
    if body["status"] != PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_STATUS:
        _fail(FAIL_READINESS_SCHEMA, "status differs")
    if body["claim_level"] != PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_CLAIM_LEVEL:
        _fail(FAIL_READINESS_SCHEMA, "claim level differs")
    if body["canonicalization"] != (
        "SORTED_KEYS_ASCII_JSON_NO_WHITESPACE_NO_NEWLINE_ALLOW_NAN_FALSE"
    ):
        _fail(FAIL_READINESS_SCHEMA, "canonicalization declaration differs")
    if not _exact_json_equal(body["exact_contract_bindings"], _EXACT_BINDINGS):
        _fail(FAIL_READINESS_SCHEMA, "exact contract bindings differ")
    if not _exact_json_equal(body["ordering_erratum"], _EXACT_ORDERING_ERRATUM):
        _fail(FAIL_READINESS_SCHEMA, "ordering erratum differs")
    return body


def _validate_dag(body: Mapping[str, object]) -> None:
    dag = body["readiness_dag"]
    if type(dag) is not dict or set(dag) != {
        "all_nodes_satisfied",
        "edge_semantics",
        "node_count",
        "nodes_in_canonical_topological_order",
        "satisfied_node_count",
    }:
        _fail(FAIL_READINESS_SCHEMA, "readiness DAG container differs")
    if dag["edge_semantics"] != (
        "ALL_DEPENDENCY_NODE_EVIDENCE_MUST_VERIFY_BEFORE_NODE_ADMISSION"
    ):
        _fail(FAIL_READINESS_SCHEMA, "DAG edge semantics differ")
    if type(dag["node_count"]) is not int or dag["node_count"] != len(_NODE_IDS):
        _fail(FAIL_READINESS_SCHEMA, "DAG node count differs")
    if type(dag["satisfied_node_count"]) is not int or dag["satisfied_node_count"] != 0:
        _fail(FAIL_READINESS_CLAIM_BOUNDARY, "DAG cannot claim a satisfied node")
    if dag["all_nodes_satisfied"] is not False:
        _fail(FAIL_READINESS_CLAIM_BOUNDARY, "DAG cannot claim readiness")
    nodes = dag["nodes_in_canonical_topological_order"]
    if type(nodes) is not list or len(nodes) != len(_NODE_IDS):
        _fail(FAIL_READINESS_SCHEMA, "canonical node list differs")
    seen: set[str] = set()
    for index, node in enumerate(nodes):
        if type(node) is not dict or set(node) != {
            "node_id",
            "dependencies",
            "evidence_requirement",
            "initial_satisfied",
        }:
            _fail(FAIL_READINESS_SCHEMA, "readiness node field set differs")
        node_id = node["node_id"]
        if node_id != _NODE_IDS[index] or node_id in seen:
            _fail(FAIL_READINESS_SCHEMA, "readiness node order or identity differs")
        dependencies = node["dependencies"]
        if type(dependencies) is not list or tuple(dependencies) != (
            _EXPECTED_DEPENDENCIES[index]
        ):
            _fail(FAIL_READINESS_SCHEMA, "readiness node dependencies differ")
        if any(type(item) is not str or item not in seen for item in dependencies):
            _fail(FAIL_READINESS_SCHEMA, "DAG is not in topological order")
        if type(node["evidence_requirement"]) is not str or not node[
            "evidence_requirement"
        ]:
            _fail(FAIL_READINESS_SCHEMA, "node evidence requirement is empty")
        if node["initial_satisfied"] is not False:
            _fail(FAIL_READINESS_CLAIM_BOUNDARY, "node cannot start satisfied")
        seen.add(node_id)


def _validate_safety_boundary(body: Mapping[str, object]) -> None:
    safety = body["safety_boundary"]
    expected_fields = set(_SAFETY_FIELDS_FALSE) | {"campaign_id", "attempt_id", "shadow_only"}
    if type(safety) is not dict or set(safety) != expected_fields:
        _fail(FAIL_READINESS_SCHEMA, "safety boundary field set differs")
    if any(safety[field] is not False for field in _SAFETY_FIELDS_FALSE):
        _fail(FAIL_READINESS_CLAIM_BOUNDARY, "a permanent false claim changed polarity")
    if safety["campaign_id"] is not None or safety["attempt_id"] is not None:
        _fail(FAIL_READINESS_CLAIM_BOUNDARY, "readiness ordering cannot allocate IDs")
    if safety["shadow_only"] is not True:
        _fail(FAIL_READINESS_CLAIM_BOUNDARY, "readiness ordering must remain shadow-only")


def _validate_wire_covert_contract(body: Mapping[str, object]) -> None:
    contract = body["wire_covert_qualification_contract"]
    if type(contract) is not dict:
        _fail(FAIL_READINESS_SCHEMA, "wire/covert contract must be an object")
    required = {
        "canonical_json": "RFC-8785",
        "envelope_bytes": 65_536,
        "fixed_envelope_capacity_audit_required": True,
        "id_assignment_after_global_shuffle": True,
        "id_algorithm": "HMAC_SHA256_TRUNCATE_128_SET_UUIDV4_BITS",
        "namespace_aware_audit_required": True,
        "global_consistent_renamings": 32,
        "global_case_order_permutations": 32,
        "within_case_observation_order_permutations": 16,
        "standard_error_enabled_by_default": False,
        "standard_error_status": "STANDARD_ERROR_UNSUPPORTED",
        "standard_error_encounter_policy": "ABORT_WHOLE_BATCH_NO_PARTIAL_OUTPUT",
        "synthetic_dev_input_only": True,
        "synthetic_dev_only": True,
        "actual_corpus_input_allowed": False,
        "actual_corpus_qualification_claimed": False,
    }
    for key, expected in required.items():
        if key not in contract or not _exact_json_equal(contract[key], expected):
            _fail(FAIL_READINESS_SCHEMA, f"wire/covert contract field differs: {key}")
    if contract.get("independent_secret_keys") != ["K_shuffle", "K_id", "K_padding"]:
        _fail(FAIL_READINESS_SCHEMA, "wire secret-key roles differ")
    if contract.get("keys_separate_from") != ["latent_generator_seed", "answer_seed"]:
        _fail(FAIL_READINESS_SCHEMA, "wire key-separation sources differ")
    if contract.get("implementation_qualification_evidence_fields_in_order") != [
        "source_commit_sha",
        "source_archive_root",
        "dependency_lock_sha256",
        "runtime_image_digest",
        "synthetic_fixture_root",
        "qualification_receipt_id",
    ]:
        _fail(FAIL_READINESS_SCHEMA, "implementation evidence binding fields differ")
    if contract.get("existing_mechanics_nonqualifications") != [
        "ACCEPTED_JCS_SUBSET_IS_NOT_COMPLETE_RFC8785_QUALIFICATION",
        "SYNTHETIC_COVERT_STATISTICS_CAP_64_IS_NOT_ACTUAL_960_QUALIFICATION",
        "NON_AUTHORITATIVE_MECHANICS_DO_NOT_SATISFY_R03_R04_OR_R05",
    ]:
        _fail(FAIL_READINESS_SCHEMA, "existing-mechanics nonqualification list differs")
    namespaces = contract.get("renaming_namespaces")
    if type(namespaces) is not list or len(namespaces) != 10 or len(set(namespaces)) != 10:
        _fail(FAIL_READINESS_SCHEMA, "renaming namespace registry differs")
    atomic = contract.get("atomic_failure_requirements")
    if type(atomic) is not list or len(atomic) != 9 or len(set(atomic)) != 9:
        _fail(FAIL_READINESS_SCHEMA, "atomic-failure registry differs")
    channel_gate = contract.get("channel_only_gate")
    if type(channel_gate) is not dict or set(channel_gate) != {
        "family_wise_alpha",
        "label_permutations",
        "maximum_balanced_accuracy_advantage",
        "maximum_normalized_mutual_information",
        "minimum_adjusted_p",
        "multiple_testing",
    }:
        _fail(FAIL_READINESS_SCHEMA, "channel-only gate field set differs")


def _validate_body(body: object) -> dict[str, object]:
    exact_body = _strict_body_shape(body)
    _validate_dag(exact_body)
    _validate_safety_boundary(exact_body)
    _validate_wire_covert_contract(exact_body)
    canonical = _canonical_json_bytes(exact_body)
    computed_id = PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_PREFIX + hashlib.sha256(
        PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_DOMAIN + canonical
    ).hexdigest()
    if computed_id != PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_ID:
        _fail(FAIL_READINESS_SCHEMA, "readiness DAG body differs from exact V1")
    return exact_body


def canonical_phase2b_actual_unsealed_960_readiness_dag_body_bytes_v1(
    body: object,
) -> bytes:
    """Return canonical bytes only for the one exact V1 readiness DAG body."""

    return _canonical_json_bytes(_validate_body(body))


def phase2b_actual_unsealed_960_readiness_dag_id_v1(body: object) -> str:
    """Return the ordering artifact ID; this never allocates campaign IDs."""

    canonical = canonical_phase2b_actual_unsealed_960_readiness_dag_body_bytes_v1(body)
    return PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_PREFIX + hashlib.sha256(
        PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_DOMAIN + canonical
    ).hexdigest()


def _validate_wrapper(value: object) -> tuple[Mapping[str, object], str]:
    if type(value) is not dict or set(value) != _WRAPPER_FIELDS:
        _fail(FAIL_READINESS_SCHEMA, "readiness DAG wrapper field set differs")
    body = _validate_body(value["readiness_dag_body"])
    dag_id = value["readiness_dag_id"]
    if type(dag_id) is not str or re.fullmatch(
        re.escape(PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_PREFIX) + r"[0-9a-f]{64}",
        dag_id,
    ) is None:
        _fail(FAIL_READINESS_CONTENT_ID, "readiness DAG ID syntax differs")
    expected_id = phase2b_actual_unsealed_960_readiness_dag_id_v1(body)
    if dag_id != expected_id or dag_id != PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_ID:
        _fail(FAIL_READINESS_CONTENT_ID, "readiness DAG content ID differs")
    frozen = _deep_freeze(_detached_json_copy(body))
    assert isinstance(frozen, Mapping)
    return frozen, dag_id


class Phase2BActualUnsealed960ReadinessDispositionV1(str, Enum):
    READINESS_ORDERING_VERIFIED_NOT_GATE_EVIDENCE = (
        "READINESS_ORDERING_VERIFIED_NOT_GATE_EVIDENCE"
    )
    REJECTED = "REJECTED"


class Phase2BActualUnsealed960ReadinessReasonV1(str, Enum):
    EXACT_READINESS_ORDERING_ERRATUM_VERIFIED = (
        "EXACT_READINESS_ORDERING_ERRATUM_VERIFIED"
    )
    WRONG_INPUT_TYPE = "WRONG_INPUT_TYPE"
    READINESS_DAG_REJECTED = "READINESS_DAG_REJECTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_RESULT_TOKEN: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class Phase2BActualUnsealed960ReadinessVerificationV1:
    disposition: Phase2BActualUnsealed960ReadinessDispositionV1
    reason: Phase2BActualUnsealed960ReadinessReasonV1
    readiness_dag_id: str | None
    claim_level: str
    policy: Mapping[str, object] | None
    readiness_ordering_verified: bool
    ordering_erratum_verified: bool
    node_count: int
    satisfied_node_count: int
    formal_trusted_wire_implementation_qualified: bool
    formal_covert_channel_audit_passed: bool
    synthetic_dev_qualification_executed: bool
    formal_corpus_audit_executed: bool
    actual_960_capacity_qualified: bool
    generator_exists: bool
    baseline_registrations_exist: bool
    prerequisites_verified: bool
    campaign_id: None
    attempt_id: None
    campaign_id_allocation_authorized: bool
    actual_campaign_eligible: bool
    execution_authorized: bool
    formal_holdout_generation_authorized: bool
    actual_holdout_generated: bool
    inference_admission_unlocked: bool
    inference_performed: bool
    shadow_only: bool
    docker_invoked: bool
    verifier_filesystem_accessed: bool
    network_accessed: bool
    keys_generated: bool
    scoring_performed: bool
    actual_execution_claimed: bool

    def __new__(
        cls, token: object = None
    ) -> "Phase2BActualUnsealed960ReadinessVerificationV1":
        if token is not _RESULT_TOKEN:
            raise TypeError("readiness verification results are privately issued")
        return object.__new__(cls)


def _issue_result(
    *,
    disposition: Phase2BActualUnsealed960ReadinessDispositionV1,
    reason: Phase2BActualUnsealed960ReadinessReasonV1,
    readiness_dag_id: str | None,
    policy: Mapping[str, object] | None,
    accepted: bool,
) -> Phase2BActualUnsealed960ReadinessVerificationV1:
    result = Phase2BActualUnsealed960ReadinessVerificationV1(_RESULT_TOKEN)
    values = {
        "disposition": disposition,
        "reason": reason,
        "readiness_dag_id": readiness_dag_id,
        "claim_level": PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_CLAIM_LEVEL,
        "policy": policy,
        "readiness_ordering_verified": accepted,
        "ordering_erratum_verified": accepted,
        "node_count": len(_NODE_IDS) if accepted else 0,
        "satisfied_node_count": 0,
        "formal_trusted_wire_implementation_qualified": False,
        "formal_covert_channel_audit_passed": False,
        "synthetic_dev_qualification_executed": False,
        "formal_corpus_audit_executed": False,
        "actual_960_capacity_qualified": False,
        "generator_exists": False,
        "baseline_registrations_exist": False,
        "prerequisites_verified": False,
        "campaign_id": None,
        "attempt_id": None,
        "campaign_id_allocation_authorized": False,
        "actual_campaign_eligible": False,
        "execution_authorized": False,
        "formal_holdout_generation_authorized": False,
        "actual_holdout_generated": False,
        "inference_admission_unlocked": False,
        "inference_performed": False,
        "shadow_only": True,
        "docker_invoked": False,
        "verifier_filesystem_accessed": False,
        "network_accessed": False,
        "keys_generated": False,
        "scoring_performed": False,
        "actual_execution_claimed": False,
    }
    for name, field_value in values.items():
        object.__setattr__(result, name, field_value)
    return result


def verify_phase2b_actual_unsealed_960_readiness_dag_v1(
    value: object,
) -> Phase2BActualUnsealed960ReadinessVerificationV1:
    """Verify a decoded readiness DAG without reading or changing external state."""

    if type(value) is not dict:
        return _issue_result(
            disposition=Phase2BActualUnsealed960ReadinessDispositionV1.REJECTED,
            reason=Phase2BActualUnsealed960ReadinessReasonV1.WRONG_INPUT_TYPE,
            readiness_dag_id=None,
            policy=None,
            accepted=False,
        )
    try:
        policy, dag_id = _validate_wrapper(value)
    except Phase2BActualUnsealed960ReadinessDagErrorV1:
        return _issue_result(
            disposition=Phase2BActualUnsealed960ReadinessDispositionV1.REJECTED,
            reason=Phase2BActualUnsealed960ReadinessReasonV1.READINESS_DAG_REJECTED,
            readiness_dag_id=None,
            policy=None,
            accepted=False,
        )
    except Exception:
        return _issue_result(
            disposition=Phase2BActualUnsealed960ReadinessDispositionV1.REJECTED,
            reason=Phase2BActualUnsealed960ReadinessReasonV1.INTERNAL_ERROR,
            readiness_dag_id=None,
            policy=None,
            accepted=False,
        )
    return _issue_result(
        disposition=(
            Phase2BActualUnsealed960ReadinessDispositionV1.
            READINESS_ORDERING_VERIFIED_NOT_GATE_EVIDENCE
        ),
        reason=(
            Phase2BActualUnsealed960ReadinessReasonV1.
            EXACT_READINESS_ORDERING_ERRATUM_VERIFIED
        ),
        readiness_dag_id=dag_id,
        policy=policy,
        accepted=True,
    )


def load_phase2b_actual_unsealed_960_readiness_dag_v1(
    payload: bytes,
) -> Phase2BActualUnsealed960ReadinessVerificationV1:
    """Parse exact canonical bytes and return a private verification result."""

    if type(payload) is not bytes or not 1 <= len(payload) <= _MAXIMUM_PAYLOAD_BYTES:
        return _issue_result(
            disposition=Phase2BActualUnsealed960ReadinessDispositionV1.REJECTED,
            reason=Phase2BActualUnsealed960ReadinessReasonV1.WRONG_INPUT_TYPE,
            readiness_dag_id=None,
            policy=None,
            accepted=False,
        )
    try:
        decoded = json.loads(
            payload.decode("ascii"),
            parse_float=_reject_float,
            parse_constant=_reject_constant,
            object_pairs_hook=_pairs_without_duplicates,
        )
        if type(decoded) is not dict or payload != _canonical_json_bytes(decoded):
            _fail(FAIL_READINESS_CANONICAL, "source bytes are not exact canonical JSON")
        policy, dag_id = _validate_wrapper(decoded)
    except Phase2BActualUnsealed960ReadinessDagErrorV1:
        return _issue_result(
            disposition=Phase2BActualUnsealed960ReadinessDispositionV1.REJECTED,
            reason=Phase2BActualUnsealed960ReadinessReasonV1.READINESS_DAG_REJECTED,
            readiness_dag_id=None,
            policy=None,
            accepted=False,
        )
    except (UnicodeError, json.JSONDecodeError):
        return _issue_result(
            disposition=Phase2BActualUnsealed960ReadinessDispositionV1.REJECTED,
            reason=Phase2BActualUnsealed960ReadinessReasonV1.READINESS_DAG_REJECTED,
            readiness_dag_id=None,
            policy=None,
            accepted=False,
        )
    except Exception:
        return _issue_result(
            disposition=Phase2BActualUnsealed960ReadinessDispositionV1.REJECTED,
            reason=Phase2BActualUnsealed960ReadinessReasonV1.INTERNAL_ERROR,
            readiness_dag_id=None,
            policy=None,
            accepted=False,
        )
    return _issue_result(
        disposition=(
            Phase2BActualUnsealed960ReadinessDispositionV1.
            READINESS_ORDERING_VERIFIED_NOT_GATE_EVIDENCE
        ),
        reason=(
            Phase2BActualUnsealed960ReadinessReasonV1.
            EXACT_READINESS_ORDERING_ERRATUM_VERIFIED
        ),
        readiness_dag_id=dag_id,
        policy=policy,
        accepted=True,
    )


__all__ = [
    "EXACT_FREEZE_ID",
    "EXACT_FREEZE_VERSION",
    "FAIL_READINESS_CANONICAL",
    "FAIL_READINESS_CLAIM_BOUNDARY",
    "FAIL_READINESS_CONTENT_ID",
    "FAIL_READINESS_INPUT",
    "FAIL_READINESS_JSON",
    "FAIL_READINESS_SCHEMA",
    "OWNER_AUTHORIZATION_ID_V1",
    "PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_CLAIM_LEVEL",
    "PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_DOCUMENT_ID",
    "PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_DOMAIN",
    "PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_ID",
    "PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_PREFIX",
    "PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_SCHEMA",
    "PHASE2B_ACTUAL_UNSEALED_960_READINESS_DAG_V1_STATUS",
    "PROTOCOL_ID",
    "Phase2BActualUnsealed960ReadinessDagErrorV1",
    "Phase2BActualUnsealed960ReadinessDispositionV1",
    "Phase2BActualUnsealed960ReadinessReasonV1",
    "Phase2BActualUnsealed960ReadinessVerificationV1",
    "canonical_phase2b_actual_unsealed_960_readiness_dag_body_bytes_v1",
    "load_phase2b_actual_unsealed_960_readiness_dag_v1",
    "phase2b_actual_unsealed_960_readiness_dag_id_v1",
    "verify_phase2b_actual_unsealed_960_readiness_dag_v1",
]
