"""Static Phase-2B C1 owner-accepted container-actor profile mechanics.

This module only loads, validates, content-addresses, and deeply freezes the
committed six-role policy profile.  It performs no Docker operation, key or
signature generation, filesystem ledger mutation, recognizer execution,
scoring, gate evaluation, or C1 promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Final, Mapping, NoReturn


PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_SCHEMA: Final = (
    "hegel-phase2b-c1-container-actor-profile/1"
)
PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_ID: Final = (
    "hegel-owner-accepted-phase2b-c1-container-technical-actors-v1"
)
PHASE2B_C1_CONTAINER_ACTOR_AUTHORITY_CLASS_V1: Final = (
    "OWNER_ACCEPTED_CONTAINER_TECHNICAL_ACTORS_V1"
)
PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_CONTENT_DOMAIN: Final = (
    b"HEGEL/PHASE2B/C1/CONTAINER_ACTOR_PROFILE/V1\x00"
)
PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_CONTENT_PREFIX: Final = (
    "phase2b_c1_container_actor_profile_v1_"
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
PROFILE_PATH: Final = PROJECT_ROOT / "config/phase2b_c1_container_actor_profile_v1.json"
_MAXIMUM_PROFILE_BYTES: Final = 256 * 1024
_MAXIMUM_BOUND_SOURCE_BYTES: Final = 16 * 1024 * 1024

FAIL_PROFILE_IO: Final = "FAIL_PHASE2B_C1_CONTAINER_ACTOR_PROFILE_IO"
FAIL_PROFILE_JSON: Final = "FAIL_PHASE2B_C1_CONTAINER_ACTOR_PROFILE_JSON"
FAIL_PROFILE_SCHEMA: Final = "FAIL_PHASE2B_C1_CONTAINER_ACTOR_PROFILE_SCHEMA"
FAIL_PROFILE_BINDING: Final = "FAIL_PHASE2B_C1_CONTAINER_ACTOR_PROFILE_BINDING"
FAIL_PROFILE_CLAIM_BOUNDARY: Final = (
    "FAIL_PHASE2B_C1_CONTAINER_ACTOR_PROFILE_CLAIM_BOUNDARY"
)


class Phase2BC1ContainerActorProfileErrorV1(ValueError):
    """Stable fail-closed error for the static profile boundary."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BC1ContainerActorProfileErrorV1(code, detail)


_TOP_LEVEL_FIELDS: Final = frozenset(
    {
        "schema_version",
        "profile_id",
        "authority_class",
        "scope",
        "owner_accepted_authority_requirement",
        "base_control_bindings",
        "reuse_boundary",
        "authority_disclosure",
        "actor_roles",
        "checkpoint_signature_policy",
        "execution_signature_policy",
        "key_policy",
        "docker_control_policy",
        "filesystem_policy",
        "chronology_policy",
        "claim_policy",
    }
)

_EXPECTED_SCOPE: Final = [
    "PHASE2B_C1_TECHNICAL_ACTOR_ELIGIBILITY",
    "PHASE2B_ACTUAL_UNSEALED_960_CEREMONY",
]

_OWNER_ACCEPTED_REQUIREMENT_DOMAIN_HEX: Final = (
    "484547454c2f504841534532422f43312f4f574e45525f41434345505445445f"
    "434f4e5441494e45525f415554484f524954595f524551554952454d454e542f"
    "563100"
)
_OWNER_ACCEPTED_REQUIREMENT_PREFIX: Final = (
    "phase2b_c1_owner_accepted_container_authority_requirement_v1_"
)
_OWNER_ACCEPTED_REQUIREMENT_STATEMENTS: Final = [
    [
        "claim_boundary",
        "owner_accepted_single_admin_container_technical_actor_eligibility_not_external_organizational_independence",
    ],
    ["authority_class", "OWNER_ACCEPTED_CONTAINER_TECHNICAL_ACTORS_V1"],
    ["profile_id", "hegel-owner-accepted-phase2b-c1-container-technical-actors-v1"],
    [
        "required_roles",
        [
            "GOVERNANCE_ANCHOR",
            "CUSTODIAN",
            "STORE_RECEIPT",
            "TIME_ATTESTOR",
            "RECOGNIZER",
            "SCORER_EVALUATOR",
        ],
    ],
    [
        "checkpoint_graph_roles",
        ["GOVERNANCE_ANCHOR", "CUSTODIAN", "STORE_RECEIPT", "TIME_ATTESTOR"],
    ],
    ["execution_operation_roles", ["RECOGNIZER", "SCORER_EVALUATOR"]],
    [
        "registry_binding",
        "fresh_registry_must_bind_requirement_id_profile_content_id_campaign_and_attempt_before_holdout_generation",
    ],
    [
        "store_semantics",
        "repository_external_linux_local_if_absent_and_rollback_detection_for_cooperating_processes_not_hostile_admin",
    ],
    [
        "time_semantics",
        "signed_monotonic_logical_order_not_utc_or_external_timestamp_authority",
    ],
    [
        "permanent_nonclaims",
        [
            "ORGANIZATIONAL_INDEPENDENCE",
            "INDEPENDENT_HUMAN_ACTORS",
            "MALICIOUS_HOST_OR_DOCKER_DAEMON_RESISTANCE",
            "HOSTILE_ADMINISTRATOR_DELETION_OR_ROLLBACK_RESISTANCE",
            "CROSS_HOST_GLOBAL_ONE_SHOT",
            "AUTHORITATIVE_UTC_OR_EXTERNAL_TIME",
        ],
    ],
]
_EXPECTED_OWNER_ACCEPTED_AUTHORITY_REQUIREMENT: Final = {
    "domain_hex": _OWNER_ACCEPTED_REQUIREMENT_DOMAIN_HEX,
    "id_prefix": _OWNER_ACCEPTED_REQUIREMENT_PREFIX,
    "statements": _OWNER_ACCEPTED_REQUIREMENT_STATEMENTS,
    "requirement_id": (
        "phase2b_c1_owner_accepted_container_authority_requirement_v1_"
        "53631d563f8da8f26c7202b7dd0681be5423fe2ca5c15e9a9f5deaa2920d9940"
    ),
    "legacy_external_authority_requirement_id": (
        "phase2b_actual_unsealed_960_external_authority_requirement_v2_"
        "fea188566b28e0a8bf50ce0ae760a63e5ee8d0355a737fea30ed8cd8f05779e0"
    ),
    "legacy_external_candidate_verifier_alone_satisfies_requirement": False,
    "future_execution_registry_must_bind_requirement_id": True,
    "future_execution_registry_must_bind_profile_content_id": True,
}

_EXPECTED_BASE_CONTROL_BINDINGS: Final = {
    "phase3_container_actor_profile": {
        "path": "config/phase3_container_actor_profile_v1.json",
        "profile_id": "hegel-owner-accepted-container-technical-actors-v1",
        "sha256": "0419b66ad72580a78f1145b1db6b08d8ce40d1211f3554f548690949ee3c0ad0",
        "reuse_kind": "SECURITY_CONTROL_SOURCE_ONLY_NOT_PHASE2B_QUALIFICATION",
    },
    "runtime_seccomp": {
        "path": "config/phase3_internal_actor_seccomp_v1.json",
        "sha256": "d1284e4731683b73352ecdd1577704ea87aa0b5c582b7b00757c3db4d2c950ca",
        "reuse_kind": "EXACT_RUNTIME_SYSCALL_FILTER",
    },
    "offline_build_seccomp": {
        "path": "config/phase3_m3_offline_build_seccomp_v1.json",
        "sha256": "41ebe406e2ea5c30572db12566e8c52b2986d23e8b1d6523dff00f5f7fce01fe",
        "reuse_kind": "EXACT_OFFLINE_BUILD_SYSCALL_FILTER",
    },
    "local_runtime_source": {
        "path": "src/hegel_machine/phase3_local_runtime_v1.py",
        "sha256": "35c8360f3eca46ef4b6759547a37e6c39f83acdf0e98c4a525698bb501cf9802",
        "reuse_kind": "EXACT_LOCAL_FILESYSTEM_AND_DOCKER_CONTROL_PLANE_MECHANICS",
    },
    "python_live_probe_source": {
        "path": "tools/phase3_container_actor_probe_v1.py",
        "sha256": "170b5c0ef5bdc6a587564e160304f4dc293c5bac4879991632128b3bd020469d",
        "reuse_kind": "PROBE_LOGIC_SOURCE_REQUIRES_SIX_ROLE_PHASE2B_ADAPTER",
    },
}

_EXPECTED_REUSE_BOUNDARY: Final = {
    "phase3_qualification_report_reusable_as_phase2b_evidence": False,
    "phase3_actor_keys_reusable": False,
    "phase3_actor_volumes_reusable": False,
    "phase3_formal_executor_reusable_without_phase2b_adapter": False,
    "fresh_phase2b_six_role_qualification_required": True,
}

_EXPECTED_AUTHORITY_DISCLOSURE: Final = {
    "same_admin_controller": True,
    "organizational_independence": False,
    "independent_human_actors": False,
    "technical_role_independence": True,
    "owner_accepted_threat_model": True,
    "remote_attestation": False,
    "hardware_key_nonexportability": False,
}

_EXPECTED_ACTOR_ROLES: Final = [
    {
        "purpose_id": 1,
        "role": "GOVERNANCE_ANCHOR",
        "actor_class": "AUTHORITY_SIGNER",
        "checkpoint_signature_purpose": "GOVERNANCE_CHECKPOINT",
        "fresh_ed25519_operation_key_required": True,
        "persistent_private_state_required": True,
        "exclusive_writable_mounts": ["/state"],
        "read_only_input_classes": [
            "PROFILE",
            "OWNER_ACCEPTED_AUTHORITY_REQUIREMENT",
            "SIGNER_REGISTRY",
            "PUBLIC_COMMITMENT_ROOTS",
            "EXECUTION_OPERATION_RECEIPTS",
            "TERMINAL_REPORT_STORE_RECEIPT",
            "SIGNED_LOGICAL_TERMINAL_TOKEN",
        ],
        "forbidden_input_classes": [
            "RAW_HOLDOUT_SECRETS",
            "ANSWER_MANIFEST_PLAINTEXT",
            "RECOGNIZER_WORKSPACE",
        ],
        "responsibility": "AUTHORIZE_PROFILE_REGISTRY_AND_TERMINAL_CHECKPOINT",
    },
    {
        "purpose_id": 2,
        "role": "CUSTODIAN",
        "actor_class": "AUTHORITY_SIGNER",
        "checkpoint_signature_purpose": "CUSTODIAN_CHECKPOINT",
        "fresh_ed25519_operation_key_required": True,
        "persistent_private_state_required": True,
        "exclusive_writable_mounts": ["/state", "/custody"],
        "read_only_input_classes": [
            "PROFILE",
            "FROZEN_GENERATION_INPUTS",
            "PREDICTION_COMMITMENT_ROOT",
            "PREDICTION_STORE_IF_ABSENT_RECEIPT",
            "SIGNED_PREDICTIONS_COMMITTED_TOKEN",
        ],
        "forbidden_input_classes": [
            "RECOGNIZER_WORKSPACE",
            "SCORER_WORKSPACE",
            "OTHER_ROLE_PRIVATE_KEYS",
        ],
        "responsibility": "GENERATE_SEAL_HOLD_REVEAL_AND_SIGN_CUSTODY_EVENTS",
    },
    {
        "purpose_id": 3,
        "role": "STORE_RECEIPT",
        "actor_class": "AUTHORITY_SIGNER",
        "checkpoint_signature_purpose": "STORE_IF_ABSENT_RECEIPT",
        "fresh_ed25519_operation_key_required": True,
        "persistent_private_state_required": True,
        "exclusive_writable_mounts": ["/state", "/ledger", "/cas"],
        "read_only_input_classes": [
            "PROFILE",
            "OPAQUE_CONTENT_BYTES",
            "CONTENT_ADDRESS_REQUESTS",
        ],
        "forbidden_input_classes": [
            "ANSWER_MANIFEST_PLAINTEXT",
            "RAW_CUSTODIAN_SEED",
            "OTHER_ROLE_PRIVATE_KEYS",
        ],
        "responsibility": "SOLE_WRITER_FOR_DURABLE_IF_ABSENT_LEDGER_AND_CAS_RECEIPTS",
    },
    {
        "purpose_id": 4,
        "role": "TIME_ATTESTOR",
        "actor_class": "AUTHORITY_SIGNER",
        "checkpoint_signature_purpose": "DECLARED_TIME_TOKEN",
        "fresh_ed25519_operation_key_required": True,
        "persistent_private_state_required": True,
        "exclusive_writable_mounts": ["/state"],
        "read_only_input_classes": [
            "PROFILE",
            "EVENT_CONTENT_ROOTS",
            "STORE_RECEIPT_ROOTS",
        ],
        "forbidden_input_classes": [
            "WALL_CLOCK_AUTHORITY_CLAIMS",
            "ANSWER_MANIFEST_PLAINTEXT",
            "OTHER_ROLE_PRIVATE_KEYS",
        ],
        "responsibility": "ISSUE_SIGNED_MONOTONIC_LOGICAL_ORDER_TOKENS",
    },
    {
        "purpose_id": 5,
        "role": "RECOGNIZER",
        "actor_class": "EXECUTION_ACTOR",
        "checkpoint_signature_purpose": None,
        "fresh_ed25519_operation_key_required": True,
        "persistent_private_state_required": True,
        "exclusive_writable_mounts": ["/state", "/prediction"],
        "read_only_input_classes": [
            "PROFILE",
            "EXECUTION_FREEZE_MANIFEST",
            "SIGNED_RUN_STARTED_TOKEN",
            "BLINDED_RECOGNIZER_INPUT_ARCHIVE",
        ],
        "forbidden_input_classes": [
            "ANSWER_MANIFEST_PLAINTEXT",
            "GATE_INPUT_MANIFEST_PLAINTEXT",
            "GENERATOR_SOURCE",
            "CUSTODY",
            "LEDGER",
            "LIVE_REPOSITORY",
        ],
        "responsibility": "EXECUTE_EXACTLY_ONE_BLINDED_960_CASE_RECOGNIZER_OPERATION",
    },
    {
        "purpose_id": 6,
        "role": "SCORER_EVALUATOR",
        "actor_class": "EXECUTION_ACTOR",
        "checkpoint_signature_purpose": None,
        "fresh_ed25519_operation_key_required": True,
        "persistent_private_state_required": True,
        "exclusive_writable_mounts": ["/state", "/score"],
        "read_only_input_classes": [
            "PROFILE",
            "PREDICTION_STORE_IF_ABSENT_RECEIPT",
            "SIGNED_REVEAL_TOKEN",
            "COMMITTED_PREDICTION_ARCHIVE",
            "REVEALED_ANSWER_MANIFEST",
            "REVEALED_GATE_INPUT_MANIFEST",
        ],
        "forbidden_input_classes": [
            "UNCOMMITTED_PREDICTIONS",
            "PRE_REVEAL_ANSWER_ACCESS",
            "RECOGNIZER_WORKSPACE",
            "CUSTODY",
            "LEDGER",
            "LIVE_REPOSITORY",
        ],
        "responsibility": "SCORE_ONCE_AFTER_REVEAL_AND_SIGN_REPORT_ROOTS",
    },
]

_EXPECTED_CHECKPOINT_SIGNATURE_POLICY: Final = {
    "required_graph_roles_in_order": [
        "GOVERNANCE_ANCHOR",
        "CUSTODIAN",
        "STORE_RECEIPT",
        "TIME_ATTESTOR",
    ],
    "required_graph_signature_purposes_in_order": [
        "GOVERNANCE_CHECKPOINT",
        "CUSTODIAN_CHECKPOINT",
        "STORE_IF_ABSENT_RECEIPT",
        "DECLARED_TIME_TOKEN",
    ],
    "required_graph_signature_count": 4,
    "graph_signer_locations": [
        ["GOVERNANCE_ANCHOR", "checkpoint.checkpoint_signatures[0]"],
        ["CUSTODIAN", "checkpoint.checkpoint_signatures[1]"],
        ["STORE_RECEIPT", "store_receipt.detached_signature"],
        ["TIME_ATTESTOR", "time_token.detached_signature"],
    ],
    "key_epoch": 1,
    "checkpoint_sequence": 0,
    "store_previous_generation": 0,
    "store_new_generation": 1,
    "store_condition": "IF_ABSENT",
}

_EXPECTED_EXECUTION_SIGNATURE_POLICY: Final = {
    "required_roles_in_order": ["RECOGNIZER", "SCORER_EVALUATOR"],
    "required_signature_purposes_in_order": [
        "RECOGNIZER_OPERATION_RECEIPT",
        "SCORER_EVALUATION_REPORT",
    ],
    "signature_domains_hex_in_order": [
        (
            "484547454c2f504841534532422f43312f5245434f474e495a45525f4f504552"
            "4154494f4e5f524543454950542f563100"
        ),
        (
            "484547454c2f504841534532422f43312f53434f5245525f4556414c55415449"
            "4f4e5f5245504f52542f563100"
        ),
    ],
    "signed_object_fields_in_order": {
        "RECOGNIZER_OPERATION_RECEIPT": [
            "authority_requirement_id",
            "actor_profile_content_id",
            "campaign_id",
            "attempt_id",
            "execution_freeze_manifest_id",
            "recognizer_image_digest",
            "recognizer_source_root",
            "input_archive_id",
            "input_archive_sha256",
            "prediction_archive_id",
            "prediction_archive_sha256",
            "run_started_token_id",
            "run_finished_token_id",
        ],
        "SCORER_EVALUATION_REPORT": [
            "authority_requirement_id",
            "actor_profile_content_id",
            "campaign_id",
            "attempt_id",
            "scorer_image_digest",
            "scorer_source_root",
            "prediction_store_receipt_id",
            "reveal_token_id",
            "answer_manifest_id",
            "gate_input_manifest_id",
            "score_report_root",
            "gate_report_root",
            "reports_committed_token_id",
        ],
    },
    "signer_locations": [
        ["RECOGNIZER", "recognizer_operation_receipt.detached_signature"],
        ["SCORER_EVALUATOR", "scorer_evaluation_report.detached_signature"],
    ],
    "execution_registry_must_bind_owner_accepted_requirement_id": True,
    "execution_registry_must_bind_actor_profile_content_id": True,
    "execution_registry_must_bind_campaign_and_attempt": True,
    "execution_registry_must_bind_image_and_source_digests": True,
    "implemented_by_static_profile_loader": False,
}

_EXPECTED_KEY_POLICY: Final = {
    "fresh_key_count": 6,
    "algorithm": "Ed25519",
    "generated_only_during_future_live_ceremony": True,
    "generated_by_profile_loader": False,
    "purpose_key_reuse_allowed": False,
    "qualification_key_reuse_allowed": False,
    "private_key_export_allowed": False,
    "public_registry_frozen_before_holdout_generation": True,
}

_EXPECTED_DOCKER_CONTROL_POLICY: Final = {
    "executable": "/usr/bin/docker",
    "explicit_host_argument": "--host=unix:///var/run/docker.sock",
    "socket": "/var/run/docker.sock",
    "pull_policy": "never",
    "network": "none",
    "root_filesystem_read_only": True,
    "nonroot_uid_gid": "65534:65534",
    "capabilities_dropped": "ALL",
    "no_new_privileges": True,
    "custom_seccomp_required": True,
    "six_live_containers_concurrent_for_separation_probe": True,
    "purpose_private_state_volume_required": True,
    "shared_writable_role_mount_allowed": False,
    "pairwise_distinct_namespaces": ["pid", "mnt", "net", "ipc", "uts"],
    "forbidden_mounts": [
        "DOCKER_SOCKET",
        "LIVE_REPOSITORY",
        "HOST_HOME",
        "/mnt/c",
        "OTHER_ROLE_PRIVATE_STATE",
    ],
    "authority_actor_limits": {"memory": "512m", "pids": 64},
    "recognizer_actor_limits": {
        "memory": "2g",
        "pids": 128,
        "timeout_seconds": 3600,
        "maximum_output_bytes": 536870912,
    },
    "scorer_actor_limits": {
        "memory": "2g",
        "pids": 128,
        "timeout_seconds": 3600,
        "maximum_output_bytes": 536870912,
    },
}

_EXPECTED_FILESYSTEM_POLICY: Final = {
    "durable_root_kind": "REPOSITORY_EXTERNAL_LINUX_LOCAL",
    "under_tmp_allowed": False,
    "under_mnt_c_allowed": False,
    "cloud_or_network_filesystem_allowed": False,
    "symlink_allowed": False,
    "directory_mode_octal": "0700",
    "record_mode_octal": "0600",
    "file_fsync_required": True,
    "directory_fsync_required": True,
    "atomic_rename_probe_required": True,
    "hardlink_no_replace_publication_required": True,
    "cooperating_process_lock_required": True,
    "crash_recovery_required": True,
    "hostile_owner_rollback_resistance_claimed": False,
    "cross_host_global_one_shot_claimed": False,
}

_EXPECTED_CHRONOLOGY_POLICY: Final = {
    "time_semantics": "SIGNED_MONOTONIC_LOGICAL_ORDER_NOT_UTC",
    "wall_clock_accuracy_claimed": False,
    "external_timestamp_authority_claimed": False,
    "signed_attempt_stage_order": [
        "PREREGISTERED",
        "RAW_INPUT_COMMITTED",
        "PACKAGE_COMMITMENTS_FROZEN",
        "RUN_STARTED",
        "RUN_FINISHED",
        "PREDICTIONS_COMMITTED",
        "REVEALED",
        "SCORING_STARTED",
        "REPORTS_COMMITTED",
        "TERMINAL_CONSUMED",
    ],
    "actor_enforced_gate_inputs": {
        "CUSTODIAN_REVEAL": [
            "PREDICTION_STORE_IF_ABSENT_RECEIPT",
            "SIGNED_PREDICTIONS_COMMITTED_TOKEN",
        ],
        "SCORER_START": [
            "PREDICTION_STORE_IF_ABSENT_RECEIPT",
            "SIGNED_REVEAL_TOKEN",
        ],
        "GOVERNANCE_TERMINAL": [
            "EXECUTION_OPERATION_RECEIPTS",
            "TERMINAL_REPORT_STORE_RECEIPT",
            "SIGNED_LOGICAL_TERMINAL_TOKEN",
        ],
    },
    "reveal_requires_prediction_store_receipt": True,
    "scoring_requires_reveal_token": True,
    "terminal_requires_report_store_receipt": True,
}

_EXPECTED_STATIC_PROFILE_CLAIMS: Final = {
    "profile_schema_validated": True,
    "six_role_contract_frozen": True,
    "base_control_bindings_frozen": True,
    "authority_disclosure_frozen": True,
    "actual_execution_claimed": False,
}

_EXPECTED_ACTUAL_RUN_CLAIMS: Final = {
    "docker_invoked": False,
    "actor_qualification_executed": False,
    "technical_role_independence_live_verified": False,
    "fresh_keys_generated": False,
    "signatures_generated": False,
    "owner_accepted_authority_requirement_signed_registry_verified": False,
    "execution_operation_signatures_verified": False,
    "actor_enforced_chronology_gate_inputs_verified": False,
    "registry_issuance_authority_verified": False,
    "custodian_identity_verified": False,
    "durable_attempt_ledger_verified": False,
    "one_shot_policy_enforced": False,
    "timeline_observed": False,
    "timeline_order_verified": False,
    "pre_reveal_commitment_timing_verified": False,
    "runtime_attestation_authority_verified": False,
    "recognizer_executed": False,
    "runtime_executed": False,
    "actual_960_case_run_verified": False,
    "scoring_performed": False,
    "prediction_scored": False,
    "actual_prediction_scoring_evidence": False,
    "formal_gate_evaluation_performed": False,
    "effect_evidence": False,
    "formal_c1_report_verified": False,
    "c1_exit_evidence": False,
}

_EXPECTED_PERMANENT_NONCLAIMS: Final = {
    "organizational_independence": False,
    "independent_human_actors": False,
    "malicious_host_root_resistance": False,
    "malicious_docker_daemon_resistance": False,
    "hostile_administrator_tamper_resistance_verified": False,
    "owner_deletion_resistance_verified": False,
    "owner_rollback_resistance_verified": False,
    "cross_host_global_one_shot_verified": False,
    "remote_attestation": False,
    "hardware_key_nonexportability": False,
    "authoritative_utc_time_verified": False,
    "absence_of_all_covert_channels": False,
}


def _reject_duplicate_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise ValueError("duplicate or non-string JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _reject_json_float(value: str) -> NoReturn:
    raise ValueError(f"floating-point JSON number is forbidden: {value}")


def _decode_profile(payload: bytes) -> dict[str, object]:
    if not 1 <= len(payload) <= _MAXIMUM_PROFILE_BYTES:
        _fail(FAIL_PROFILE_IO, "profile byte length is outside the frozen bound")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        _fail(FAIL_PROFILE_JSON, f"profile is not UTF-8: {exc}")
    if text.startswith("\ufeff"):
        _fail(FAIL_PROFILE_JSON, "UTF-8 BOM is forbidden")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_pairs,
            parse_constant=_reject_json_constant,
            parse_float=_reject_json_float,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        _fail(FAIL_PROFILE_JSON, f"profile is not strict JSON: {exc}")
    if type(value) is not dict:
        _fail(FAIL_PROFILE_SCHEMA, "profile root must be an exact JSON object")
    return value


def _canonical_profile_bytes(value: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        _fail(FAIL_PROFILE_SCHEMA, f"profile cannot be canonically encoded: {exc}")


def _owner_accepted_requirement_id(statements: object) -> str:
    try:
        encoded = json.dumps(
            statements,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        domain = bytes.fromhex(_OWNER_ACCEPTED_REQUIREMENT_DOMAIN_HEX)
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        _fail(FAIL_PROFILE_SCHEMA, f"authority requirement cannot be encoded: {exc}")
    return _OWNER_ACCEPTED_REQUIREMENT_PREFIX + hashlib.sha256(domain + encoded).hexdigest()


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


def _require_exact(value: object, expected: object, field: str) -> None:
    if not _exact_json_equal(value, expected):
        _fail(FAIL_PROFILE_SCHEMA, f"{field} differs from the frozen profile")


def _validate_claim_policy(value: object) -> None:
    if type(value) is not dict or set(value) != {
        "static_profile_claims",
        "actual_run_claims_required_false",
        "permanent_nonclaims_required_false",
    }:
        _fail(FAIL_PROFILE_CLAIM_BOUNDARY, "claim-policy field set differs")
    _require_exact(
        value["static_profile_claims"],
        _EXPECTED_STATIC_PROFILE_CLAIMS,
        "claim_policy.static_profile_claims",
    )
    _require_exact(
        value["actual_run_claims_required_false"],
        _EXPECTED_ACTUAL_RUN_CLAIMS,
        "claim_policy.actual_run_claims_required_false",
    )
    _require_exact(
        value["permanent_nonclaims_required_false"],
        _EXPECTED_PERMANENT_NONCLAIMS,
        "claim_policy.permanent_nonclaims_required_false",
    )
    if any(
        type(item) is not bool or item
        for item in value["actual_run_claims_required_false"].values()
    ):
        _fail(FAIL_PROFILE_CLAIM_BOUNDARY, "an actual-run claim is not exact false")
    if any(
        type(item) is not bool or item
        for item in value["permanent_nonclaims_required_false"].values()
    ):
        _fail(FAIL_PROFILE_CLAIM_BOUNDARY, "a permanent nonclaim is not exact false")


def _validate_mount_separation(roles: list[dict[str, object]]) -> None:
    owners: dict[str, int] = {}
    for role in roles:
        purpose_id = role["purpose_id"]
        mounts = role["exclusive_writable_mounts"]
        assert type(purpose_id) is int and type(mounts) is list
        for mount in mounts:
            if mount == "/state":
                continue
            if mount in owners:
                _fail(FAIL_PROFILE_SCHEMA, "a non-state writable mount is shared")
            owners[mount] = purpose_id
    if owners != {
        "/custody": 2,
        "/ledger": 3,
        "/cas": 3,
        "/prediction": 5,
        "/score": 6,
    }:
        _fail(FAIL_PROFILE_SCHEMA, "exclusive writable mount ownership differs")


def validate_phase2b_c1_container_actor_profile_v1(
    value: object,
) -> Mapping[str, object]:
    """Validate one decoded profile and return a fresh deeply immutable copy."""

    if type(value) is not dict or set(value) != _TOP_LEVEL_FIELDS:
        _fail(FAIL_PROFILE_SCHEMA, "top-level profile field set differs")
    _require_exact(
        value["schema_version"],
        PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_SCHEMA,
        "schema_version",
    )
    _require_exact(
        value["profile_id"],
        PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_ID,
        "profile_id",
    )
    _require_exact(
        value["authority_class"],
        PHASE2B_C1_CONTAINER_ACTOR_AUTHORITY_CLASS_V1,
        "authority_class",
    )
    _require_exact(value["scope"], _EXPECTED_SCOPE, "scope")
    _require_exact(
        value["owner_accepted_authority_requirement"],
        _EXPECTED_OWNER_ACCEPTED_AUTHORITY_REQUIREMENT,
        "owner_accepted_authority_requirement",
    )
    authority_requirement = value["owner_accepted_authority_requirement"]
    if authority_requirement["requirement_id"] != _owner_accepted_requirement_id(
        authority_requirement["statements"]
    ):
        _fail(FAIL_PROFILE_SCHEMA, "owner-accepted authority requirement ID differs")
    _require_exact(
        value["base_control_bindings"],
        _EXPECTED_BASE_CONTROL_BINDINGS,
        "base_control_bindings",
    )
    _require_exact(value["reuse_boundary"], _EXPECTED_REUSE_BOUNDARY, "reuse_boundary")
    _require_exact(
        value["authority_disclosure"],
        _EXPECTED_AUTHORITY_DISCLOSURE,
        "authority_disclosure",
    )
    _require_exact(value["actor_roles"], _EXPECTED_ACTOR_ROLES, "actor_roles")
    _validate_mount_separation(value["actor_roles"])
    _require_exact(
        value["checkpoint_signature_policy"],
        _EXPECTED_CHECKPOINT_SIGNATURE_POLICY,
        "checkpoint_signature_policy",
    )
    _require_exact(
        value["execution_signature_policy"],
        _EXPECTED_EXECUTION_SIGNATURE_POLICY,
        "execution_signature_policy",
    )
    _require_exact(value["key_policy"], _EXPECTED_KEY_POLICY, "key_policy")
    _require_exact(
        value["docker_control_policy"],
        _EXPECTED_DOCKER_CONTROL_POLICY,
        "docker_control_policy",
    )
    _require_exact(
        value["filesystem_policy"],
        _EXPECTED_FILESYSTEM_POLICY,
        "filesystem_policy",
    )
    _require_exact(
        value["chronology_policy"],
        _EXPECTED_CHRONOLOGY_POLICY,
        "chronology_policy",
    )
    _validate_claim_policy(value["claim_policy"])
    return _deep_freeze(_detached_json_copy(value))


def _detached_json_copy(value: object) -> object:
    if type(value) is dict:
        return {key: _detached_json_copy(item) for key, item in value.items()}
    if type(value) is list:
        return [_detached_json_copy(item) for item in value]
    if value is None or type(value) in {str, int, bool}:
        return value
    _fail(FAIL_PROFILE_SCHEMA, "profile contains a non-JSON runtime type")


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


def _is_lower_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _required_open_flag(name: str, code: str) -> int:
    value = getattr(os, name, None)
    if type(value) is not int or value == 0:
        _fail(code, f"required {name} open flag is unavailable")
    return value


def _directory_open_flags(code: str) -> int:
    return (
        os.O_RDONLY
        | _required_open_flag("O_DIRECTORY", code)
        | _required_open_flag("O_NOFOLLOW", code)
        | _required_open_flag("O_NONBLOCK", code)
        | _required_open_flag("O_CLOEXEC", code)
    )


def _regular_open_flags(code: str) -> int:
    return (
        os.O_RDONLY
        | _required_open_flag("O_NOFOLLOW", code)
        | _required_open_flag("O_NONBLOCK", code)
        | _required_open_flag("O_CLOEXEC", code)
        | getattr(os, "O_NOCTTY", 0)
    )


def _normalized_absolute_path(path: Path, code: str, label: str) -> Path:
    if not path.is_absolute():
        _fail(code, f"{label} must be absolute")
    normalized = Path(os.path.abspath(os.fspath(path)))
    if normalized != path or any(part in {"", ".", ".."} for part in path.parts[1:]):
        _fail(code, f"{label} must be normalized without traversal")
    return normalized


def _open_absolute_directory(path: Path, code: str, label: str) -> tuple[int, Path]:
    normalized = _normalized_absolute_path(path, code, label)
    flags = _directory_open_flags(code)
    descriptor: int | None = None
    try:
        descriptor = os.open("/", flags)
        for component in normalized.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            _fail(code, f"{label} is not a directory")
        return descriptor, normalized
    except Phase2BC1ContainerActorProfileErrorV1:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        _fail(code, f"{label} cannot be opened componentwise: {exc}")


def _stable_metadata(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _bounded_read_regular_fd(
    descriptor: int,
    *,
    maximum_bytes: int,
    code: str,
    label: str,
) -> bytes:
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > maximum_bytes
        ):
            _fail(code, f"{label} is not one bounded regular file")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    except Phase2BC1ContainerActorProfileErrorV1:
        raise
    except OSError as exc:
        _fail(code, f"{label} cannot be read: {exc}")
    if len(payload) != before.st_size or len(payload) > maximum_bytes:
        _fail(code, f"{label} byte length changed or exceeds its bound")
    if _stable_metadata(before) != _stable_metadata(after):
        _fail(code, f"{label} metadata changed during the read")
    return payload


def _read_relative_regular_file(
    root_descriptor: int,
    relative: str,
    *,
    maximum_bytes: int,
    code: str,
    label: str,
) -> bytes:
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        _fail(code, f"{label} relative path escapes its root")
    directory_descriptor = os.dup(root_descriptor)
    file_descriptor: int | None = None
    recheck_directory_descriptor: int | None = None
    recheck_file_descriptor: int | None = None
    try:
        for component in relative_path.parts[:-1]:
            next_descriptor = os.open(
                component,
                _directory_open_flags(code),
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        file_descriptor = os.open(
            relative_path.parts[-1],
            _regular_open_flags(code),
            dir_fd=directory_descriptor,
        )
        payload = _bounded_read_regular_fd(
            file_descriptor,
            maximum_bytes=maximum_bytes,
            code=code,
            label=label,
        )
        original_metadata = os.fstat(file_descriptor)
        recheck_directory_descriptor = os.dup(root_descriptor)
        for component in relative_path.parts[:-1]:
            next_descriptor = os.open(
                component,
                _directory_open_flags(code),
                dir_fd=recheck_directory_descriptor,
            )
            os.close(recheck_directory_descriptor)
            recheck_directory_descriptor = next_descriptor
        recheck_file_descriptor = os.open(
            relative_path.parts[-1],
            _regular_open_flags(code),
            dir_fd=recheck_directory_descriptor,
        )
        current_metadata = os.fstat(recheck_file_descriptor)
        if _stable_metadata(original_metadata) != _stable_metadata(current_metadata):
            _fail(code, f"{label} namespace identity changed during the read")
        return payload
    except Phase2BC1ContainerActorProfileErrorV1:
        raise
    except OSError as exc:
        _fail(code, f"{label} cannot be opened below its root: {exc}")
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if recheck_file_descriptor is not None:
            os.close(recheck_file_descriptor)
        if recheck_directory_descriptor is not None:
            os.close(recheck_directory_descriptor)
        os.close(directory_descriptor)


def _read_absolute_regular_file(
    path: Path,
    *,
    maximum_bytes: int,
    code: str,
    label: str,
) -> tuple[bytes, Path]:
    normalized = _normalized_absolute_path(path, code, label)
    parent_descriptor, _ = _open_absolute_directory(normalized.parent, code, f"{label} parent")
    try:
        parent_identity = _stable_metadata(os.fstat(parent_descriptor))
        payload = _read_relative_regular_file(
            parent_descriptor,
            normalized.name,
            maximum_bytes=maximum_bytes,
            code=code,
            label=label,
        )
        recheck_descriptor, _ = _open_absolute_directory(
            normalized.parent,
            code,
            f"{label} parent recheck",
        )
        try:
            if parent_identity != _stable_metadata(os.fstat(recheck_descriptor)):
                _fail(code, f"{label} parent identity changed during the read")
        finally:
            os.close(recheck_descriptor)
    finally:
        os.close(parent_descriptor)
    return payload, normalized


def _verify_bound_sources(repository_root: Path, bindings: Mapping[str, object]) -> None:
    root_descriptor, normalized_root = _open_absolute_directory(
        repository_root,
        FAIL_PROFILE_BINDING,
        "repository root",
    )
    try:
        root_identity = _stable_metadata(os.fstat(root_descriptor))
        for binding_name, binding_value in bindings.items():
            if not isinstance(binding_value, Mapping):
                _fail(FAIL_PROFILE_BINDING, f"binding {binding_name} is not a mapping")
            relative = binding_value.get("path")
            claimed = binding_value.get("sha256")
            if type(relative) is not str or type(claimed) is not str:
                _fail(FAIL_PROFILE_BINDING, f"binding {binding_name} identity is malformed")
            payload = _read_relative_regular_file(
                root_descriptor,
                relative,
                maximum_bytes=_MAXIMUM_BOUND_SOURCE_BYTES,
                code=FAIL_PROFILE_BINDING,
                label=f"binding {binding_name}",
            )
            if hashlib.sha256(payload).hexdigest() != claimed:
                _fail(FAIL_PROFILE_BINDING, f"binding {binding_name} SHA-256 differs")
        recheck_descriptor, _ = _open_absolute_directory(
            normalized_root,
            FAIL_PROFILE_BINDING,
            "repository root recheck",
        )
        try:
            if root_identity != _stable_metadata(os.fstat(recheck_descriptor)):
                _fail(FAIL_PROFILE_BINDING, "repository root identity changed")
        finally:
            os.close(recheck_descriptor)
    finally:
        os.close(root_descriptor)


_PROFILE_SNAPSHOT_ISSUANCE_TOKEN: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class Phase2BC1ContainerActorProfileSnapshotV1:
    def __new__(
        cls,
        token: object = None,
    ) -> "Phase2BC1ContainerActorProfileSnapshotV1":
        if token is not _PROFILE_SNAPSHOT_ISSUANCE_TOKEN:
            raise TypeError("profile snapshots are privately issued")
        return object.__new__(cls)

    schema_version: str
    profile_id: str
    authority_class: str
    profile_content_id: str
    canonical_profile_sha256: str
    source_profile_sha256: str
    source_path: str
    bound_source_sha256_verified: bool
    profile: Mapping[str, object]
    docker_invoked: bool = False
    keys_generated: bool = False
    signatures_generated: bool = False
    recognizer_executed: bool = False
    scoring_performed: bool = False
    c1_exit_evidence: bool = False

    def __post_init__(self) -> None:
        plain_profile = _deep_thaw(self.profile)
        if type(plain_profile) is not dict:
            _fail(FAIL_PROFILE_SCHEMA, "snapshot profile is not an exact JSON object")
        canonical = _canonical_profile_bytes(plain_profile)
        canonical_sha256 = hashlib.sha256(canonical).hexdigest()
        expected_content_id = (
            PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_CONTENT_PREFIX
            + hashlib.sha256(
                PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_CONTENT_DOMAIN + canonical
            ).hexdigest()
        )
        if (
            self.schema_version != PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_SCHEMA
            or self.profile_id != PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_ID
            or self.authority_class
            != PHASE2B_C1_CONTAINER_ACTOR_AUTHORITY_CLASS_V1
            or self.profile_content_id != expected_content_id
            or not _is_lower_sha256(self.canonical_profile_sha256)
            or self.canonical_profile_sha256 != canonical_sha256
            or not _is_lower_sha256(self.source_profile_sha256)
            or type(self.source_path) is not str
            or not self.source_path.startswith("/")
            or type(self.bound_source_sha256_verified) is not bool
            or self.bound_source_sha256_verified is not True
            or not isinstance(self.profile, Mapping)
            or plain_profile.get("schema_version")
            != PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_SCHEMA
            or plain_profile.get("profile_id")
            != PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_ID
            or plain_profile.get("authority_class")
            != PHASE2B_C1_CONTAINER_ACTOR_AUTHORITY_CLASS_V1
        ):
            _fail(FAIL_PROFILE_SCHEMA, "snapshot identity differs")
        for name in (
            "docker_invoked",
            "keys_generated",
            "signatures_generated",
            "recognizer_executed",
            "scoring_performed",
            "c1_exit_evidence",
        ):
            if type(getattr(self, name)) is not bool or getattr(self, name):
                _fail(FAIL_PROFILE_CLAIM_BOUNDARY, f"snapshot {name} must be exact false")


def _issue_profile_snapshot(
    *,
    profile_content_id: str,
    canonical_profile_sha256: str,
    source_profile_sha256: str,
    source_path: str,
    profile: Mapping[str, object],
) -> Phase2BC1ContainerActorProfileSnapshotV1:
    value = Phase2BC1ContainerActorProfileSnapshotV1(
        _PROFILE_SNAPSHOT_ISSUANCE_TOKEN
    )
    fields: tuple[tuple[str, object], ...] = (
        ("schema_version", PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_SCHEMA),
        ("profile_id", PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_ID),
        ("authority_class", PHASE2B_C1_CONTAINER_ACTOR_AUTHORITY_CLASS_V1),
        ("profile_content_id", profile_content_id),
        ("canonical_profile_sha256", canonical_profile_sha256),
        ("source_profile_sha256", source_profile_sha256),
        ("source_path", source_path),
        ("bound_source_sha256_verified", True),
        ("profile", profile),
        ("docker_invoked", False),
        ("keys_generated", False),
        ("signatures_generated", False),
        ("recognizer_executed", False),
        ("scoring_performed", False),
        ("c1_exit_evidence", False),
    )
    for name, field_value in fields:
        object.__setattr__(value, name, field_value)
    value.__post_init__()
    return value


def load_phase2b_c1_container_actor_profile_v1(
    profile_path: Path | None = None,
    *,
    repository_root: Path | None = None,
) -> Phase2BC1ContainerActorProfileSnapshotV1:
    """Return one fresh immutable static snapshot of the committed profile."""

    selected = PROFILE_PATH if profile_path is None else profile_path
    root = PROJECT_ROOT if repository_root is None else repository_root
    if not isinstance(selected, Path) or not isinstance(root, Path):
        _fail(FAIL_PROFILE_IO, "profile path and repository root must be exact Path values")
    payload, resolved = _read_absolute_regular_file(
        selected,
        maximum_bytes=_MAXIMUM_PROFILE_BYTES,
        code=FAIL_PROFILE_IO,
        label="profile",
    )
    decoded = _decode_profile(payload)
    immutable_profile = validate_phase2b_c1_container_actor_profile_v1(decoded)
    canonical = _canonical_profile_bytes(decoded)
    _verify_bound_sources(root, immutable_profile["base_control_bindings"])
    content_digest = hashlib.sha256(
        PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_CONTENT_DOMAIN + canonical
    ).hexdigest()
    return _issue_profile_snapshot(
        profile_content_id=(
            PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_CONTENT_PREFIX + content_digest
        ),
        canonical_profile_sha256=hashlib.sha256(canonical).hexdigest(),
        source_profile_sha256=hashlib.sha256(payload).hexdigest(),
        source_path=resolved.as_posix(),
        profile=immutable_profile,
    )


__all__ = [
    "FAIL_PROFILE_BINDING",
    "FAIL_PROFILE_CLAIM_BOUNDARY",
    "FAIL_PROFILE_IO",
    "FAIL_PROFILE_JSON",
    "FAIL_PROFILE_SCHEMA",
    "PHASE2B_C1_CONTAINER_ACTOR_AUTHORITY_CLASS_V1",
    "PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_CONTENT_DOMAIN",
    "PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_CONTENT_PREFIX",
    "PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_ID",
    "PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V1_SCHEMA",
    "PROFILE_PATH",
    "Phase2BC1ContainerActorProfileErrorV1",
    "Phase2BC1ContainerActorProfileSnapshotV1",
    "load_phase2b_c1_container_actor_profile_v1",
    "validate_phase2b_c1_container_actor_profile_v1",
]
