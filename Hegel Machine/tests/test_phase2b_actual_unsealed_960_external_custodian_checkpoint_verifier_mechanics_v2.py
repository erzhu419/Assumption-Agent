"""Independent tests for the candidate external-custodian verifier V2.

The Ed25519 keys below are deterministic and explicitly TEST_ONLY.  Passing
these tests establishes content-address and signature mechanics only.  It does
not provide a pretrusted registry, independent custodian, authoritative store
or time source, actual-960 execution, scoring, gates, effect, or C1 evidence.
"""

from __future__ import annotations

import ast
from dataclasses import fields, replace
from enum import Enum
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

import hegel_machine.phase2b_actual_unsealed_960_external_custodian_checkpoint_verifier_mechanics_v2 as verifier_v2


VERSION = "hegel-machine-phase2b-actual-unsealed-960-external-custodian-checkpoint-verifier-mechanics/2"
CLAIM_LEVEL = "NON_AUTHORITATIVE_SUPPLIED_EXTERNAL_AUTHORITY_CANDIDATE_MECHANICS_ONLY"
SCOPE = "SUPPLIED_EXTERNAL_AUTHORITY_CANDIDATE_NOT_PRETRUSTED_SINGLE_TERMINAL_CHECKPOINT"
SCHEMA_ID = "phase2b_actual_unsealed_960_external_checkpoint_verifier_schema_v2_8f62cf96090d3ea24d1b0d6d9f93ff67004d2913b9d573b8670507bf45182968"
POLICY_ID = "phase2b_actual_unsealed_960_external_checkpoint_verifier_policy_v2_daff9ee739aa2c44b74f67612761e852b471ea03006d9fd41b08523811b42976"
REQUIREMENT_ID = "phase2b_actual_unsealed_960_external_authority_requirement_v2_fea188566b28e0a8bf50ce0ae760a63e5ee8d0355a737fea30ed8cd8f05779e0"

PUBLIC_SURFACE = (
    "PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_VERSION",
    "PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_CLAIM_LEVEL",
    "PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_SCOPE",
    "PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_SCHEMA_ID",
    "PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_POLICY_ID",
    "PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_ED25519_BACKEND_AVAILABLE",
    "PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_AUTHORITY_REQUIREMENT_ID",
    "PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_AUTHORITY_REQUIREMENT_STATEMENTS",
    "ExternalCustodianCandidateRoleV2",
    "ExternalCustodianSignaturePurposeV2",
    "ExternalCustodianCheckpointDispositionV2",
    "ExternalCustodianCheckpointReasonV2",
    "ExternalCustodianCandidateSignerV2",
    "ExternalCustodianAuthorityCandidateProfileV2",
    "ExternalCustodianDetachedSignatureV2",
    "ExternalCustodianStoreReceiptCandidateV2",
    "ExternalCustodianTimeTokenCandidateV2",
    "ExternalCustodianCheckpointCandidateV2",
    "ExternalCustodianCheckpointVerifierMechanicsV2",
    "ExternalCustodianCheckpointVerifierRejectionV2",
    "verify_phase2b_actual_unsealed_960_external_custodian_checkpoint_verifier_mechanics_v2",
)

SIGNER_FIELDS = ("role", "key_epoch", "public_key", "key_id")
PROFILE_FIELDS = (
    "version", "schema_id", "policy_id", "claim_level", "scope",
    "authority_requirement_id", "campaign_id", "attempt_id",
    "external_anchor_locator", "external_anchor_evidence_sha256", "signer_keys",
    "signer_key_ids_root_sha256", "required_signature_count", "profile_id",
)
SIGNATURE_FIELDS = ("role", "purpose", "key_epoch", "key_id", "signature")
STORE_FIELDS = (
    "version", "profile_id", "campaign_id", "attempt_id", "subject_id",
    "store_service_id", "object_key", "previous_generation", "new_generation",
    "cas_condition", "stored_content_sha256", "detached_signature", "receipt_id",
)
TIME_FIELDS = (
    "version", "profile_id", "campaign_id", "attempt_id", "subject_id",
    "time_source_id", "previous_declared_time_ns", "declared_time_ns",
    "detached_signature", "time_token_id",
)
CHECKPOINT_FIELDS = (
    "version", "profile_id", "campaign_id", "attempt_id", "checkpoint_sequence",
    "predecessor_checkpoint_id", "signed_attempt_chain_id",
    "signed_attempt_chain_root_sha256", "terminal_event_id", "local_ledger_id",
    "local_terminal_record_id", "subject_id", "store_receipt_id", "time_token_id",
    "checkpoint_signatures", "checkpoint_id",
)

TRUE_CLAIMS = (
    "candidate_profile_content_address_verified",
    "four_role_distinct_key_registry_verified",
    "role_epoch_purpose_domain_binding_verified",
    "terminal_subject_commitment_verified",
    "store_if_absent_generation_transition_verified",
    "store_receipt_signature_verified",
    "declared_time_monotonicity_verified",
    "time_token_signature_verified",
    "governance_checkpoint_signature_verified",
    "custodian_checkpoint_signature_verified",
    "checkpoint_cross_bindings_verified",
    "checkpoint_content_address_verified",
    "ordinary_exception_atomic_rejection_verified",
)
FALSE_CLAIMS = (
    "external_trust_anchor_verified", "registry_issuance_authority_verified",
    "custodian_identity_verified", "signer_independence_verified",
    "qualified_cryptographic_backend_verified", "external_store_identity_verified",
    "external_store_authority_verified", "external_store_receipt_authority_verified",
    "external_store_rollback_resistance_verified", "external_store_durability_verified",
    "hostile_administrator_tamper_resistance_verified", "cross_host_global_one_shot_verified",
    "authoritative_time_source_verified", "authoritative_time_verified", "timeline_observed",
    "authoritative_timeline_order_verified", "signed_attempt_chain_authority_verified",
    "local_ledger_externalized_verified", "authoritative_attempt_denominator_verified",
    "attempt_registry_complete", "retry_authorization_verified", "rerun_policy_enforced",
    "one_shot_policy_enforced", "terminal_append_receipt_verified",
    "campaign_finalization_verified", "actual_evidence_inputs_accepted",
    "admission_evidence_contract_complete", "authoritative_evidence_verifier_implemented",
    "admission_ready", "execution_authorized", "evidence_supplied", "evidence_verified",
    "input_archive_authority_verified", "prediction_archive_authority_verified",
    "answer_commitment_authority_verified", "gate_input_commitment_authority_verified",
    "pre_reveal_commitment_timing_verified", "runtime_attestation_authority_verified",
    "recognizer_executed", "runtime_executed", "actual_960_case_run_verified",
    "scoring_performed", "prediction_scored", "actual_prediction_scoring_evidence",
    "formal_gate_evaluation_performed", "overall_gate_results_materialized",
    "slice_gate_results_materialized", "wilson_bounds_evaluated", "preservation_evaluated",
    "scale_regret_evaluated", "bootstrap_evaluated", "baseline_outputs_verified",
    "effect_evidence", "formal_c1_report_verified", "c1_exit_evidence",
)
RESULT_PREFIX_FIELDS = (
    "disposition", "reason", "version", "schema_id", "policy_id", "claim_level",
    "scope", "result_id", "profile_id", "subject_id", "store_receipt_id",
    "time_token_id", "checkpoint_id", "required_signature_count",
    "verified_signature_count",
)
RESULT_FIELDS = (*RESULT_PREFIX_FIELDS, *TRUE_CLAIMS, *FALSE_CLAIMS)
REJECTION_FIELDS = (
    *RESULT_PREFIX_FIELDS, "partial_output_published", *TRUE_CLAIMS, *FALSE_CLAIMS,
)

REQUIREMENT_STATEMENTS = (
    ("claim_boundary", "requirements_for_live_external_authority_not_evidence_that_authority_exists"),
    ("required_distinct_roles", ("GOVERNANCE_ANCHOR", "CUSTODIAN", "STORE_RECEIPT", "TIME_ATTESTOR")),
    ("registry_requirement", "pretrusted_out_of_band_registry_issuance_and_identity_ceremony_required"),
    ("custodian_requirement", "organizationally_independent_custodian_identity_evidence_required"),
    ("store_requirement", "external_compare_and_swap_receipt_and_rollback_witness_required"),
    ("time_requirement", "independently_pinned_authoritative_time_provider_evidence_required"),
)

AUTHORITY_REQUIREMENT_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/AUTHORITY_REQUIREMENT/V2\x00"
KEY_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/KEY/V2\x00"
KEY_ROOT_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/KEY_ROOT/V2\x00"
PROFILE_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/PROFILE/V2\x00"
SUBJECT_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/SUBJECT/V2\x00"
STORE_SIGNATURE_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/STORE_SIGNATURE/V2\x00"
STORE_RECEIPT_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/STORE_RECEIPT/V2\x00"
TIME_SIGNATURE_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/TIME_SIGNATURE/V2\x00"
TIME_TOKEN_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/TIME_TOKEN/V2\x00"
CHECKPOINT_SIGNATURE_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/CHECKPOINT_SIGNATURE/V2\x00"
CHECKPOINT_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/CHECKPOINT/V2\x00"
RESULT_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/EXTERNAL_CHECKPOINT/RESULT/V2\x00"

KEY_PREFIX = "phase2b_actual_unsealed_960_external_candidate_key_v2_"
PROFILE_PREFIX = "phase2b_actual_unsealed_960_external_candidate_profile_v2_"
SUBJECT_PREFIX = "phase2b_actual_unsealed_960_external_checkpoint_subject_v2_"
STORE_SERVICE_PREFIX = "phase2b_actual_unsealed_960_external_store_service_v2_"
STORE_RECEIPT_PREFIX = "phase2b_actual_unsealed_960_external_store_receipt_v2_"
TIME_SOURCE_PREFIX = "phase2b_actual_unsealed_960_external_time_source_v2_"
TIME_TOKEN_PREFIX = "phase2b_actual_unsealed_960_external_time_token_v2_"
CHECKPOINT_PREFIX = "phase2b_actual_unsealed_960_external_checkpoint_v2_"
RESULT_PREFIX = "phase2b_actual_unsealed_960_external_checkpoint_verifier_result_v2_"
CAMPAIGN_PREFIX = "phase2b_actual_unsealed_960_campaign_v2_"
ATTEMPT_PREFIX = "phase2b_actual_unsealed_960_attempt_v2_"
SIGNED_CHAIN_PREFIX = "phase2b_actual_unsealed_960_signed_attempt_chain_v2_"
TERMINAL_EVENT_PREFIX = "phase2b_actual_unsealed_960_attempt_event_v2_"
LOCAL_LEDGER_PREFIX = "phase2b_actual_unsealed_960_local_attempt_ledger_v2_"
LOCAL_TERMINAL_PREFIX = "phase2b_actual_unsealed_960_local_terminal_record_v2_"

Role = verifier_v2.ExternalCustodianCandidateRoleV2
Purpose = verifier_v2.ExternalCustodianSignaturePurposeV2


class _PrehashBoundaryReached(BaseException):
    """Sentinel proving malformed input reached a hash/signature boundary."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(domain: bytes, pairs: tuple[tuple[str, object], ...]) -> str:
    return hashlib.sha256(domain + _canonical_json(pairs).encode("utf-8")).hexdigest()


def _opaque(prefix: str, label: str) -> str:
    return prefix + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _key_pairs(role: Role, public_key: bytes) -> tuple[tuple[str, object], ...]:
    return (
        ("role", role.value), ("key_epoch", 1), ("algorithm", "Ed25519"),
        ("public_key_hex", public_key.hex()),
    )


def _signature_pairs(value: verifier_v2.ExternalCustodianDetachedSignatureV2) -> tuple[tuple[str, object], ...]:
    return (
        ("role", value.role.value), ("purpose", value.purpose.value),
        ("key_epoch", value.key_epoch), ("key_id", value.key_id),
        ("signature_hex", value.signature.hex()),
    )


def _signature_message(
    domain: bytes,
    unsigned_pairs: tuple[tuple[str, object], ...],
    role: Role,
    purpose: Purpose,
    key_id: str,
) -> bytes:
    framing = (
        ("role", role.value), ("purpose", purpose.value), ("key_epoch", 1),
        ("key_id", key_id), ("unsigned_record", unsigned_pairs),
    )
    return domain + _canonical_json(framing).encode("utf-8")


def _detached(
    private_key: Ed25519PrivateKey,
    role: Role,
    purpose: Purpose,
    key_id: str,
    domain: bytes,
    unsigned_pairs: tuple[tuple[str, object], ...],
) -> verifier_v2.ExternalCustodianDetachedSignatureV2:
    return verifier_v2.ExternalCustodianDetachedSignatureV2(
        role=role,
        purpose=purpose,
        key_epoch=1,
        key_id=key_id,
        signature=private_key.sign(
            _signature_message(domain, unsigned_pairs, role, purpose, key_id)
        ),
    )


def _build_graph(seed: str = "A") -> SimpleNamespace:
    roles = tuple(Role)
    private_keys: dict[Role, Ed25519PrivateKey] = {}
    signers: list[verifier_v2.ExternalCustodianCandidateSignerV2] = []
    for role in roles:
        key_seed = hashlib.sha256(
            f"TEST_ONLY external checkpoint {seed} {role.value}".encode("utf-8")
        ).digest()
        private_key = Ed25519PrivateKey.from_private_bytes(key_seed)
        public_key = private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        key_id = KEY_PREFIX + _digest(KEY_DOMAIN, _key_pairs(role, public_key))
        private_keys[role] = private_key
        signers.append(verifier_v2.ExternalCustodianCandidateSignerV2(role, 1, public_key, key_id))
    signer_keys = tuple(signers)
    key_ids = tuple(value.key_id for value in signer_keys)
    root = hashlib.sha256()
    root.update(KEY_ROOT_DOMAIN)
    root.update(len(key_ids).to_bytes(4, "big"))
    for key_id in key_ids:
        encoded = key_id.encode("ascii")
        root.update(len(encoded).to_bytes(2, "big"))
        root.update(encoded)
    key_root = root.hexdigest()

    campaign_id = _opaque(CAMPAIGN_PREFIX, f"campaign-{seed}")
    attempt_id = _opaque(ATTEMPT_PREFIX, f"attempt-{seed}")
    profile_unsigned = (
        ("version", VERSION), ("schema_id", SCHEMA_ID), ("policy_id", POLICY_ID),
        ("claim_level", CLAIM_LEVEL), ("scope", SCOPE),
        ("authority_requirement_id", REQUIREMENT_ID), ("campaign_id", campaign_id),
        ("attempt_id", attempt_id),
        ("external_anchor_locator", f"test-only://candidate/{seed}"),
        ("external_anchor_evidence_sha256", hashlib.sha256(f"anchor-{seed}".encode()).hexdigest()),
        ("signer_keys", tuple(tuple(_key_pairs(item.role, item.public_key) + (("key_id", item.key_id),)) for item in signer_keys)),
        ("signer_key_ids_root_sha256", key_root), ("required_signature_count", 4),
    )
    profile_id = PROFILE_PREFIX + _digest(PROFILE_DOMAIN, profile_unsigned)
    profile = verifier_v2.ExternalCustodianAuthorityCandidateProfileV2(
        VERSION, SCHEMA_ID, POLICY_ID, CLAIM_LEVEL, SCOPE, REQUIREMENT_ID,
        campaign_id, attempt_id, f"test-only://candidate/{seed}",
        hashlib.sha256(f"anchor-{seed}".encode()).hexdigest(), signer_keys, key_root, 4,
        profile_id,
    )

    chain_id = _opaque(SIGNED_CHAIN_PREFIX, f"chain-{seed}")
    chain_root = hashlib.sha256(f"chain-root-{seed}".encode()).hexdigest()
    terminal_event_id = _opaque(TERMINAL_EVENT_PREFIX, f"terminal-event-{seed}")
    ledger_id = _opaque(LOCAL_LEDGER_PREFIX, f"ledger-{seed}")
    terminal_digest = hashlib.sha256(f"local-terminal-{seed}".encode()).hexdigest()
    local_terminal_id = LOCAL_TERMINAL_PREFIX + terminal_digest
    subject_pairs = (
        ("campaign_id", campaign_id), ("attempt_id", attempt_id),
        ("signed_attempt_chain_id", chain_id),
        ("signed_attempt_chain_root_sha256", chain_root),
        ("terminal_event_id", terminal_event_id), ("local_ledger_id", ledger_id),
        ("local_terminal_record_id", local_terminal_id),
    )
    subject_id = SUBJECT_PREFIX + _digest(SUBJECT_DOMAIN, subject_pairs)

    store_service_id = _opaque(STORE_SERVICE_PREFIX, f"store-{seed}")
    object_key = f"phase2b/actual-unsealed-960/{campaign_id}/{attempt_id}/terminal.json"
    store_unsigned = (
        ("version", VERSION), ("profile_id", profile_id), ("campaign_id", campaign_id),
        ("attempt_id", attempt_id), ("subject_id", subject_id),
        ("store_service_id", store_service_id), ("object_key", object_key),
        ("previous_generation", 0), ("new_generation", 1),
        ("cas_condition", "IF_ABSENT"), ("stored_content_sha256", terminal_digest),
    )
    store_signature = _detached(
        private_keys[Role.STORE_RECEIPT], Role.STORE_RECEIPT,
        Purpose.STORE_IF_ABSENT_RECEIPT, signer_keys[2].key_id,
        STORE_SIGNATURE_DOMAIN, store_unsigned,
    )
    receipt_id = STORE_RECEIPT_PREFIX + _digest(
        STORE_RECEIPT_DOMAIN,
        store_unsigned + (("detached_signature", tuple(_signature_pairs(store_signature))),),
    )
    store = verifier_v2.ExternalCustodianStoreReceiptCandidateV2(
        VERSION, profile_id, campaign_id, attempt_id, subject_id, store_service_id,
        object_key, 0, 1, "IF_ABSENT", terminal_digest, store_signature, receipt_id,
    )

    time_source_id = _opaque(TIME_SOURCE_PREFIX, f"time-{seed}")
    time_unsigned = (
        ("version", VERSION), ("profile_id", profile_id), ("campaign_id", campaign_id),
        ("attempt_id", attempt_id), ("subject_id", subject_id),
        ("time_source_id", time_source_id), ("previous_declared_time_ns", 1_000),
        ("declared_time_ns", 2_000),
    )
    time_signature = _detached(
        private_keys[Role.TIME_ATTESTOR], Role.TIME_ATTESTOR,
        Purpose.DECLARED_TIME_TOKEN, signer_keys[3].key_id,
        TIME_SIGNATURE_DOMAIN, time_unsigned,
    )
    time_token_id = TIME_TOKEN_PREFIX + _digest(
        TIME_TOKEN_DOMAIN,
        time_unsigned + (("detached_signature", tuple(_signature_pairs(time_signature))),),
    )
    time_token = verifier_v2.ExternalCustodianTimeTokenCandidateV2(
        VERSION, profile_id, campaign_id, attempt_id, subject_id, time_source_id,
        1_000, 2_000, time_signature, time_token_id,
    )

    checkpoint_unsigned = (
        ("version", VERSION), ("profile_id", profile_id), ("campaign_id", campaign_id),
        ("attempt_id", attempt_id), ("checkpoint_sequence", 0),
        ("predecessor_checkpoint_id", None), ("signed_attempt_chain_id", chain_id),
        ("signed_attempt_chain_root_sha256", chain_root),
        ("terminal_event_id", terminal_event_id), ("local_ledger_id", ledger_id),
        ("local_terminal_record_id", local_terminal_id), ("subject_id", subject_id),
        ("store_receipt_id", receipt_id), ("time_token_id", time_token_id),
    )
    checkpoint_signatures = (
        _detached(
            private_keys[Role.GOVERNANCE_ANCHOR], Role.GOVERNANCE_ANCHOR,
            Purpose.GOVERNANCE_CHECKPOINT, signer_keys[0].key_id,
            CHECKPOINT_SIGNATURE_DOMAIN, checkpoint_unsigned,
        ),
        _detached(
            private_keys[Role.CUSTODIAN], Role.CUSTODIAN,
            Purpose.CUSTODIAN_CHECKPOINT, signer_keys[1].key_id,
            CHECKPOINT_SIGNATURE_DOMAIN, checkpoint_unsigned,
        ),
    )
    checkpoint_id = CHECKPOINT_PREFIX + _digest(
        CHECKPOINT_DOMAIN,
        checkpoint_unsigned + (("checkpoint_signatures", tuple(tuple(_signature_pairs(item)) for item in checkpoint_signatures)),),
    )
    checkpoint = verifier_v2.ExternalCustodianCheckpointCandidateV2(
        VERSION, profile_id, campaign_id, attempt_id, 0, None, chain_id, chain_root,
        terminal_event_id, ledger_id, local_terminal_id, subject_id, receipt_id,
        time_token_id, checkpoint_signatures, checkpoint_id,
    )
    return SimpleNamespace(
        profile=profile, store_receipt=store, time_token=time_token,
        checkpoint=checkpoint, private_keys=private_keys,
    )


@pytest.fixture
def valid_graph() -> SimpleNamespace:
    return _build_graph()


def _verify(graph: SimpleNamespace, **changes: object) -> object:
    kwargs = {
        "profile": graph.profile,
        "store_receipt": graph.store_receipt,
        "time_token": graph.time_token,
        "checkpoint": graph.checkpoint,
    }
    kwargs.update(changes)
    return verifier_v2.verify_phase2b_actual_unsealed_960_external_custodian_checkpoint_verifier_mechanics_v2(**kwargs)


def _assert_atomic_rejection(
    value: object,
    reason: verifier_v2.ExternalCustodianCheckpointReasonV2,
) -> None:
    assert type(value) is verifier_v2.ExternalCustodianCheckpointVerifierRejectionV2
    assert value.disposition is verifier_v2.ExternalCustodianCheckpointDispositionV2.REJECTED
    assert value.reason is reason
    for name in RESULT_PREFIX_FIELDS[2:13]:
        assert getattr(value, name) is None
    assert value.required_signature_count == 0
    assert value.verified_signature_count == 0
    assert value.partial_output_published is False
    assert not any(getattr(value, name) for name in (*TRUE_CLAIMS, *FALSE_CLAIMS))


def _independent_result_id(value: object) -> str:
    pairs = tuple(
        (
            field.name,
            getattr(value, field.name).value
            if isinstance(getattr(value, field.name), Enum)
            else getattr(value, field.name),
        )
        for field in fields(type(value))
        if field.name != "result_id"
    )
    return RESULT_PREFIX + _digest(RESULT_DOMAIN, pairs)


def test_public_surface_constants_enums_and_field_manifests_are_exact() -> None:
    assert tuple(verifier_v2.__all__) == PUBLIC_SURFACE
    assert verifier_v2.PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_VERSION == VERSION
    assert verifier_v2.PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_CLAIM_LEVEL == CLAIM_LEVEL
    assert verifier_v2.PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_SCOPE == SCOPE
    assert verifier_v2.PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_SCHEMA_ID == SCHEMA_ID
    assert verifier_v2.PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_POLICY_ID == POLICY_ID
    assert verifier_v2.PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_AUTHORITY_REQUIREMENT_ID == REQUIREMENT_ID
    assert verifier_v2.PHASE2B_ACTUAL_UNSEALED_960_EXTERNAL_CUSTODIAN_CHECKPOINT_VERIFIER_MECHANICS_V2_AUTHORITY_REQUIREMENT_STATEMENTS == REQUIREMENT_STATEMENTS
    assert tuple(item.value for item in Role) == ("GOVERNANCE_ANCHOR", "CUSTODIAN", "STORE_RECEIPT", "TIME_ATTESTOR")
    assert tuple(item.value for item in Purpose) == ("GOVERNANCE_CHECKPOINT", "CUSTODIAN_CHECKPOINT", "STORE_IF_ABSENT_RECEIPT", "DECLARED_TIME_TOKEN")
    assert tuple(item.value for item in verifier_v2.ExternalCustodianCheckpointReasonV2) == (
        "FOUR_ROLE_CANDIDATE_CHECKPOINT_VERIFIED_NOT_PRETRUSTED", "WRONG_INPUT_TYPE",
        "CROSS_VERSION_INPUT", "CROSS_BINDING_INVALID",
        "SIGNATURE_BACKEND_UNAVAILABLE", "SIGNATURE_INVALID", "INTERNAL_ERROR",
    )
    assert tuple(item.name for item in fields(verifier_v2.ExternalCustodianCandidateSignerV2)) == SIGNER_FIELDS
    assert tuple(item.name for item in fields(verifier_v2.ExternalCustodianAuthorityCandidateProfileV2)) == PROFILE_FIELDS
    assert tuple(item.name for item in fields(verifier_v2.ExternalCustodianDetachedSignatureV2)) == SIGNATURE_FIELDS
    assert tuple(item.name for item in fields(verifier_v2.ExternalCustodianStoreReceiptCandidateV2)) == STORE_FIELDS
    assert tuple(item.name for item in fields(verifier_v2.ExternalCustodianTimeTokenCandidateV2)) == TIME_FIELDS
    assert tuple(item.name for item in fields(verifier_v2.ExternalCustodianCheckpointCandidateV2)) == CHECKPOINT_FIELDS
    assert tuple(item.name for item in fields(verifier_v2.ExternalCustodianCheckpointVerifierMechanicsV2)) == RESULT_FIELDS
    assert tuple(item.name for item in fields(verifier_v2.ExternalCustodianCheckpointVerifierRejectionV2)) == REJECTION_FIELDS


def test_authority_requirement_id_is_publicly_recomputable() -> None:
    expected = "phase2b_actual_unsealed_960_external_authority_requirement_v2_" + hashlib.sha256(
        AUTHORITY_REQUIREMENT_DOMAIN + _canonical_json(REQUIREMENT_STATEMENTS).encode("utf-8")
    ).hexdigest()
    assert expected == REQUIREMENT_ID


def test_real_four_key_ed25519_graph_verifies_all_ids_but_no_authority(
    valid_graph: SimpleNamespace,
) -> None:
    value = _verify(valid_graph)
    assert type(value) is verifier_v2.ExternalCustodianCheckpointVerifierMechanicsV2
    assert value.disposition is verifier_v2.ExternalCustodianCheckpointDispositionV2.CANDIDATE_EXTERNAL_CHECKPOINT_MECHANICS_VERIFIED_NOT_EXTERNAL_AUTHORITY
    assert value.reason is verifier_v2.ExternalCustodianCheckpointReasonV2.FOUR_ROLE_CANDIDATE_CHECKPOINT_VERIFIED_NOT_PRETRUSTED
    assert value.version == VERSION and value.schema_id == SCHEMA_ID and value.policy_id == POLICY_ID
    assert value.profile_id == valid_graph.profile.profile_id
    assert value.subject_id == valid_graph.checkpoint.subject_id
    assert value.store_receipt_id == valid_graph.store_receipt.receipt_id
    assert value.time_token_id == valid_graph.time_token.time_token_id
    assert value.checkpoint_id == valid_graph.checkpoint.checkpoint_id
    assert value.required_signature_count == value.verified_signature_count == 4
    assert all(getattr(value, name) is True for name in TRUE_CLAIMS)
    assert all(getattr(value, name) is False for name in FALSE_CLAIMS)
    assert value.result_id == _independent_result_id(value)


def test_attacker_owned_second_registry_also_never_becomes_external_authority() -> None:
    first = _verify(_build_graph("A"))
    second = _verify(_build_graph("B"))
    assert type(first) is type(second) is verifier_v2.ExternalCustodianCheckpointVerifierMechanicsV2
    assert first.profile_id != second.profile_id
    assert first.checkpoint_id != second.checkpoint_id
    for value in (first, second):
        assert all(getattr(value, name) is False for name in FALSE_CLAIMS)


def test_success_and_rejection_outputs_are_privately_issued_and_fresh(valid_graph: SimpleNamespace) -> None:
    with pytest.raises(TypeError, match="privately issued"):
        verifier_v2.ExternalCustodianCheckpointVerifierMechanicsV2()
    with pytest.raises(TypeError, match="privately issued"):
        verifier_v2.ExternalCustodianCheckpointVerifierRejectionV2()
    first = _verify(valid_graph)
    second = _verify(valid_graph)
    reject_a = _verify(valid_graph, profile=None)
    reject_b = _verify(valid_graph, profile=None)
    assert first is not second and first.result_id == second.result_id
    assert reject_a is not reject_b and reject_a == reject_b


@pytest.mark.parametrize("name", ("profile", "store_receipt", "time_token", "checkpoint"))
def test_wrong_top_level_types_reject_atomically(valid_graph: SimpleNamespace, name: str) -> None:
    _assert_atomic_rejection(
        _verify(valid_graph, **{name: None}),
        verifier_v2.ExternalCustodianCheckpointReasonV2.WRONG_INPUT_TYPE,
    )


@pytest.mark.parametrize("name", ("profile", "store_receipt", "time_token", "checkpoint"))
def test_cross_version_each_graph_rejects_atomically(valid_graph: SimpleNamespace, name: str) -> None:
    changed = replace(getattr(valid_graph, name), version=VERSION[:-1] + "3")
    _assert_atomic_rejection(
        _verify(valid_graph, **{name: changed}),
        verifier_v2.ExternalCustodianCheckpointReasonV2.CROSS_VERSION_INPUT,
    )


@pytest.mark.parametrize(
    ("component", "field", "value"),
    (
        ("profile", "required_signature_count", True),
        ("store_receipt", "previous_generation", True),
        ("store_receipt", "detached_signature", None),
        ("time_token", "declared_time_ns", True),
        ("checkpoint", "checkpoint_sequence", True),
        ("checkpoint", "predecessor_checkpoint_id", "not-none"),
        ("checkpoint", "checkpoint_signatures", []),
    ),
)
def test_exact_type_and_none_rules_close_before_content_work(
    valid_graph: SimpleNamespace,
    component: str,
    field: str,
    value: object,
) -> None:
    malformed = replace(getattr(valid_graph, component), **{field: value})
    _assert_atomic_rejection(
        _verify(valid_graph, **{component: malformed}),
        verifier_v2.ExternalCustodianCheckpointReasonV2.WRONG_INPUT_TYPE,
    )


@pytest.mark.parametrize(
    "signers",
    (
        lambda values: values[:3],
        lambda values: (values[1], values[0], values[2], values[3]),
        lambda values: (values[0], values[0], values[2], values[3]),
        lambda values: (values[0], replace(values[1], public_key=values[0].public_key), values[2], values[3]),
        lambda values: (values[0], replace(values[1], key_id=values[0].key_id), values[2], values[3]),
        lambda values: (replace(values[0], key_epoch=2), values[1], values[2], values[3]),
    ),
)
def test_registry_count_order_duplicate_roles_keys_ids_and_epoch_reject(
    valid_graph: SimpleNamespace,
    signers: Callable[[tuple[object, ...]], tuple[object, ...]],
) -> None:
    malformed = replace(valid_graph.profile, signer_keys=signers(valid_graph.profile.signer_keys))
    value = _verify(valid_graph, profile=malformed)
    assert type(value) is verifier_v2.ExternalCustodianCheckpointVerifierRejectionV2
    assert value.reason in {
        verifier_v2.ExternalCustodianCheckpointReasonV2.WRONG_INPUT_TYPE,
        verifier_v2.ExternalCustodianCheckpointReasonV2.CROSS_BINDING_INVALID,
    }
    _assert_atomic_rejection(value, value.reason)


@pytest.mark.parametrize(
    ("component", "changes"),
    (
        ("store_receipt", {"previous_generation": 1}),
        ("store_receipt", {"new_generation": 2}),
        ("store_receipt", {"cas_condition": "UPSERT"}),
        ("store_receipt", {"object_key": "wrong/terminal.json"}),
        ("store_receipt", {"stored_content_sha256": "00" * 32}),
        ("time_token", {"previous_declared_time_ns": 2_000}),
        ("time_token", {"declared_time_ns": 1_000}),
        ("checkpoint", {"checkpoint_sequence": 1}),
        ("checkpoint", {"subject_id": SUBJECT_PREFIX + "00" * 32}),
        ("checkpoint", {"store_receipt_id": STORE_RECEIPT_PREFIX + "00" * 32}),
        ("checkpoint", {"time_token_id": TIME_TOKEN_PREFIX + "00" * 32}),
    ),
)
def test_cas_time_checkpoint_and_crossbindings_reject(
    valid_graph: SimpleNamespace,
    component: str,
    changes: dict[str, object],
) -> None:
    malformed = replace(getattr(valid_graph, component), **changes)
    _assert_atomic_rejection(
        _verify(valid_graph, **{component: malformed}),
        verifier_v2.ExternalCustodianCheckpointReasonV2.CROSS_BINDING_INVALID,
    )


@pytest.mark.parametrize(
    ("component", "field"),
    (
        ("profile", "profile_id"), ("store_receipt", "receipt_id"),
        ("time_token", "time_token_id"), ("checkpoint", "checkpoint_id"),
    ),
)
def test_each_content_address_rejects_rehashed_or_spliced_identity(
    valid_graph: SimpleNamespace,
    component: str,
    field: str,
) -> None:
    prefix = {
        "profile_id": PROFILE_PREFIX, "receipt_id": STORE_RECEIPT_PREFIX,
        "time_token_id": TIME_TOKEN_PREFIX, "checkpoint_id": CHECKPOINT_PREFIX,
    }[field]
    malformed = replace(getattr(valid_graph, component), **{field: prefix + "00" * 32})
    _assert_atomic_rejection(
        _verify(valid_graph, **{component: malformed}),
        verifier_v2.ExternalCustodianCheckpointReasonV2.CROSS_BINDING_INVALID,
    )


@pytest.mark.parametrize("target", ("store", "time", "governance", "custodian"))
def test_each_role_signature_is_cryptographically_required(valid_graph: SimpleNamespace, target: str) -> None:
    if target == "store":
        signature = replace(valid_graph.store_receipt.detached_signature, signature=b"\x00" * 64)
        component = "store_receipt"
        malformed = replace(valid_graph.store_receipt, detached_signature=signature)
    elif target == "time":
        signature = replace(valid_graph.time_token.detached_signature, signature=b"\x00" * 64)
        component = "time_token"
        malformed = replace(valid_graph.time_token, detached_signature=signature)
    else:
        index = 0 if target == "governance" else 1
        signatures = list(valid_graph.checkpoint.checkpoint_signatures)
        signatures[index] = replace(signatures[index], signature=b"\x00" * 64)
        component = "checkpoint"
        malformed = replace(valid_graph.checkpoint, checkpoint_signatures=tuple(signatures))
    _assert_atomic_rejection(
        _verify(valid_graph, **{component: malformed}),
        verifier_v2.ExternalCustodianCheckpointReasonV2.SIGNATURE_INVALID,
    )


def test_cross_campaign_rehashed_splice_is_not_accepted() -> None:
    first = _build_graph("splice-A")
    second = _build_graph("splice-B")
    for changes in (
        {"store_receipt": second.store_receipt},
        {"time_token": second.time_token},
        {"checkpoint": second.checkpoint},
    ):
        _assert_atomic_rejection(
            _verify(first, **changes),
            verifier_v2.ExternalCustodianCheckpointReasonV2.CROSS_BINDING_INVALID,
        )


def test_backend_unavailable_is_an_explicit_atomic_rejection(
    valid_graph: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verifier_v2, "_Ed25519PublicKey", None)
    _assert_atomic_rejection(
        _verify(valid_graph),
        verifier_v2.ExternalCustodianCheckpointReasonV2.SIGNATURE_BACKEND_UNAVAILABLE,
    )


def test_global_malformed_prehash_closure_precedes_hash_and_signature(
    valid_graph: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = replace(valid_graph.checkpoint, checkpoint_sequence=True)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise _PrehashBoundaryReached("malformed graph reached content work")

    monkeypatch.setattr(verifier_v2, "canonical_json", forbidden)
    monkeypatch.setattr(verifier_v2.hashlib, "sha256", forbidden)
    monkeypatch.setattr(verifier_v2, "_verify_signature", forbidden)
    _assert_atomic_rejection(
        _verify(valid_graph, checkpoint=malformed),
        verifier_v2.ExternalCustodianCheckpointReasonV2.WRONG_INPUT_TYPE,
    )


def test_baseexception_at_content_boundary_is_not_false_atomic_normalization(
    valid_graph: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stop(*args: object, **kwargs: object) -> object:
        raise _PrehashBoundaryReached("expected BaseException escape")

    monkeypatch.setattr(verifier_v2, "canonical_json", stop)
    with pytest.raises(_PrehashBoundaryReached, match="expected BaseException escape"):
        _verify(valid_graph)


def test_hash_time_caller_mutation_cannot_change_closed_snapshot(
    valid_graph: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _verify(valid_graph)
    original: Callable[[object], str] = verifier_v2.canonical_json
    calls = 0

    def mutate_on_first_hash(value: object) -> str:
        nonlocal calls
        if calls == 0:
            object.__setattr__(valid_graph.profile, "signer_keys", ())
            object.__setattr__(valid_graph.store_receipt, "object_key", "mutated")
            object.__setattr__(valid_graph.time_token, "declared_time_ns", 0)
            object.__setattr__(valid_graph.checkpoint, "checkpoint_id", CHECKPOINT_PREFIX + "00" * 32)
        calls += 1
        return original(value)

    monkeypatch.setattr(verifier_v2, "canonical_json", mutate_on_first_hash)
    value = _verify(valid_graph)
    assert calls > 0
    assert type(value) is verifier_v2.ExternalCustodianCheckpointVerifierMechanicsV2
    assert value.result_id == baseline.result_id


def test_ordinary_internal_exception_is_fresh_atomic_internal_error(
    valid_graph: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("TEST_ONLY ordinary internal error")

    monkeypatch.setattr(verifier_v2, "_verify_content_and_signatures", explode)
    first = _verify(valid_graph)
    second = _verify(valid_graph)
    _assert_atomic_rejection(first, verifier_v2.ExternalCustodianCheckpointReasonV2.INTERNAL_ERROR)
    _assert_atomic_rejection(second, verifier_v2.ExternalCustodianCheckpointReasonV2.INTERNAL_ERROR)
    assert first is not second


def _ast_imports_and_calls(source: str) -> tuple[frozenset[str], frozenset[str]]:
    tree = ast.parse(source)
    aliases: dict[str, str] = {}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                imported.add(item.name)
                aliases[item.asname or item.name.split(".", 1)[0]] = item.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.add(module)
            for item in node.names:
                aliases[item.asname or item.name] = f"{module}.{item.name}".strip(".")

    def qualified(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            parent = qualified(node.value)
            return node.attr if parent is None else f"{parent}.{node.attr}"
        return None

    calls = {
        name for node in ast.walk(tree) if isinstance(node, ast.Call)
        for name in (qualified(node.func),) if name is not None
    }
    return frozenset(imported), frozenset(calls)


def test_api_is_keyword_only_and_source_ast_has_no_operational_boundaries() -> None:
    signature = inspect.signature(
        verifier_v2.verify_phase2b_actual_unsealed_960_external_custodian_checkpoint_verifier_mechanics_v2
    )
    assert tuple(signature.parameters) == ("profile", "store_receipt", "time_token", "checkpoint")
    assert all(item.kind is inspect.Parameter.KEYWORD_ONLY for item in signature.parameters.values())
    source = Path(verifier_v2.__file__).read_text(encoding="utf-8")
    imported, calls = _ast_imports_and_calls(source)
    forbidden_roots = {
        "asyncio", "datetime", "docker", "fcntl", "http", "multiprocessing", "os",
        "pathlib", "random", "requests", "secrets", "shutil", "socket", "sqlite3",
        "subprocess", "tempfile", "time", "urllib",
    }
    assert not {name.split(".", 1)[0] for name in imported}.intersection(forbidden_roots)
    forbidden_calls = {
        "open", "eval", "exec", "os.open", "os.write", "os.replace", "os.rename",
        "subprocess.run", "subprocess.Popen", "pathlib.Path.read_bytes",
        "pathlib.Path.write_bytes", "requests.get", "requests.post",
    }
    assert not calls.intersection(forbidden_calls)
    forbidden_suffixes = (
        ".from_private_bytes", ".generate", ".sign", ".time", ".time_ns",
        ".monotonic", ".sleep", ".urandom", ".token_bytes",
    )
    assert not {call for call in calls if call.lower().endswith(forbidden_suffixes)}
    assert any(call == "verify" or call.endswith(".verify") for call in calls)
    assert "actual_960_case_run_verified" in source
    assert "execution_authorized" in source
    assert "scoring_performed" in source
