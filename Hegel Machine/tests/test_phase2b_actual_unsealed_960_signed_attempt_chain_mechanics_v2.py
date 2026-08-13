"""Independent tests for the Phase-2B signed attempt-chain mechanics V2.

The fixture uses a real Ed25519 key derived from an explicitly test-only seed.
It proves only that the in-memory content-address and signature mechanics work
under a caller-supplied ``TEST_ONLY`` anchor.  Nothing here supplies a durable
ledger, external custodian, authoritative time, actual-960 run, score, gate,
effect, or C1 evidence.
"""

from __future__ import annotations

import ast
from dataclasses import fields
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

import hegel_machine.phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2 as signed_v2


VERSION = "hegel-machine-phase2b-actual-unsealed-960-signed-attempt-chain-mechanics/2"
CLAIM_LEVEL = "NON_AUTHORITATIVE_TEST_ONLY_SIGNED_ATTEMPT_CHAIN_MECHANICS"
ANCHOR_SCOPE = "TEST_ONLY_SUPPLIED_ANCHOR"
PROTOCOL_ID = "phase2b_protocol_62ad411b5b5a0f912626c54e4bd822a8c585a8f612f5fce6040cce500a11756a"
EXACT_FREEZE_ID = "phase2b_exact_freeze_ffa1fd4fed0b5c2c018803aa9f730b8c85c144efe7e4aa324256681d1c742cbe"

PUBLIC_SURFACE = (
    "PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_VERSION",
    "PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_CLAIM_LEVEL",
    "PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_TEST_ANCHOR_SCOPE_V2",
    "PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_ED25519_BACKEND_AVAILABLE",
    "PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_SCHEMA_ID",
    "PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_POLICY_ID",
    "SignedAttemptSignerRoleV2",
    "SignedAttemptStageV2",
    "SignedAttemptChainDispositionV2",
    "SignedAttemptChainReasonV2",
    "ActualUnsealed960SignerKeyV2",
    "ActualUnsealed960TestOnlyAnchorV2",
    "ActualUnsealed960AttemptEventV2",
    "ActualUnsealed960SignedAttemptChainV2",
    "ActualUnsealed960SignedAttemptChainMechanicsV2",
    "ActualUnsealed960SignedAttemptChainRejectionV2",
    "verify_phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2",
)

KEY_FIELDS = ("role", "public_key", "key_id")
ANCHOR_FIELDS = (
    "version",
    "claim_level",
    "anchor_scope",
    "schema_id",
    "policy_id",
    "protocol_id",
    "exact_freeze_id",
    "execution_freeze_manifest_id",
    "attempt_nonce_sha256",
    "attempt_id",
    "signer_keys",
    "signer_key_ids_root",
    "anchor_root_sha256",
    "anchor_id",
)
EVENT_FIELDS = (
    "version",
    "anchor_id",
    "attempt_id",
    "event_index",
    "stage",
    "predecessor_event_id",
    "payload_root_sha256",
    "signer_key_id",
    "signature",
    "event_id",
)
CHAIN_FIELDS = (
    "version",
    "anchor_id",
    "attempt_id",
    "events",
    "chain_root_sha256",
    "chain_id",
)
TRUE_CLAIMS = (
    "test_only_anchor_scope_enforced",
    "test_only_anchor_content_addresses_verified",
    "test_only_signer_key_content_address_verified",
    "test_only_anchor_ed25519_event_signatures_verified",
    "signed_event_content_addresses_verified",
    "signed_test_chain_contiguous_event_indexes_verified",
    "signed_test_chain_predecessor_links_verified",
    "signed_test_chain_exact_ten_stage_order_verified",
    "signed_test_chain_terminal_consumed_label_verified",
    "signed_test_chain_root_and_id_verified",
    "atomic_fail_closed_rejection_verified",
)
FALSE_CLAIMS = (
    "external_trust_anchor_verified",
    "external_signature_authority_verified",
    "qualified_cryptographic_backend_verified",
    "signature_coverage_verified",
    "authoritative_evidence_chain_verified",
    "custodian_identity_verified",
    "pinned_signer_key_registry_verified",
    "signer_independence_verified",
    "durable_external_one_shot_ledger_verified",
    "append_only_storage_verified",
    "cross_process_fork_prevention_verified",
    "external_timestamp_authority_verified",
    "timeline_observed",
    "authoritative_timeline_order_verified",
    "authoritative_attempt_terminal_state_verified",
    "full_c1_timeline_schema_complete",
    "attempt_registry_complete",
    "terminal_append_receipt_verified",
    "campaign_finalization_verified",
    "retry_authorization_verified",
    "rerun_policy_enforced",
    "one_shot_policy_enforced",
    "actual_evidence_inputs_accepted",
    "admission_evidence_contract_complete",
    "authoritative_evidence_verifier_implemented",
    "admission_ready",
    "execution_authorized",
    "evidence_supplied",
    "evidence_verified",
    "input_archive_authority_verified",
    "prediction_archive_authority_verified",
    "answer_commitment_authority_verified",
    "gate_input_commitment_authority_verified",
    "answer_commitment_opening_verified",
    "gate_input_commitment_opening_verified",
    "pre_reveal_commitment_timing_verified",
    "runtime_attestation_authority_verified",
    "recognizer_executed",
    "runtime_executed",
    "actual_960_case_run_verified",
    "origin_authenticated",
    "scoring_performed",
    "prediction_scored",
    "actual_prediction_scoring_evidence",
    "formal_gate_evaluation_performed",
    "overall_gate_results_materialized",
    "slice_gate_results_materialized",
    "preservation_evaluated",
    "scale_regret_evaluated",
    "bootstrap_evaluated",
    "effect_evidence",
    "formal_c1_report_verified",
    "c1_exit_evidence",
)
SUCCESS_FIELDS = (
    "disposition",
    "reason",
    "version",
    "schema_id",
    "policy_id",
    "claim_level",
    "result_id",
    "anchor_id",
    "attempt_id",
    "signer_key_id",
    "raw_input_committed_payload_root_sha256",
    "package_commitments_frozen_payload_root_sha256",
    "chain_root_sha256",
    "chain_id",
    "event_count",
    "terminal_stage",
    *TRUE_CLAIMS,
    *FALSE_CLAIMS,
)
REJECTION_FIELDS = (
    "disposition",
    "reason",
    "version",
    "schema_id",
    "policy_id",
    "claim_level",
    "validation",
    "result_id",
    "anchor_id",
    "attempt_id",
    "signer_key_id",
    "raw_input_committed_payload_root_sha256",
    "package_commitments_frozen_payload_root_sha256",
    "chain_root_sha256",
    "chain_id",
    "event_count",
    "terminal_stage",
    "partial_output_published",
    *TRUE_CLAIMS,
    *FALSE_CLAIMS,
)

STAGES = tuple(signed_v2.SignedAttemptStageV2)

KEY_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/KEY/V2\x00"
KEY_ROOT_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/KEY_ROOT/V2\x00"
ATTEMPT_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/ATTEMPT/V2\x00"
ANCHOR_ROOT_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/ANCHOR_ROOT/V2\x00"
ANCHOR_ID_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/ANCHOR_ID/V2\x00"
EVENT_SIGNATURE_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/EVENT_SIGNATURE/V2\x00"
EVENT_ID_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/EVENT_ID/V2\x00"
CHAIN_ROOT_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/CHAIN_ROOT/V2\x00"
CHAIN_ID_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/CHAIN_ID/V2\x00"
RESULT_ID_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/RESULT/V2\x00"
SCHEMA_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/SCHEMA/V2\x00"
POLICY_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/POLICY/V2\x00"

KEY_PREFIX = "phase2b_actual_unsealed_960_signer_key_v2_"
ATTEMPT_PREFIX = "phase2b_actual_unsealed_960_attempt_v2_"
ANCHOR_PREFIX = "phase2b_actual_unsealed_960_test_anchor_v2_"
EVENT_PREFIX = "phase2b_actual_unsealed_960_attempt_event_v2_"
CHAIN_PREFIX = "phase2b_actual_unsealed_960_signed_attempt_chain_v2_"
RESULT_PREFIX = "phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2_"
SCHEMA_PREFIX = "phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_schema_v2_"
POLICY_PREFIX = "phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_policy_v2_"
EXPECTED_POLICY_ID = POLICY_PREFIX + "bc9aba86d575035608278044c22731eb29195ef1c32e6e5d585c005882d46b54"


def _canonical_json(value: object) -> str:
    """Independent canonical encoder for the primitive test preimages."""

    def normalize(item: object) -> object:
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, tuple):
            return [normalize(child) for child in item]
        if isinstance(item, list):
            return [normalize(child) for child in item]
        if isinstance(item, dict):
            return {str(key): normalize(child) for key, child in item.items()}
        return item

    return json.dumps(
        normalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(domain: bytes, pairs: tuple[tuple[str, object], ...]) -> str:
    return hashlib.sha256(domain + _canonical_json(pairs).encode("utf-8")).hexdigest()


def _sequence_digest(domain: bytes, identifiers: tuple[str, ...]) -> str:
    framed = bytearray(domain)
    framed.extend(len(identifiers).to_bytes(4, "big"))
    for identifier in identifiers:
        encoded = identifier.encode("ascii")
        framed.extend(len(encoded).to_bytes(2, "big"))
        framed.extend(encoded)
    return hashlib.sha256(bytes(framed)).hexdigest()


def _event_pairs(
    *,
    anchor_id: str,
    attempt_id: str,
    index: int,
    stage: signed_v2.SignedAttemptStageV2,
    predecessor: str | None,
    payload: str,
    key_id: str,
    version: str = VERSION,
) -> tuple[tuple[str, object], ...]:
    return (
        ("version", version),
        ("anchor_id", anchor_id),
        ("attempt_id", attempt_id),
        ("event_index", index),
        ("stage", stage.value),
        ("predecessor_event_id", predecessor),
        ("payload_root_sha256", payload),
        ("signer_key_id", key_id),
    )


def _unchecked_copy(value: object, **changes: object) -> object:
    copied = object.__new__(type(value))
    for item in fields(value):
        object.__setattr__(copied, item.name, changes.get(item.name, getattr(value, item.name)))
    return copied


def _test_private(label: bytes = b"HEGEL-PHASE2B-TEST-ONLY-CUSTODIAN-V2") -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(label).digest())


def _raw_public(private: Ed25519PrivateKey) -> bytes:
    return private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )


def _build_anchor(
    private: Ed25519PrivateKey,
) -> signed_v2.ActualUnsealed960TestOnlyAnchorV2:
    public = _raw_public(private)
    key_pairs = (
        ("role", "CUSTODIAN"),
        ("algorithm", "Ed25519"),
        ("public_key_hex", public.hex()),
    )
    key_id = KEY_PREFIX + _digest(KEY_DOMAIN, key_pairs)
    key = signed_v2.ActualUnsealed960SignerKeyV2(
        role=signed_v2.SignedAttemptSignerRoleV2.CUSTODIAN,
        public_key=public,
        key_id=key_id,
    )
    execution_freeze = "phase2b_execution_freeze_" + hashlib.sha256(b"TEST_ONLY execution freeze").hexdigest()
    attempt_nonce = hashlib.sha256(b"TEST_ONLY attempt nonce before future commitments").hexdigest()
    schema_id = signed_v2.PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_SCHEMA_ID
    policy_id = signed_v2.PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_POLICY_ID
    attempt_pairs = (
        ("schema_id", schema_id),
        ("policy_id", policy_id),
        ("protocol_id", PROTOCOL_ID),
        ("exact_freeze_id", EXACT_FREEZE_ID),
        ("execution_freeze_manifest_id", execution_freeze),
        ("attempt_nonce_sha256", attempt_nonce),
    )
    attempt_id = ATTEMPT_PREFIX + _digest(ATTEMPT_DOMAIN, attempt_pairs)
    key_root = _sequence_digest(KEY_ROOT_DOMAIN, (key_id,))
    anchor_pairs = (
        ("version", VERSION),
        ("claim_level", CLAIM_LEVEL),
        ("anchor_scope", ANCHOR_SCOPE),
        ("schema_id", schema_id),
        ("policy_id", policy_id),
        ("protocol_id", PROTOCOL_ID),
        ("exact_freeze_id", EXACT_FREEZE_ID),
        ("execution_freeze_manifest_id", execution_freeze),
        ("attempt_nonce_sha256", attempt_nonce),
        ("attempt_id", attempt_id),
        ("signer_key_ids", (key_id,)),
        ("signer_key_ids_root", key_root),
    )
    anchor_root = _digest(ANCHOR_ROOT_DOMAIN, anchor_pairs)
    anchor_id = ANCHOR_PREFIX + _digest(
        ANCHOR_ID_DOMAIN,
        (*anchor_pairs, ("anchor_root_sha256", anchor_root)),
    )
    return signed_v2.ActualUnsealed960TestOnlyAnchorV2(
        version=VERSION,
        claim_level=CLAIM_LEVEL,
        anchor_scope=ANCHOR_SCOPE,
        schema_id=schema_id,
        policy_id=policy_id,
        protocol_id=PROTOCOL_ID,
        exact_freeze_id=EXACT_FREEZE_ID,
        execution_freeze_manifest_id=execution_freeze,
        attempt_nonce_sha256=attempt_nonce,
        attempt_id=attempt_id,
        signer_keys=(key,),
        signer_key_ids_root=key_root,
        anchor_root_sha256=anchor_root,
        anchor_id=anchor_id,
    )


def _default_payloads(
    anchor: signed_v2.ActualUnsealed960TestOnlyAnchorV2,
) -> tuple[str, ...]:
    return (
        anchor.anchor_root_sha256,
        hashlib.sha256(b"TEST_ONLY raw input commitment introduced after preregistration").hexdigest(),
        hashlib.sha256(b"TEST_ONLY package commitments introduced after raw input").hexdigest(),
        *(hashlib.sha256(f"TEST_ONLY stage payload {index}".encode()).hexdigest() for index in range(3, 10)),
    )


def _build_chain(
    anchor: signed_v2.ActualUnsealed960TestOnlyAnchorV2,
    private: Ed25519PrivateKey,
    *,
    stages: tuple[signed_v2.SignedAttemptStageV2, ...] = STAGES,
    payloads: tuple[str, ...] | None = None,
    signer_ids: tuple[str, ...] | None = None,
    signing_keys: tuple[Ed25519PrivateKey, ...] | None = None,
    signature_overrides: dict[int, bytes] | None = None,
    predecessor_overrides: dict[int, str | None] | None = None,
    version: str = VERSION,
) -> signed_v2.ActualUnsealed960SignedAttemptChainV2:
    payload_values = _default_payloads(anchor) if payloads is None else payloads
    key_ids = (anchor.signer_keys[0].key_id,) * len(stages) if signer_ids is None else signer_ids
    keys = (private,) * len(stages) if signing_keys is None else signing_keys
    signature_overrides = {} if signature_overrides is None else signature_overrides
    predecessor_overrides = {} if predecessor_overrides is None else predecessor_overrides
    events: list[signed_v2.ActualUnsealed960AttemptEventV2] = []
    prior: str | None = None
    for index, stage in enumerate(stages):
        predecessor = predecessor_overrides.get(index, prior)
        pairs = _event_pairs(
            anchor_id=anchor.anchor_id,
            attempt_id=anchor.attempt_id,
            index=index,
            stage=stage,
            predecessor=predecessor,
            payload=payload_values[index],
            key_id=key_ids[index],
            version=version,
        )
        preimage = EVENT_SIGNATURE_DOMAIN + _canonical_json(pairs).encode("utf-8")
        signature = signature_overrides.get(index, keys[index].sign(preimage))
        event_id = EVENT_PREFIX + _digest(
            EVENT_ID_DOMAIN,
            (*pairs, ("signature_hex", signature.hex())),
        )
        event = signed_v2.ActualUnsealed960AttemptEventV2(
            version=version,
            anchor_id=anchor.anchor_id,
            attempt_id=anchor.attempt_id,
            event_index=index,
            stage=stage,
            predecessor_event_id=predecessor,
            payload_root_sha256=payload_values[index],
            signer_key_id=key_ids[index],
            signature=signature,
            event_id=event_id,
        )
        events.append(event)
        prior = event_id
    event_ids = tuple(event.event_id for event in events)
    chain_root = _sequence_digest(CHAIN_ROOT_DOMAIN, event_ids)
    chain_pairs = (
        ("version", version),
        ("anchor_id", anchor.anchor_id),
        ("attempt_id", anchor.attempt_id),
        ("event_ids", event_ids),
        ("chain_root_sha256", chain_root),
    )
    chain_id = CHAIN_PREFIX + _digest(CHAIN_ID_DOMAIN, chain_pairs)
    return signed_v2.ActualUnsealed960SignedAttemptChainV2(
        version=version,
        anchor_id=anchor.anchor_id,
        attempt_id=anchor.attempt_id,
        events=tuple(events),
        chain_root_sha256=chain_root,
        chain_id=chain_id,
    )


@pytest.fixture
def valid_graph() -> SimpleNamespace:
    private = _test_private()
    anchor = _build_anchor(private)
    chain = _build_chain(anchor, private)
    return SimpleNamespace(private=private, anchor=anchor, chain=chain)


class _PrehashBoundaryReached(BaseException):
    """Must not be caught by the ordinary internal-error normalizer."""


def _assert_atomic_rejection(
    result: object,
    reason: signed_v2.SignedAttemptChainReasonV2,
) -> None:
    assert type(result) is signed_v2.ActualUnsealed960SignedAttemptChainRejectionV2
    assert result.disposition is signed_v2.SignedAttemptChainDispositionV2.REJECTED
    assert result.reason is reason
    assert result.validation is None
    for name in (
        "result_id",
        "anchor_id",
        "attempt_id",
        "signer_key_id",
        "raw_input_committed_payload_root_sha256",
        "package_commitments_frozen_payload_root_sha256",
        "chain_root_sha256",
        "chain_id",
        "terminal_stage",
    ):
        assert getattr(result, name) is None
    assert result.event_count == 0
    assert result.partial_output_published is False
    assert all(getattr(result, name) is False for name in (*TRUE_CLAIMS, *FALSE_CLAIMS))


def _verify(anchor: object, chain: object) -> object:
    return signed_v2.verify_phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2(
        anchor=anchor,  # type: ignore[arg-type]
        chain=chain,  # type: ignore[arg-type]
    )


def test_exact_public_surface_fields_enums_and_call_signature() -> None:
    assert tuple(signed_v2.__all__) == PUBLIC_SURFACE
    manifests = (
        (signed_v2.ActualUnsealed960SignerKeyV2, KEY_FIELDS),
        (signed_v2.ActualUnsealed960TestOnlyAnchorV2, ANCHOR_FIELDS),
        (signed_v2.ActualUnsealed960AttemptEventV2, EVENT_FIELDS),
        (signed_v2.ActualUnsealed960SignedAttemptChainV2, CHAIN_FIELDS),
        (signed_v2.ActualUnsealed960SignedAttemptChainMechanicsV2, SUCCESS_FIELDS),
        (signed_v2.ActualUnsealed960SignedAttemptChainRejectionV2, REJECTION_FIELDS),
    )
    for cls, expected in manifests:
        assert tuple(item.name for item in fields(cls)) == expected
    assert tuple(item.value for item in signed_v2.SignedAttemptSignerRoleV2) == ("CUSTODIAN",)
    assert tuple(item.value for item in signed_v2.SignedAttemptStageV2) == (
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
    )
    assert tuple(item.value for item in signed_v2.SignedAttemptChainDispositionV2) == (
        "SIGNED_ATTEMPT_CHAIN_MECHANICS_VERIFIED_NOT_DURABLE_AUTHORITY",
        "REJECTED",
    )
    assert tuple(item.value for item in signed_v2.SignedAttemptChainReasonV2) == (
        "TEST_ONLY_TEN_STAGE_ED25519_CHAIN_VERIFIED",
        "WRONG_INPUT_TYPE",
        "CROSS_VERSION_INPUT",
        "TEST_ANCHOR_INVALID",
        "CHAIN_INVALID",
        "SIGNATURE_BACKEND_UNAVAILABLE",
        "SIGNATURE_INVALID",
        "INTERNAL_ERROR",
    )
    signature = inspect.signature(
        signed_v2.verify_phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2
    )
    assert tuple(signature.parameters) == ("anchor", "chain")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )


def test_schema_policy_are_independently_recomputed_and_claim_boundary_is_exact() -> None:
    assert signed_v2.PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_VERSION == VERSION
    assert signed_v2.PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_CLAIM_LEVEL == CLAIM_LEVEL
    assert signed_v2.PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_TEST_ANCHOR_SCOPE_V2 == ANCHOR_SCOPE
    field_type_manifests = (
        (
            "ActualUnsealed960SignerKeyV2",
            (
                ("role", "exact_SignedAttemptSignerRoleV2"),
                ("public_key", "exact_bytes_len_32"),
                ("key_id", "exact_prefixed_id"),
            ),
        ),
        (
            "ActualUnsealed960TestOnlyAnchorV2",
            tuple(
                (name, "exact_tuple_len_1")
                if name == "signer_keys"
                else (name, "exact_lowercase_sha256")
                if name in {"attempt_nonce_sha256", "signer_key_ids_root", "anchor_root_sha256"}
                else (name, "exact_prefixed_id")
                if name.endswith("_id")
                else (name, "exact_utf8_text")
                for name in ANCHOR_FIELDS
            ),
        ),
        (
            "ActualUnsealed960AttemptEventV2",
            tuple(
                (name, "exact_int_0_through_9")
                if name == "event_index"
                else (name, "exact_SignedAttemptStageV2")
                if name == "stage"
                else (name, "exact_optional_prefixed_id")
                if name == "predecessor_event_id"
                else (name, "exact_bytes_len_64")
                if name == "signature"
                else (name, "exact_lowercase_sha256")
                if name == "payload_root_sha256"
                else (name, "exact_prefixed_id")
                if name.endswith("_id")
                else (name, "exact_utf8_text")
                for name in EVENT_FIELDS
            ),
        ),
        (
            "ActualUnsealed960SignedAttemptChainV2",
            tuple(
                (name, "exact_tuple_len_10")
                if name == "events"
                else (name, "exact_lowercase_sha256")
                if name == "chain_root_sha256"
                else (name, "exact_prefixed_id")
                if name.endswith("_id")
                else (name, "exact_utf8_text")
                for name in CHAIN_FIELDS
            ),
        ),
    )
    schema_preimage = (
        ("version", VERSION),
        ("claim_level", CLAIM_LEVEL),
        ("key_fields", KEY_FIELDS),
        ("anchor_fields", ANCHOR_FIELDS),
        ("event_fields", EVENT_FIELDS),
        ("chain_fields", CHAIN_FIELDS),
        ("success_fields", SUCCESS_FIELDS),
        ("rejection_fields", REJECTION_FIELDS),
        ("field_type_manifests", field_type_manifests),
        ("signer_role_values", tuple(item.value for item in signed_v2.SignedAttemptSignerRoleV2)),
        ("stage_values", tuple(item.value for item in signed_v2.SignedAttemptStageV2)),
        ("disposition_values", tuple(item.value for item in signed_v2.SignedAttemptChainDispositionV2)),
        ("reason_values", tuple(item.value for item in signed_v2.SignedAttemptChainReasonV2)),
        (
            "caps",
            (
                ("maximum_utf8_bytes", 4096),
                ("exact_event_count", 10),
                ("raw_public_key_bytes", 32),
                ("raw_signature_bytes", 64),
            ),
        ),
        (
            "public_operation_signature",
            (
                "verify_phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2",
                "keyword_only_anchor_exact_ActualUnsealed960TestOnlyAnchorV2",
                "keyword_only_chain_exact_ActualUnsealed960SignedAttemptChainV2",
            ),
        ),
    )
    expected_schema = SCHEMA_PREFIX + _digest(SCHEMA_DOMAIN, schema_preimage)
    assert signed_v2.PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_SCHEMA_ID == expected_schema
    policy_preimage = (
        ("version", VERSION),
        ("claim_level", CLAIM_LEVEL),
        ("anchor_scope", ANCHOR_SCOPE),
        ("frozen_protocol_id", PROTOCOL_ID),
        ("frozen_exact_freeze_id", EXACT_FREEZE_ID),
        ("stage_order", tuple(item.value for item in STAGES)),
        ("true_claims", TRUE_CLAIMS),
        ("false_claims", FALSE_CLAIMS),
        ("schema_id", expected_schema),
        (
            "domains",
            tuple(
                item.hex()
                for item in (
                    KEY_DOMAIN,
                    KEY_ROOT_DOMAIN,
                    ATTEMPT_DOMAIN,
                    ANCHOR_ROOT_DOMAIN,
                    ANCHOR_ID_DOMAIN,
                    EVENT_SIGNATURE_DOMAIN,
                    EVENT_ID_DOMAIN,
                    CHAIN_ROOT_DOMAIN,
                    CHAIN_ID_DOMAIN,
                    RESULT_ID_DOMAIN,
                    SCHEMA_DOMAIN,
                    POLICY_DOMAIN,
                )
            ),
        ),
        (
            "prefixes",
            (
                KEY_PREFIX,
                ATTEMPT_PREFIX,
                ANCHOR_PREFIX,
                EVENT_PREFIX,
                CHAIN_PREFIX,
                RESULT_PREFIX,
                SCHEMA_PREFIX,
                POLICY_PREFIX,
            ),
        ),
        (
            "id_prefix_rules",
            (
                ("key_id", KEY_PREFIX),
                ("attempt_id", ATTEMPT_PREFIX),
                ("anchor_id", ANCHOR_PREFIX),
                ("event_id", EVENT_PREFIX),
                ("predecessor_event_id", EVENT_PREFIX),
                ("chain_id", CHAIN_PREFIX),
                ("result_id", RESULT_PREFIX),
                ("schema_id", SCHEMA_PREFIX),
                ("policy_id", POLICY_PREFIX),
                ("protocol_id", "phase2b_protocol_"),
                ("exact_freeze_id", "phase2b_exact_freeze_"),
                ("execution_freeze_manifest_id", "phase2b_execution_freeze_"),
            ),
        ),
        (
            "content_address_formula",
            "prefix || lowercase_hex_sha256(domain_bytes || utf8(canonical_json(tuple_of_declared_order_name_value_pairs)))",
        ),
        ("key_id_preimage_fields", ("role", "algorithm=Ed25519", "public_key_hex")),
        (
            "attempt_id_preimage_fields",
            (
                "schema_id",
                "policy_id",
                "protocol_id",
                "exact_freeze_id",
                "execution_freeze_manifest_id",
                "attempt_nonce_sha256",
            ),
        ),
        (
            "anchor_root_preimage_fields",
            (
                "version",
                "claim_level",
                "anchor_scope",
                "schema_id",
                "policy_id",
                "protocol_id",
                "exact_freeze_id",
                "execution_freeze_manifest_id",
                "attempt_nonce_sha256",
                "attempt_id",
                "signer_key_ids",
                "signer_key_ids_root",
            ),
        ),
        ("anchor_id_additional_preimage_field", "anchor_root_sha256"),
        (
            "unsigned_event_preimage_fields",
            (
                "version",
                "anchor_id",
                "attempt_id",
                "event_index",
                "stage",
                "predecessor_event_id",
                "payload_root_sha256",
                "signer_key_id",
            ),
        ),
        ("event_id_additional_preimage_field", "signature_hex"),
        (
            "chain_id_preimage_fields",
            ("version", "anchor_id", "attempt_id", "event_ids", "chain_root_sha256"),
        ),
        (
            "result_id_preimage_fields",
            (
                "version",
                "schema_id",
                "policy_id",
                "anchor_id",
                "attempt_id",
                "signer_key_id",
                "raw_input_committed_payload_root_sha256",
                "package_commitments_frozen_payload_root_sha256",
                "chain_root_sha256",
                "chain_id",
                "event_count",
                "terminal_stage",
            ),
        ),
        (
            "success_semantics",
            (
                "SIGNED_ATTEMPT_CHAIN_MECHANICS_VERIFIED_NOT_DURABLE_AUTHORITY",
                "TEST_ONLY_TEN_STAGE_ED25519_CHAIN_VERIFIED",
                "all_true_claims_exact_true",
                "all_false_claims_exact_false",
            ),
        ),
        (
            "ordered_root_formula",
            "lowercase_hex_sha256(domain_bytes || u32_big_endian_count || repeated_u16_big_endian_length_plus_ascii_id)",
        ),
        (
            "event_signature_formula",
            "Ed25519_sign(domain_bytes || utf8(canonical_json(exact_unsigned_event_named_pairs))); signature_and_event_id_excluded",
        ),
        (
            "event_id_formula",
            "event_prefix || lowercase_hex_sha256(event_id_domain || utf8(canonical_json(unsigned_event_named_pairs_plus_signature_hex)))",
        ),
        (
            "attempt_preregistration_formula",
            "attempt_id_binds_schema_policy_protocol_exact_freeze_execution_freeze_and_attempt_nonce_sha256_no_future_input_or_package_values",
        ),
        (
            "payload_introduction_rules",
            (
                "PREREGISTERED_payload_equals_test_anchor_root",
                "RAW_INPUT_COMMITTED_payload_introduces_raw_input_commitment_root",
                "PACKAGE_COMMITMENTS_FROZEN_payload_introduces_package_commitments_root",
                "all_later_stage_payloads_are_signed_opaque_sha256_roots_only",
            ),
        ),
        (
            "rules",
            (
                "complete_anchor_and_ten_event_chain_exact_type_format_cap_snapshot_before_any_operation_hash_or_signature",
                "exact_one_test_only_custodian_key_raw32_and_ed25519_signature_raw64",
                "event_signature_excludes_signature_and_event_id",
                "event_id_binds_signature_hex",
                "exact_ten_contiguous_events_and_predecessor_chain",
                "no_future_value_placeholder_in_test_anchor_or_attempt_id",
                "cryptography_backend_unavailable_fails_closed",
                "atomic_rejection_covers_exact_input_and_ordinary_backend_failures_not_injected_BaseException_or_process_termination",
                "caller_cannot_supply_signature_verifier_or_signature_validity_boolean",
                "verification_performs_no_signature_or_key_generation",
                "no_external_authority_durability_time_execution_scoring_gate_effect_or_c1_claim",
            ),
        ),
    )
    expected_policy = POLICY_PREFIX + _digest(POLICY_DOMAIN, policy_preimage)
    assert expected_policy == EXPECTED_POLICY_ID
    assert signed_v2.PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_POLICY_ID == expected_policy


def test_positive_real_ed25519_ten_stage_chain_and_all_content_addresses(
    valid_graph: SimpleNamespace,
) -> None:
    anchor = valid_graph.anchor
    chain = valid_graph.chain
    public = valid_graph.private.public_key()
    assert anchor.anchor_scope == "TEST_ONLY_SUPPLIED_ANCHOR"
    assert not hasattr(anchor, "input_archive_sha256")
    assert not hasattr(anchor, "package_commitments_root_sha256")
    assert len(chain.events) == 10
    prior = None
    for index, (stage, event) in enumerate(zip(STAGES, chain.events, strict=True)):
        pairs = _event_pairs(
            anchor_id=anchor.anchor_id,
            attempt_id=anchor.attempt_id,
            index=index,
            stage=stage,
            predecessor=prior,
            payload=event.payload_root_sha256,
            key_id=anchor.signer_keys[0].key_id,
        )
        preimage = EVENT_SIGNATURE_DOMAIN + _canonical_json(pairs).encode("utf-8")
        public.verify(event.signature, preimage)
        expected_id = EVENT_PREFIX + _digest(
            EVENT_ID_DOMAIN,
            (*pairs, ("signature_hex", event.signature.hex())),
        )
        assert event.event_id == expected_id
        assert event.predecessor_event_id == prior
        prior = expected_id
    event_ids = tuple(item.event_id for item in chain.events)
    expected_root = _sequence_digest(CHAIN_ROOT_DOMAIN, event_ids)
    assert chain.chain_root_sha256 == expected_root
    expected_chain_id = CHAIN_PREFIX + _digest(
        CHAIN_ID_DOMAIN,
        (
            ("version", VERSION),
            ("anchor_id", anchor.anchor_id),
            ("attempt_id", anchor.attempt_id),
            ("event_ids", event_ids),
            ("chain_root_sha256", expected_root),
        ),
    )
    assert chain.chain_id == expected_chain_id
    result = _verify(anchor, chain)
    assert type(result) is signed_v2.ActualUnsealed960SignedAttemptChainMechanicsV2
    assert result.disposition is signed_v2.SignedAttemptChainDispositionV2.SIGNED_ATTEMPT_CHAIN_MECHANICS_VERIFIED_NOT_DURABLE_AUTHORITY
    assert result.reason is signed_v2.SignedAttemptChainReasonV2.TEST_ONLY_TEN_STAGE_ED25519_CHAIN_VERIFIED
    assert result.event_count == 10
    assert result.terminal_stage is signed_v2.SignedAttemptStageV2.TERMINAL_CONSUMED
    assert result.raw_input_committed_payload_root_sha256 == chain.events[1].payload_root_sha256
    assert result.package_commitments_frozen_payload_root_sha256 == chain.events[2].payload_root_sha256
    assert all(getattr(result, name) is True for name in TRUE_CLAIMS)
    assert all(getattr(result, name) is False for name in FALSE_CLAIMS)
    expected_result_id = RESULT_PREFIX + _digest(
        RESULT_ID_DOMAIN,
        (
            ("version", VERSION),
            ("schema_id", result.schema_id),
            ("policy_id", result.policy_id),
            ("anchor_id", anchor.anchor_id),
            ("attempt_id", anchor.attempt_id),
            ("signer_key_id", anchor.signer_keys[0].key_id),
            ("raw_input_committed_payload_root_sha256", chain.events[1].payload_root_sha256),
            ("package_commitments_frozen_payload_root_sha256", chain.events[2].payload_root_sha256),
            ("chain_root_sha256", expected_root),
            ("chain_id", expected_chain_id),
            ("event_count", 10),
            ("terminal_stage", "TERMINAL_CONSUMED"),
        ),
    )
    assert result.result_id == expected_result_id


@pytest.mark.parametrize(
    "field,value",
    [
        ("anchor_scope", "EXTERNAL_AUTHORITY"),
        ("claim_level", CLAIM_LEVEL + "-drift"),
        ("schema_id", SCHEMA_PREFIX + "0" * 64),
        ("policy_id", POLICY_PREFIX + "0" * 64),
        ("protocol_id", PROTOCOL_ID[:-1] + "0"),
        ("signer_key_ids_root", "0" * 64),
        ("anchor_root_sha256", "0" * 64),
        ("anchor_id", ANCHOR_PREFIX + "0" * 64),
    ],
)
def test_test_anchor_tamper_rejects_with_anchor_specific_reason(
    valid_graph: SimpleNamespace,
    field: str,
    value: object,
) -> None:
    anchor = _unchecked_copy(valid_graph.anchor, **{field: value})
    _assert_atomic_rejection(
        _verify(anchor, valid_graph.chain),
        signed_v2.SignedAttemptChainReasonV2.TEST_ANCHOR_INVALID,
    )


def test_test_anchor_key_role_and_key_content_address_are_exact(
    valid_graph: SimpleNamespace,
) -> None:
    key = valid_graph.anchor.signer_keys[0]
    malformed_keys = (
        _unchecked_copy(key, role="CUSTODIAN"),
        _unchecked_copy(key, public_key=key.public_key[:-1]),
        _unchecked_copy(key, key_id=KEY_PREFIX + "0" * 64),
    )
    for malformed in malformed_keys:
        anchor = _unchecked_copy(valid_graph.anchor, signer_keys=(malformed,))
        _assert_atomic_rejection(
            _verify(anchor, valid_graph.chain),
            signed_v2.SignedAttemptChainReasonV2.TEST_ANCHOR_INVALID,
        )


@pytest.mark.parametrize("attack", ["reorder", "duplicate_stage", "missing", "duplicate_event"])
def test_rehashed_reorder_duplicate_and_missing_events_fail_closed(
    valid_graph: SimpleNamespace,
    attack: str,
) -> None:
    anchor = valid_graph.anchor
    private = valid_graph.private
    if attack == "reorder":
        stages = (STAGES[0], STAGES[2], STAGES[1], *STAGES[3:])
        payloads = _default_payloads(anchor)
        payloads = (payloads[0], payloads[2], payloads[1], *payloads[3:])
        chain = _build_chain(anchor, private, stages=stages, payloads=payloads)
    elif attack == "duplicate_stage":
        stages = (*STAGES[:2], STAGES[1], *STAGES[3:])
        chain = _build_chain(anchor, private, stages=stages)
    elif attack == "missing":
        chain = _unchecked_copy(valid_graph.chain, events=valid_graph.chain.events[:-1])
    else:
        duplicated = (*valid_graph.chain.events[:-1], valid_graph.chain.events[-2])
        chain = _unchecked_copy(valid_graph.chain, events=duplicated)
    _assert_atomic_rejection(
        _verify(anchor, chain),
        signed_v2.SignedAttemptChainReasonV2.CHAIN_INVALID,
    )


def test_rehashed_wrong_predecessor_fails_closed(valid_graph: SimpleNamespace) -> None:
    chain = _build_chain(
        valid_graph.anchor,
        valid_graph.private,
        predecessor_overrides={4: valid_graph.chain.events[1].event_id},
    )
    _assert_atomic_rejection(
        _verify(valid_graph.anchor, chain),
        signed_v2.SignedAttemptChainReasonV2.CHAIN_INVALID,
    )


def test_rehashed_preregistered_payload_must_bind_anchor_root(
    valid_graph: SimpleNamespace,
) -> None:
    payloads = list(_default_payloads(valid_graph.anchor))
    payloads[0] = hashlib.sha256(b"TEST_ONLY wrong preregistration payload").hexdigest()
    chain = _build_chain(
        valid_graph.anchor,
        valid_graph.private,
        payloads=tuple(payloads),
    )
    _assert_atomic_rejection(
        _verify(valid_graph.anchor, chain),
        signed_v2.SignedAttemptChainReasonV2.CHAIN_INVALID,
    )


@pytest.mark.parametrize("index", [1, 2])
def test_raw_input_and_package_commitments_are_introduced_only_by_signed_events(
    valid_graph: SimpleNamespace,
    index: int,
) -> None:
    payloads = list(_default_payloads(valid_graph.anchor))
    introduced = hashlib.sha256(f"TEST_ONLY newly introduced payload {index}".encode()).hexdigest()
    payloads[index] = introduced
    chain = _build_chain(valid_graph.anchor, valid_graph.private, payloads=tuple(payloads))
    result = _verify(valid_graph.anchor, chain)
    assert type(result) is signed_v2.ActualUnsealed960SignedAttemptChainMechanicsV2
    if index == 1:
        assert result.raw_input_committed_payload_root_sha256 == introduced
    else:
        assert result.package_commitments_frozen_payload_root_sha256 == introduced


def test_rehashed_unanchored_key_id_and_role_fail_closed(valid_graph: SimpleNamespace) -> None:
    alternate = _test_private(b"HEGEL-PHASE2B-TEST-ONLY-UNANCHORED-V2")
    public = _raw_public(alternate)
    alternate_id = KEY_PREFIX + _digest(
        KEY_DOMAIN,
        (
            ("role", "CUSTODIAN"),
            ("algorithm", "Ed25519"),
            ("public_key_hex", public.hex()),
        ),
    )
    chain = _build_chain(
        valid_graph.anchor,
        valid_graph.private,
        signer_ids=(alternate_id,) * 10,
        signing_keys=(alternate,) * 10,
    )
    _assert_atomic_rejection(
        _verify(valid_graph.anchor, chain),
        signed_v2.SignedAttemptChainReasonV2.CHAIN_INVALID,
    )


def test_signature_invalid_is_distinct_after_signature_and_ids_are_rehashed(
    valid_graph: SimpleNamespace,
) -> None:
    chain = _build_chain(
        valid_graph.anchor,
        valid_graph.private,
        signature_overrides={0: b"\x00" * 64},
    )
    _assert_atomic_rejection(
        _verify(valid_graph.anchor, chain),
        signed_v2.SignedAttemptChainReasonV2.SIGNATURE_INVALID,
    )
    short = _unchecked_copy(valid_graph.chain.events[-1], signature=b"\x00" * 63)
    malformed = _unchecked_copy(
        valid_graph.chain,
        events=(*valid_graph.chain.events[:-1], short),
    )
    _assert_atomic_rejection(
        _verify(valid_graph.anchor, malformed),
        signed_v2.SignedAttemptChainReasonV2.CHAIN_INVALID,
    )


@pytest.mark.parametrize("which", ["anchor", "chain", "both"])
def test_cross_version_inputs_reject_atomically(valid_graph: SimpleNamespace, which: str) -> None:
    anchor = valid_graph.anchor
    chain = valid_graph.chain
    if which in {"anchor", "both"}:
        anchor = _unchecked_copy(anchor, version=VERSION + "-cross-version")
    if which in {"chain", "both"}:
        chain = _unchecked_copy(chain, version=VERSION + "-cross-version")
    _assert_atomic_rejection(
        _verify(anchor, chain),
        signed_v2.SignedAttemptChainReasonV2.CROSS_VERSION_INPUT,
    )


def test_nested_event_cross_version_rejects_as_cross_version(valid_graph: SimpleNamespace) -> None:
    event = _unchecked_copy(valid_graph.chain.events[-1], version=VERSION + "-cross-version")
    chain = _unchecked_copy(
        valid_graph.chain,
        events=(*valid_graph.chain.events[:-1], event),
    )
    _assert_atomic_rejection(
        _verify(valid_graph.anchor, chain),
        signed_v2.SignedAttemptChainReasonV2.CROSS_VERSION_INPUT,
    )


@pytest.mark.parametrize("wrong", [None, object(), (), [], {}, 0, False, "wire"])
def test_wrong_top_level_types_reject_atomically(
    valid_graph: SimpleNamespace,
    wrong: object,
) -> None:
    _assert_atomic_rejection(
        _verify(wrong, valid_graph.chain),
        signed_v2.SignedAttemptChainReasonV2.WRONG_INPUT_TYPE,
    )
    _assert_atomic_rejection(
        _verify(valid_graph.anchor, wrong),
        signed_v2.SignedAttemptChainReasonV2.WRONG_INPUT_TYPE,
    )


def test_top_and_nested_subclasses_are_not_accepted(valid_graph: SimpleNamespace) -> None:
    class AnchorSubclass(signed_v2.ActualUnsealed960TestOnlyAnchorV2):
        pass

    class ChainSubclass(signed_v2.ActualUnsealed960SignedAttemptChainV2):
        pass

    _assert_atomic_rejection(
        _verify(object.__new__(AnchorSubclass), valid_graph.chain),
        signed_v2.SignedAttemptChainReasonV2.WRONG_INPUT_TYPE,
    )
    _assert_atomic_rejection(
        _verify(valid_graph.anchor, object.__new__(ChainSubclass)),
        signed_v2.SignedAttemptChainReasonV2.WRONG_INPUT_TYPE,
    )
    class EventSubclass(signed_v2.ActualUnsealed960AttemptEventV2):
        pass

    nested = _unchecked_copy(
        valid_graph.chain,
        events=(*valid_graph.chain.events[:-1], object.__new__(EventSubclass)),
    )
    _assert_atomic_rejection(
        _verify(valid_graph.anchor, nested),
        signed_v2.SignedAttemptChainReasonV2.CHAIN_INVALID,
    )


@pytest.mark.parametrize(
    "malformation",
    ["late_event_signature_type", "late_event_payload_utf8", "chain_root_type"],
)
def test_global_shape_preflight_rejects_before_any_hash_or_signature(
    valid_graph: SimpleNamespace,
    malformation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = valid_graph.chain.events[-1]
    if malformation == "late_event_signature_type":
        final = _unchecked_copy(final, signature=bytearray(final.signature))
        chain = _unchecked_copy(valid_graph.chain, events=(*valid_graph.chain.events[:-1], final))
    elif malformation == "late_event_payload_utf8":
        final = _unchecked_copy(final, payload_root_sha256="\ud800")
        chain = _unchecked_copy(valid_graph.chain, events=(*valid_graph.chain.events[:-1], final))
    else:
        chain = _unchecked_copy(valid_graph.chain, chain_root_sha256=0)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise _PrehashBoundaryReached(f"{malformation} reached hash/signature")

    monkeypatch.setattr(signed_v2, "_digest", forbidden)
    monkeypatch.setattr(signed_v2, "_sequence_digest", forbidden)
    monkeypatch.setattr(signed_v2, "_verify_ed25519", forbidden)
    _assert_atomic_rejection(
        _verify(valid_graph.anchor, chain),
        signed_v2.SignedAttemptChainReasonV2.CHAIN_INVALID,
    )


def test_prehash_baseexception_sentinel_is_not_normalized(
    valid_graph: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reached(*args: object, **kwargs: object) -> object:
        raise _PrehashBoundaryReached("expected boundary escape")

    assert issubclass(_PrehashBoundaryReached, BaseException)
    assert not issubclass(_PrehashBoundaryReached, Exception)
    monkeypatch.setattr(signed_v2, "_digest", reached)
    with pytest.raises(_PrehashBoundaryReached, match="expected boundary escape"):
        _verify(valid_graph.anchor, valid_graph.chain)


def test_closed_snapshot_resists_hash_time_caller_mutation(
    valid_graph: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchor = _unchecked_copy(valid_graph.anchor)
    events = tuple(_unchecked_copy(event) for event in valid_graph.chain.events)
    chain = _unchecked_copy(valid_graph.chain, events=events)
    original: Callable[..., str] = signed_v2._digest
    calls = 0

    def mutate_callers_on_first_hash(*args: object, **kwargs: object) -> str:
        nonlocal calls
        if calls == 0:
            object.__setattr__(anchor, "anchor_scope", "MUTATED_AFTER_SNAPSHOT")
            object.__setattr__(anchor.signer_keys[0], "public_key", b"\x00" * 32)
            object.__setattr__(events[-1], "signature", b"\x00" * 64)
            object.__setattr__(chain, "events", ())
            object.__setattr__(chain, "chain_id", CHAIN_PREFIX + "0" * 64)
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(signed_v2, "_digest", mutate_callers_on_first_hash)
    result = _verify(anchor, chain)
    assert calls > 0
    assert type(result) is signed_v2.ActualUnsealed960SignedAttemptChainMechanicsV2
    assert result.chain_id == valid_graph.chain.chain_id
    assert chain.events == ()


def test_success_and_rejection_results_are_fresh(valid_graph: SimpleNamespace) -> None:
    first = _verify(valid_graph.anchor, valid_graph.chain)
    second = _verify(valid_graph.anchor, valid_graph.chain)
    assert first == second
    assert first is not second
    object.__setattr__(first, "result_id", "caller-pollution")
    third = _verify(valid_graph.anchor, valid_graph.chain)
    assert third == second
    assert third.result_id != "caller-pollution"
    rejection_one = _verify(None, valid_graph.chain)
    rejection_two = _verify(None, valid_graph.chain)
    assert rejection_one == rejection_two
    assert rejection_one is not rejection_two


def test_backend_unavailable_fails_closed(
    valid_graph: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(signed_v2, "_Ed25519PublicKey", None)
    _assert_atomic_rejection(
        _verify(valid_graph.anchor, valid_graph.chain),
        signed_v2.SignedAttemptChainReasonV2.SIGNATURE_BACKEND_UNAVAILABLE,
    )


def test_internal_exception_is_atomic_all_false(
    valid_graph: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*args: object, **kwargs: object) -> object:
        raise RuntimeError("TEST_ONLY injected internal failure")

    monkeypatch.setattr(signed_v2, "_verify_closed", explode)
    _assert_atomic_rejection(
        _verify(valid_graph.anchor, valid_graph.chain),
        signed_v2.SignedAttemptChainReasonV2.INTERNAL_ERROR,
    )


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
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for name in (qualified(node.func),)
        if name is not None
    }
    return frozenset(imported), frozenset(calls)


def test_source_ast_forbids_operational_boundaries_and_verifier_injection() -> None:
    source = Path(signed_v2.__file__).read_text(encoding="utf-8")
    imported, calls = _ast_imports_and_calls(source)
    forbidden_import_roots = {
        "asyncio",
        "datetime",
        "docker",
        "http",
        "multiprocessing",
        "networkx",
        "numpy",
        "os",
        "pathlib",
        "queue",
        "random",
        "requests",
        "shutil",
        "socket",
        "sqlite3",
        "subprocess",
        "tempfile",
        "time",
        "urllib",
    }
    assert not {name.split(".", 1)[0] for name in imported}.intersection(forbidden_import_roots)
    forbidden_project_fragments = (
        "runner",
        "scorer",
        "scoring",
        "evaluator",
        "decoder",
        "ledger",
        "runtime",
    )
    project_imports = {name for name in imported if name.startswith("hegel_machine") or name.startswith("phase2b")}
    assert not {
        name
        for name in project_imports
        if any(fragment in name.lower() for fragment in forbidden_project_fragments)
    }
    forbidden_call_suffixes = (
        "open",
        "read_text",
        "read_bytes",
        "write_text",
        "write_bytes",
        "system",
        "popen",
        "run",
        "call",
        "check_call",
        "check_output",
        "sleep",
        "time",
        "time_ns",
        "urandom",
        "token_bytes",
        "generate",
        "sign",
    )
    assert not {
        call
        for call in calls
        if call.lower().endswith(tuple(f".{suffix}" for suffix in forbidden_call_suffixes))
        or call.lower() in forbidden_call_suffixes
    }
    verify_signature = inspect.signature(
        signed_v2.verify_phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2
    )
    assert tuple(verify_signature.parameters) == ("anchor", "chain")
    assert "Callable" not in source
