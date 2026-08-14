"""Independent tests for the Phase-2B local attempt ledger mechanics V2.

The fixture uses a deterministic, explicitly TEST_ONLY Ed25519 key.  These
tests exercise only caller-owned local-filesystem OPEN-to-TERMINAL mechanics.
They do not supply an external trust anchor, an authoritative one-shot
registry or timeline, an actual 960-case execution, scoring, gates, effect,
or C1 evidence.
"""

from __future__ import annotations

import ast
from dataclasses import fields
from enum import Enum
import fcntl
import hashlib
import inspect
import json
import multiprocessing
import os
from pathlib import Path, PosixPath
import shutil
import stat
import tempfile
from types import SimpleNamespace
from typing import Callable

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

import hegel_machine.phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2 as ledger_v2
import hegel_machine.phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2 as signed_v2


VERSION = "hegel-machine-phase2b-actual-unsealed-960-local-attempt-ledger-mechanics/2"
CLAIM_LEVEL = "NON_AUTHORITATIVE_LOCAL_FILESYSTEM_ATTEMPT_LEDGER_MECHANICS_ONLY"
SCOPE = "CALLER_OWNED_SINGLE_DIRECTORY_ATTEMPT_ZERO_ONLY"

SIGNED_VERSION = "hegel-machine-phase2b-actual-unsealed-960-signed-attempt-chain-mechanics/2"
SIGNED_CLAIM_LEVEL = "NON_AUTHORITATIVE_TEST_ONLY_SIGNED_ATTEMPT_CHAIN_MECHANICS"
SIGNED_ANCHOR_SCOPE = "TEST_ONLY_SUPPLIED_ANCHOR"
PROTOCOL_ID = "phase2b_protocol_62ad411b5b5a0f912626c54e4bd822a8c585a8f612f5fce6040cce500a11756a"
EXACT_FREEZE_ID = "phase2b_exact_freeze_ffa1fd4fed0b5c2c018803aa9f730b8c85c144efe7e4aa324256681d1c742cbe"

PUBLIC_SURFACE = (
    "PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_VERSION",
    "PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_CLAIM_LEVEL",
    "PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_SCOPE_V2",
    "PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_SCHEMA_ID",
    "PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_POLICY_ID",
    "LocalAttemptLedgerStateV2",
    "LocalAttemptLedgerDispositionV2",
    "LocalAttemptLedgerReasonV2",
    "ActualUnsealed960LocalAttemptLedgerMechanicsV2",
    "ActualUnsealed960LocalAttemptLedgerRejectionV2",
    "open_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2",
    "terminalize_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2",
    "recover_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2",
    "verify_phase2b_actual_unsealed_960_local_attempt_ledger_snapshot_v2",
)

TRUE_CLAIMS = (
    "test_only_anchor_content_reverified",
    "test_only_signed_chain_reverified_for_terminal",
    "owned_nofollow_directory_descriptor_pinned",
    "cooperating_process_directory_lock_held",
    "fixed_namespace_allowlist_verified",
    "attempt_zero_open_record_content_address_verified",
    "attempt_zero_terminal_record_content_address_verified",
    "open_record_file_fsync_completed",
    "terminal_record_file_fsync_completed",
    "directory_fsync_completed",
    "hardlink_no_replace_publication_verified",
    "open_before_terminal_predecessor_verified",
    "single_directory_terminal_exclusion_mechanics_verified",
    "strict_canonical_record_replay_verified",
    "atomic_result_issuance_verified",
)

FALSE_CLAIMS = (
    "external_trust_anchor_verified",
    "external_signature_authority_verified",
    "qualified_cryptographic_backend_verified",
    "custodian_identity_verified",
    "signer_independence_verified",
    "externally_durable_attempt_ledger_verified",
    "hardware_power_loss_durability_qualified",
    "remote_filesystem_durability_qualified",
    "hostile_same_uid_writer_tamper_resistance_verified",
    "owner_deletion_resistance_verified",
    "owner_rollback_resistance_verified",
    "cross_host_global_one_shot_verified",
    "authoritative_attempt_denominator_verified",
    "attempt_registry_complete",
    "retry_authorization_verified",
    "rerun_policy_enforced",
    "one_shot_policy_enforced",
    "external_timestamp_authority_verified",
    "timeline_observed",
    "authoritative_timeline_order_verified",
    "authoritative_attempt_terminal_state_verified",
    "terminal_append_receipt_verified",
    "campaign_finalization_verified",
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

SUCCESS_PREFIX_FIELDS = (
    "disposition",
    "reason",
    "version",
    "schema_id",
    "policy_id",
    "claim_level",
    "scope",
    "result_id",
    "ledger_id",
    "attempt_index",
    "state",
    "anchor_id",
    "attempt_id",
    "open_record_sha256",
    "open_record_id",
    "terminal_record_sha256",
    "terminal_record_id",
    "signed_chain_result_id",
    "chain_root_sha256",
    "chain_id",
    "raw_input_committed_payload_root_sha256",
    "package_commitments_frozen_payload_root_sha256",
    "newly_published",
    "recovered_commit_window",
    "identical_existing_state_observed",
)
SUCCESS_FIELDS = (*SUCCESS_PREFIX_FIELDS, *TRUE_CLAIMS, *FALSE_CLAIMS)
REJECTION_FIELDS = (
    "disposition",
    "reason",
    "version",
    "schema_id",
    "policy_id",
    "claim_level",
    "scope",
    "validation",
    "result_id",
    "ledger_id",
    "attempt_index",
    "state",
    "anchor_id",
    "attempt_id",
    "open_record_sha256",
    "open_record_id",
    "terminal_record_sha256",
    "terminal_record_id",
    "signed_chain_result_id",
    "chain_root_sha256",
    "chain_id",
    "raw_input_committed_payload_root_sha256",
    "package_commitments_frozen_payload_root_sha256",
    "newly_published",
    "recovered_commit_window",
    "identical_existing_state_observed",
    "partial_result_graph_published",
    "filesystem_recovery_required",
    *TRUE_CLAIMS,
    *FALSE_CLAIMS,
)

OPEN_PAIR_FIELDS = (
    "record_type",
    "version",
    "schema_id",
    "policy_id",
    "claim_level",
    "scope",
    "ledger_id",
    "record_sequence",
    "predecessor_record_id",
    "attempt_index",
    "state",
    "signed_chain_schema_id",
    "signed_chain_policy_id",
    "protocol_id",
    "exact_freeze_id",
    "execution_freeze_manifest_id",
    "attempt_nonce_sha256",
    "anchor_id",
    "attempt_id",
)
TERMINAL_PAIR_FIELDS = (
    "record_type",
    "version",
    "schema_id",
    "policy_id",
    "claim_level",
    "scope",
    "ledger_id",
    "record_sequence",
    "predecessor_record_id",
    "attempt_index",
    "state",
    "signed_chain_schema_id",
    "signed_chain_policy_id",
    "protocol_id",
    "exact_freeze_id",
    "execution_freeze_manifest_id",
    "anchor_id",
    "attempt_id",
    "signed_chain_result_id",
    "chain_root_sha256",
    "chain_id",
    "terminal_event_id",
    "raw_input_committed_payload_root_sha256",
    "package_commitments_frozen_payload_root_sha256",
)

OPEN_TARGET = "attempt-000.open.json"
OPEN_PENDING = ".attempt-000.open.json.next"
TERMINAL_TARGET = "attempt-000.terminal.json"
TERMINAL_PENDING = ".attempt-000.terminal.json.next"
ALLOWED_NAMES = frozenset({OPEN_TARGET, OPEN_PENDING, TERMINAL_TARGET, TERMINAL_PENDING})

LEDGER_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/LOCAL_LEDGER/LEDGER/V2\x00"
OPEN_RECORD_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/LOCAL_LEDGER/OPEN/V2\x00"
TERMINAL_RECORD_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/LOCAL_LEDGER/TERMINAL/V2\x00"
RESULT_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/LOCAL_LEDGER/RESULT/V2\x00"
SCHEMA_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/LOCAL_LEDGER/SCHEMA/V2\x00"
POLICY_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/LOCAL_LEDGER/POLICY/V2\x00"

LEDGER_PREFIX = "phase2b_actual_unsealed_960_local_attempt_ledger_v2_"
OPEN_RECORD_PREFIX = "phase2b_actual_unsealed_960_local_open_record_v2_"
TERMINAL_RECORD_PREFIX = "phase2b_actual_unsealed_960_local_terminal_record_v2_"
RESULT_PREFIX = "phase2b_actual_unsealed_960_local_attempt_ledger_result_v2_"
SCHEMA_PREFIX = "phase2b_actual_unsealed_960_local_attempt_ledger_schema_v2_"
POLICY_PREFIX = "phase2b_actual_unsealed_960_local_attempt_ledger_policy_v2_"
EXPECTED_SCHEMA_ID = SCHEMA_PREFIX + "11bb25c528d6c0227c51e29165909e0b9cb39bffc1434239d9872fbe56f30238"
EXPECTED_POLICY_ID = POLICY_PREFIX + "2a2e2c0f79d96f09236ad303dc7f9504b7cdb690b6a156c73d7a623dc3e2eae2"

KEY_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/KEY/V2\x00"
KEY_ROOT_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/KEY_ROOT/V2\x00"
ATTEMPT_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/ATTEMPT/V2\x00"
ANCHOR_ROOT_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/ANCHOR_ROOT/V2\x00"
ANCHOR_ID_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/ANCHOR_ID/V2\x00"
EVENT_SIGNATURE_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/EVENT_SIGNATURE/V2\x00"
EVENT_ID_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/EVENT_ID/V2\x00"
CHAIN_ROOT_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/CHAIN_ROOT/V2\x00"
CHAIN_ID_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/CHAIN_ID/V2\x00"
SIGNED_RESULT_DOMAIN = b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/RESULT/V2\x00"

KEY_PREFIX = "phase2b_actual_unsealed_960_signer_key_v2_"
ATTEMPT_PREFIX = "phase2b_actual_unsealed_960_attempt_v2_"
ANCHOR_PREFIX = "phase2b_actual_unsealed_960_test_anchor_v2_"
EVENT_PREFIX = "phase2b_actual_unsealed_960_attempt_event_v2_"
CHAIN_PREFIX = "phase2b_actual_unsealed_960_signed_attempt_chain_v2_"
SIGNED_RESULT_PREFIX = "phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2_"
SIGNED_SCHEMA_ID = "phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_schema_v2_8cef82cf64f90995092f28647decf7852d695f3d912cb6a807945016828d27c6"
SIGNED_POLICY_ID = "phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_policy_v2_bc9aba86d575035608278044c22731eb29195ef1c32e6e5d585c005882d46b54"

SIGNED_TRUE_CLAIMS = (
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
SIGNED_FALSE_CLAIMS = (
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
SIGNED_RESULT_FIELDS = (
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
    *SIGNED_TRUE_CLAIMS,
    *SIGNED_FALSE_CLAIMS,
)
SIGNED_RESULT_NONBOOL_FIELDS = (
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
)

STAGES = tuple(signed_v2.SignedAttemptStageV2)


def _canonical_json(value: object) -> str:
    """Independent canonical encoder for primitive test preimages."""

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
    version: str = SIGNED_VERSION,
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
        object.__setattr__(
            copied,
            item.name,
            changes.get(item.name, object.__getattribute__(value, item.name)),
        )
    return copied


def _test_private(
    label: bytes = b"HEGEL-PHASE2B-LEDGER-TEST-ONLY-CUSTODIAN-V2",
) -> Ed25519PrivateKey:
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
    execution_freeze = "phase2b_execution_freeze_" + hashlib.sha256(
        b"TEST_ONLY ledger execution freeze"
    ).hexdigest()
    attempt_nonce = hashlib.sha256(
        b"TEST_ONLY ledger attempt nonce before future commitments"
    ).hexdigest()
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
        ("version", SIGNED_VERSION),
        ("claim_level", SIGNED_CLAIM_LEVEL),
        ("anchor_scope", SIGNED_ANCHOR_SCOPE),
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
        version=SIGNED_VERSION,
        claim_level=SIGNED_CLAIM_LEVEL,
        anchor_scope=SIGNED_ANCHOR_SCOPE,
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
        hashlib.sha256(b"TEST_ONLY ledger raw input commitment").hexdigest(),
        hashlib.sha256(b"TEST_ONLY ledger package commitments").hexdigest(),
        *(
            hashlib.sha256(f"TEST_ONLY ledger stage payload {index}".encode()).hexdigest()
            for index in range(3, 10)
        ),
    )


def _build_chain(
    anchor: signed_v2.ActualUnsealed960TestOnlyAnchorV2,
    private: Ed25519PrivateKey,
    *,
    payloads: tuple[str, ...] | None = None,
    version: str = SIGNED_VERSION,
) -> signed_v2.ActualUnsealed960SignedAttemptChainV2:
    payload_values = _default_payloads(anchor) if payloads is None else payloads
    key_id = anchor.signer_keys[0].key_id
    events: list[signed_v2.ActualUnsealed960AttemptEventV2] = []
    predecessor: str | None = None
    for index, stage in enumerate(STAGES):
        pairs = _event_pairs(
            anchor_id=anchor.anchor_id,
            attempt_id=anchor.attempt_id,
            index=index,
            stage=stage,
            predecessor=predecessor,
            payload=payload_values[index],
            key_id=key_id,
            version=version,
        )
        signature = private.sign(
            EVENT_SIGNATURE_DOMAIN + _canonical_json(pairs).encode("utf-8")
        )
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
            signer_key_id=key_id,
            signature=signature,
            event_id=event_id,
        )
        events.append(event)
        predecessor = event_id
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


@pytest.fixture
def ledger_directory(tmp_path: Path) -> PosixPath:
    path = tmp_path / "ledger"
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    assert type(path) is PosixPath
    return path


def _open(directory: PosixPath, anchor: object) -> object:
    return ledger_v2.open_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2(
        ledger_directory=directory,
        anchor=anchor,  # type: ignore[arg-type]
    )


def _terminalize(directory: PosixPath, anchor: object, chain: object) -> object:
    return ledger_v2.terminalize_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2(
        ledger_directory=directory,
        anchor=anchor,  # type: ignore[arg-type]
        chain=chain,  # type: ignore[arg-type]
    )


def _recover(directory: PosixPath, anchor: object, chain: object | None) -> object:
    return ledger_v2.recover_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2(
        ledger_directory=directory,
        anchor=anchor,  # type: ignore[arg-type]
        chain=chain,  # type: ignore[arg-type]
    )


def _snapshot(directory: PosixPath, anchor: object, chain: object | None) -> object:
    return ledger_v2.verify_phase2b_actual_unsealed_960_local_attempt_ledger_snapshot_v2(
        ledger_directory=directory,
        anchor=anchor,  # type: ignore[arg-type]
        chain=chain,  # type: ignore[arg-type]
    )


def _assert_atomic_rejection(
    result: object,
    reason: ledger_v2.LocalAttemptLedgerReasonV2 | None = None,
) -> None:
    assert type(result) is ledger_v2.ActualUnsealed960LocalAttemptLedgerRejectionV2
    assert result.disposition is ledger_v2.LocalAttemptLedgerDispositionV2.REJECTED
    if reason is not None:
        assert result.reason is reason
    assert result.validation is None
    assert result.attempt_index == 0
    for name in (
        "result_id",
        "ledger_id",
        "state",
        "anchor_id",
        "attempt_id",
        "open_record_sha256",
        "open_record_id",
        "terminal_record_sha256",
        "terminal_record_id",
        "signed_chain_result_id",
        "chain_root_sha256",
        "chain_id",
        "raw_input_committed_payload_root_sha256",
        "package_commitments_frozen_payload_root_sha256",
    ):
        assert getattr(result, name) is None
    assert result.newly_published is False
    assert result.recovered_commit_window is False
    assert result.identical_existing_state_observed is False
    assert result.partial_result_graph_published is False
    assert all(getattr(result, name) is False for name in (*TRUE_CLAIMS, *FALSE_CLAIMS))


def _ledger_pairs(
    anchor: signed_v2.ActualUnsealed960TestOnlyAnchorV2,
) -> tuple[tuple[str, object], ...]:
    return (
        ("version", VERSION),
        ("schema_id", ledger_v2.PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_SCHEMA_ID),
        ("policy_id", ledger_v2.PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_POLICY_ID),
        ("scope", SCOPE),
        ("signed_chain_schema_id", anchor.schema_id),
        ("signed_chain_policy_id", anchor.policy_id),
        ("protocol_id", anchor.protocol_id),
        ("exact_freeze_id", anchor.exact_freeze_id),
        ("execution_freeze_manifest_id", anchor.execution_freeze_manifest_id),
        ("anchor_id", anchor.anchor_id),
        ("attempt_id", anchor.attempt_id),
        ("attempt_index", 0),
    )


def _expected_ledger_id(anchor: signed_v2.ActualUnsealed960TestOnlyAnchorV2) -> str:
    return LEDGER_PREFIX + _digest(LEDGER_DOMAIN, _ledger_pairs(anchor))


def _document(
    pairs: tuple[tuple[str, object], ...], domain: bytes, prefix: str
) -> tuple[dict[str, object], bytes, str, str]:
    record_sha = _digest(domain, pairs)
    record_id = prefix + record_sha
    document = dict(pairs)
    document["record_sha256"] = record_sha
    document["record_id"] = record_id
    return document, _canonical_json(document).encode("utf-8"), record_sha, record_id


def _expected_open_record(
    anchor: signed_v2.ActualUnsealed960TestOnlyAnchorV2,
) -> tuple[dict[str, object], bytes, str, str]:
    pairs = (
        ("record_type", "OPEN"),
        ("version", VERSION),
        ("schema_id", ledger_v2.PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_SCHEMA_ID),
        ("policy_id", ledger_v2.PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_POLICY_ID),
        ("claim_level", CLAIM_LEVEL),
        ("scope", SCOPE),
        ("ledger_id", _expected_ledger_id(anchor)),
        ("record_sequence", 0),
        ("predecessor_record_id", None),
        ("attempt_index", 0),
        ("state", "OPEN"),
        ("signed_chain_schema_id", anchor.schema_id),
        ("signed_chain_policy_id", anchor.policy_id),
        ("protocol_id", anchor.protocol_id),
        ("exact_freeze_id", anchor.exact_freeze_id),
        ("execution_freeze_manifest_id", anchor.execution_freeze_manifest_id),
        ("attempt_nonce_sha256", anchor.attempt_nonce_sha256),
        ("anchor_id", anchor.anchor_id),
        ("attempt_id", anchor.attempt_id),
    )
    assert tuple(name for name, _ in pairs) == OPEN_PAIR_FIELDS
    return _document(pairs, OPEN_RECORD_DOMAIN, OPEN_RECORD_PREFIX)


def _expected_signed_result_id(
    anchor: signed_v2.ActualUnsealed960TestOnlyAnchorV2,
    chain: signed_v2.ActualUnsealed960SignedAttemptChainV2,
) -> str:
    pairs = (
        ("version", SIGNED_VERSION),
        ("schema_id", anchor.schema_id),
        ("policy_id", anchor.policy_id),
        ("anchor_id", anchor.anchor_id),
        ("attempt_id", anchor.attempt_id),
        ("signer_key_id", anchor.signer_keys[0].key_id),
        ("raw_input_committed_payload_root_sha256", chain.events[1].payload_root_sha256),
        ("package_commitments_frozen_payload_root_sha256", chain.events[2].payload_root_sha256),
        ("chain_root_sha256", chain.chain_root_sha256),
        ("chain_id", chain.chain_id),
        ("event_count", 10),
        ("terminal_stage", "TERMINAL_CONSUMED"),
    )
    return SIGNED_RESULT_PREFIX + _digest(SIGNED_RESULT_DOMAIN, pairs)


def _expected_terminal_record(
    anchor: signed_v2.ActualUnsealed960TestOnlyAnchorV2,
    chain: signed_v2.ActualUnsealed960SignedAttemptChainV2,
) -> tuple[dict[str, object], bytes, str, str]:
    open_record = _expected_open_record(anchor)
    pairs = (
        ("record_type", "TERMINAL"),
        ("version", VERSION),
        ("schema_id", ledger_v2.PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_SCHEMA_ID),
        ("policy_id", ledger_v2.PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_POLICY_ID),
        ("claim_level", CLAIM_LEVEL),
        ("scope", SCOPE),
        ("ledger_id", _expected_ledger_id(anchor)),
        ("record_sequence", 1),
        ("predecessor_record_id", open_record[3]),
        ("attempt_index", 0),
        ("state", "TERMINAL_CONSUMED"),
        ("signed_chain_schema_id", anchor.schema_id),
        ("signed_chain_policy_id", anchor.policy_id),
        ("protocol_id", anchor.protocol_id),
        ("exact_freeze_id", anchor.exact_freeze_id),
        ("execution_freeze_manifest_id", anchor.execution_freeze_manifest_id),
        ("anchor_id", anchor.anchor_id),
        ("attempt_id", anchor.attempt_id),
        ("signed_chain_result_id", _expected_signed_result_id(anchor, chain)),
        ("chain_root_sha256", chain.chain_root_sha256),
        ("chain_id", chain.chain_id),
        ("terminal_event_id", chain.events[-1].event_id),
        ("raw_input_committed_payload_root_sha256", chain.events[1].payload_root_sha256),
        ("package_commitments_frozen_payload_root_sha256", chain.events[2].payload_root_sha256),
    )
    assert tuple(name for name, _ in pairs) == TERMINAL_PAIR_FIELDS
    return _document(pairs, TERMINAL_RECORD_DOMAIN, TERMINAL_RECORD_PREFIX)


def _assert_result_id(result: object) -> None:
    pairs = tuple(
        (name, getattr(result, name))
        for name in SUCCESS_FIELDS
        if name != "result_id"
    )
    assert result.result_id == RESULT_PREFIX + _digest(RESULT_DOMAIN, pairs)


def _assert_no_authority_claims(result: object) -> None:
    assert all(getattr(result, name) is False for name in FALSE_CLAIMS)
    assert result.external_trust_anchor_verified is False
    assert result.execution_authorized is False
    assert result.actual_960_case_run_verified is False
    assert result.scoring_performed is False
    assert result.formal_c1_report_verified is False
    assert result.c1_exit_evidence is False


def test_exact_public_surface_fields_enums_and_keyword_only_signatures() -> None:
    assert tuple(ledger_v2.__all__) == PUBLIC_SURFACE
    assert tuple(
        item.name
        for item in fields(ledger_v2.ActualUnsealed960LocalAttemptLedgerMechanicsV2)
    ) == SUCCESS_FIELDS
    assert tuple(
        item.name
        for item in fields(ledger_v2.ActualUnsealed960LocalAttemptLedgerRejectionV2)
    ) == REJECTION_FIELDS
    assert tuple(item.value for item in ledger_v2.LocalAttemptLedgerStateV2) == (
        "EMPTY",
        "OPEN",
        "TERMINAL_CONSUMED",
    )
    assert tuple(item.value for item in ledger_v2.LocalAttemptLedgerDispositionV2) == (
        "LOCAL_ATTEMPT_ZERO_OPEN_PUBLISHED",
        "LOCAL_ATTEMPT_ZERO_OPEN_RECOVERED",
        "LOCAL_ATTEMPT_ZERO_OPEN_ALREADY_PRESENT_IDENTICAL",
        "LOCAL_ATTEMPT_ZERO_TERMINAL_PUBLISHED",
        "LOCAL_ATTEMPT_ZERO_TERMINAL_RECOVERED",
        "LOCAL_ATTEMPT_ZERO_TERMINAL_ALREADY_PRESENT_IDENTICAL",
        "LOCAL_ATTEMPT_LEDGER_EMPTY_VERIFIED",
        "LOCAL_ATTEMPT_LEDGER_OPEN_VERIFIED",
        "LOCAL_ATTEMPT_LEDGER_TERMINAL_VERIFIED",
        "REJECTED",
    )
    assert tuple(item.value for item in ledger_v2.LocalAttemptLedgerReasonV2) == (
        "OPEN_RECORD_FILE_AND_DIRECTORY_FSYNC_PUBLICATION_COMPLETED_NOT_EXECUTION_AUTHORIZATION",
        "OPEN_RECORD_RECOVERED_NOT_EXECUTION_AUTHORIZATION",
        "IDENTICAL_OPEN_ALREADY_PRESENT_EXECUTION_STATE_UNKNOWN",
        "TERMINAL_RECORD_FILE_AND_DIRECTORY_FSYNC_PUBLICATION_COMPLETED_LOCAL_MECHANICS_ONLY",
        "TERMINAL_RECORD_RECOVERED_LOCAL_MECHANICS_ONLY",
        "IDENTICAL_TERMINAL_ALREADY_PRESENT_LOCAL_MECHANICS_ONLY",
        "EMPTY_DIRECTORY_VERIFIED",
        "OPEN_RECORD_VERIFIED_EXECUTION_STATE_UNKNOWN",
        "TERMINAL_RECORD_VERIFIED_LOCAL_MECHANICS_ONLY",
        "WRONG_INPUT_TYPE",
        "CROSS_VERSION_INPUT",
        "TEST_ANCHOR_INVALID",
        "SIGNED_CHAIN_REJECTED",
        "STATE_DIRECTORY_INVALID",
        "UNSUPPORTED_PLATFORM",
        "DIRECTORY_LOCK_BUSY",
        "NAMESPACE_TAMPERED",
        "OPEN_RECORD_REQUIRED",
        "TERMINAL_STATE_ALREADY_CLOSED",
        "DIFFERENT_ATTEMPT_OR_CHAIN_CONFLICT",
        "RECOVERY_INPUT_REQUIRED",
        "CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED",
        "FILESYSTEM_OPERATION_FAILED_RECOVERY_REQUIRED",
        "INTERNAL_ERROR",
    )
    signatures = (
        (
            ledger_v2.open_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2,
            ("ledger_directory", "anchor"),
        ),
        (
            ledger_v2.terminalize_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2,
            ("ledger_directory", "anchor", "chain"),
        ),
        (
            ledger_v2.recover_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2,
            ("ledger_directory", "anchor", "chain"),
        ),
        (
            ledger_v2.verify_phase2b_actual_unsealed_960_local_attempt_ledger_snapshot_v2,
            ("ledger_directory", "anchor", "chain"),
        ),
    )
    for operation, names in signatures:
        signature = inspect.signature(operation)
        assert tuple(signature.parameters) == names
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in signature.parameters.values()
        )
        assert all(
            parameter.default is inspect.Parameter.empty
            for parameter in signature.parameters.values()
        )
    with pytest.raises(TypeError, match="privately issued"):
        ledger_v2.ActualUnsealed960LocalAttemptLedgerMechanicsV2()
    with pytest.raises(TypeError, match="privately issued"):
        ledger_v2.ActualUnsealed960LocalAttemptLedgerRejectionV2()


def test_schema_policy_and_content_address_contract_are_independently_recomputed() -> None:
    assert ledger_v2.PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_VERSION == VERSION
    assert ledger_v2.PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_CLAIM_LEVEL == CLAIM_LEVEL
    assert ledger_v2.PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_SCOPE_V2 == SCOPE
    success_types = tuple(
        (item.name, str(item.type))
        for item in fields(ledger_v2.ActualUnsealed960LocalAttemptLedgerMechanicsV2)
    )
    rejection_types = tuple(
        (item.name, str(item.type))
        for item in fields(ledger_v2.ActualUnsealed960LocalAttemptLedgerRejectionV2)
    )
    input_types = (
        ("ledger_directory", "exact_pathlib.PosixPath_absolute_existing_directory"),
        ("anchor", "exact_ActualUnsealed960TestOnlyAnchorV2"),
        ("chain", "exact_ActualUnsealed960SignedAttemptChainV2_or_exact_None"),
    )
    schema_preimage = (
        ("version", VERSION),
        ("claim_level", CLAIM_LEVEL),
        ("scope", SCOPE),
        ("success_fields", SUCCESS_FIELDS),
        ("rejection_fields", REJECTION_FIELDS),
        ("success_type_manifest", success_types),
        ("rejection_type_manifest", rejection_types),
        ("input_type_manifest", input_types),
        ("signed_chain_result_fields", SIGNED_RESULT_FIELDS),
        ("signed_chain_result_nonbool_fields", SIGNED_RESULT_NONBOOL_FIELDS),
        ("signed_chain_result_true_claims", SIGNED_TRUE_CLAIMS),
        ("state_values", tuple(item.value for item in ledger_v2.LocalAttemptLedgerStateV2)),
        ("disposition_values", tuple(item.value for item in ledger_v2.LocalAttemptLedgerDispositionV2)),
        ("reason_values", tuple(item.value for item in ledger_v2.LocalAttemptLedgerReasonV2)),
        ("open_record_pair_fields", OPEN_PAIR_FIELDS),
        ("terminal_record_pair_fields", TERMINAL_PAIR_FIELDS),
        ("fixed_names", tuple(sorted(ALLOWED_NAMES))),
        ("max_text_bytes", 4_096),
        ("max_path_component_bytes", 255),
        ("max_path_components", 64),
        ("max_path_bytes", 4_096),
        ("max_namespace_entries", 4),
        ("max_record_bytes", 16_384),
    )
    expected_schema = SCHEMA_PREFIX + _digest(SCHEMA_DOMAIN, schema_preimage)
    assert expected_schema == EXPECTED_SCHEMA_ID
    assert ledger_v2.PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_SCHEMA_ID == expected_schema
    content_parameters = (
        ("ledger_domain_hex", LEDGER_DOMAIN.hex()),
        ("open_record_domain_hex", OPEN_RECORD_DOMAIN.hex()),
        ("terminal_record_domain_hex", TERMINAL_RECORD_DOMAIN.hex()),
        ("result_domain_hex", RESULT_DOMAIN.hex()),
        ("signed_chain_result_domain_hex", SIGNED_RESULT_DOMAIN.hex()),
        ("schema_domain_hex", SCHEMA_DOMAIN.hex()),
        ("policy_domain_hex", POLICY_DOMAIN.hex()),
        ("ledger_prefix", LEDGER_PREFIX),
        ("open_record_prefix", OPEN_RECORD_PREFIX),
        ("terminal_record_prefix", TERMINAL_RECORD_PREFIX),
        ("result_prefix", RESULT_PREFIX),
        ("signed_chain_result_prefix", SIGNED_RESULT_PREFIX),
        ("schema_prefix", SCHEMA_PREFIX),
        ("policy_prefix", POLICY_PREFIX),
    )
    policy_preimage = (
        ("version", VERSION),
        ("claim_level", CLAIM_LEVEL),
        ("scope", SCOPE),
        ("schema_id", expected_schema),
        ("signed_chain_version", SIGNED_VERSION),
        ("signed_chain_schema_id", SIGNED_SCHEMA_ID),
        ("signed_chain_policy_id", SIGNED_POLICY_ID),
        ("signed_chain_claim_level", SIGNED_CLAIM_LEVEL),
        ("signed_chain_anchor_scope", SIGNED_ANCHOR_SCOPE),
        ("content_address_parameters", content_parameters),
        (
            "public_API",
            (
                "open_keyword_only_ledger_directory_anchor",
                "terminalize_keyword_only_ledger_directory_anchor_chain",
                "recover_keyword_only_ledger_directory_anchor_optional_chain",
                "verify_snapshot_keyword_only_ledger_directory_anchor_optional_chain",
            ),
        ),
        ("global_preflight_rule", "all_caller_path_anchor_and_optional_chain_primitives_close_before_any_content_hash_signature_or_filesystem_operation"),
        ("state_machine", (("EMPTY", "OPEN"), ("OPEN", "TERMINAL_CONSUMED"))),
        ("attempt_index_rule", "exact_zero_only_no_retry_or_reopen"),
        ("open_rule", "durable_open_must_complete_before_runtime_may_be_considered_by_external_caller"),
        ("open_receipt_rule", "never_execution_authorization_and_identical_open_has_unknown_execution_state"),
        ("terminal_rule", "internal_signed_chain_reverification_and_exact_open_predecessor_required"),
        ("signed_chain_result_rule", "exact_success_disposition_reason_identity_result_id_terminal_and_complete_claim_polarity_projection"),
        ("path_rule", "absolute_exact_PosixPath_componentwise_openat_nofollow_owned_mode0700"),
        ("path_rebind_rule", "caller_visible_path_dev_inode_rechecked_against_held_dirfd_before_success"),
        ("lock_rule", "directory_inode_flock_LOCK_EX_LOCK_NB_held_for_full_operation"),
        ("namespace_rule", tuple(sorted(ALLOWED_NAMES))),
        (
            "namespace_state_profiles",
            (
                ("EMPTY", ()),
                ("OPEN_PENDING", (OPEN_PENDING,)),
                ("OPEN_CLEAN", (OPEN_TARGET,)),
                ("OPEN_LINK_WINDOW", (OPEN_TARGET, OPEN_PENDING)),
                ("TERMINAL_PENDING", (OPEN_TARGET, TERMINAL_PENDING)),
                ("TERMINAL_CLEAN", (OPEN_TARGET, TERMINAL_TARGET)),
                ("TERMINAL_LINK_WINDOW", (OPEN_TARGET, TERMINAL_TARGET, TERMINAL_PENDING)),
            ),
        ),
        ("namespace_classification_rule", "entire_initial_name_and_inode_graph_validated_before_any_recovery_mutation"),
        (
            "publication_order",
            (
                "pending_O_CREAT_O_EXCL_O_NOFOLLOW",
                "full_write",
                "file_fsync",
                "pending_directory_fsync",
                "dirfd_hardlink_no_replace",
                "published_directory_fsync",
                "verified_pending_unlink",
                "final_directory_fsync",
            ),
        ),
        ("recovery_rule", "only_exact_pending_or_exact_target_pending_same_inode_is_recoverable"),
        ("corruption_rule", "unknown_partial_noncanonical_or_ambiguous_state_preserved_and_rejected"),
        ("record_formula", "prefix_plus_sha256(domain_plus_canonical_json_declared_order_named_pairs)"),
        ("strict_record_rule", "bounded_UTF8_duplicate_free_integer_only_exact_canonical_JSON"),
        ("ordinary_failure_rule", "no_partial_result_ids_or_true_claims_and_filesystem_evidence_preserved"),
        ("BaseException_and_process_termination_rule", "outside_ordinary_atomic_result_claim"),
        ("local_claims", TRUE_CLAIMS),
        ("nonclaims", FALSE_CLAIMS),
        (
            "forbidden_operations",
            (
                "os.replace",
                "os.rename",
                "filesystem_cleanup_api",
                "clock",
                "datetime",
                "random",
                "secrets",
                "network",
                "subprocess",
                "docker",
                "runner",
                "scorer",
                "decoder",
                "key_generation",
                "signature_generation",
                "caller_supplied_verifier",
            ),
        ),
    )
    expected_policy = POLICY_PREFIX + _digest(POLICY_DOMAIN, policy_preimage)
    assert expected_policy == EXPECTED_POLICY_ID
    assert ledger_v2.PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_POLICY_ID == expected_policy


def test_empty_snapshot_is_not_identical_existing_state_or_authority(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    result = _snapshot(ledger_directory, valid_graph.anchor, None)
    assert type(result) is ledger_v2.ActualUnsealed960LocalAttemptLedgerMechanicsV2
    assert result.disposition is ledger_v2.LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_LEDGER_EMPTY_VERIFIED
    assert result.reason is ledger_v2.LocalAttemptLedgerReasonV2.EMPTY_DIRECTORY_VERIFIED
    assert result.state is ledger_v2.LocalAttemptLedgerStateV2.EMPTY
    assert result.attempt_index == 0
    assert result.newly_published is False
    assert result.recovered_commit_window is False
    assert result.identical_existing_state_observed is False
    assert result.open_record_id is None
    assert result.terminal_record_id is None
    assert result.open_record_file_fsync_completed is False
    assert result.directory_fsync_completed is False
    assert result.hardlink_no_replace_publication_verified is False
    assert result.strict_canonical_record_replay_verified is False
    assert set(ledger_directory.iterdir()) == set()
    _assert_result_id(result)
    _assert_no_authority_claims(result)


def test_positive_open_record_is_exact_canonical_attempt_zero_and_not_authorization(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    expected = _expected_open_record(valid_graph.anchor)
    result = _open(ledger_directory, valid_graph.anchor)
    assert type(result) is ledger_v2.ActualUnsealed960LocalAttemptLedgerMechanicsV2
    assert result.disposition is ledger_v2.LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_ZERO_OPEN_PUBLISHED
    assert result.reason is ledger_v2.LocalAttemptLedgerReasonV2.OPEN_RECORD_FILE_AND_DIRECTORY_FSYNC_PUBLICATION_COMPLETED_NOT_EXECUTION_AUTHORIZATION
    assert result.state is ledger_v2.LocalAttemptLedgerStateV2.OPEN
    assert result.attempt_index == 0
    assert result.ledger_id == _expected_ledger_id(valid_graph.anchor)
    assert result.open_record_sha256 == expected[2]
    assert result.open_record_id == expected[3]
    assert result.terminal_record_id is None
    assert result.signed_chain_result_id is None
    assert result.newly_published is True
    assert result.recovered_commit_window is False
    assert result.identical_existing_state_observed is False
    assert result.open_record_file_fsync_completed is True
    assert result.terminal_record_file_fsync_completed is False
    assert result.directory_fsync_completed is True
    assert result.hardlink_no_replace_publication_verified is True
    assert result.test_only_signed_chain_reverified_for_terminal is False
    assert result.open_before_terminal_predecessor_verified is False
    assert result.single_directory_terminal_exclusion_mechanics_verified is False
    record_path = ledger_directory / OPEN_TARGET
    assert record_path.read_bytes() == expected[1]
    assert json.loads(record_path.read_text(encoding="utf-8")) == expected[0]
    metadata = record_path.stat()
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_nlink == 1
    assert {item.name for item in ledger_directory.iterdir()} == {OPEN_TARGET}
    _assert_result_id(result)
    _assert_no_authority_claims(result)


def test_open_snapshot_and_duplicate_are_deterministic_but_fresh_and_invocation_scoped(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    published = _open(ledger_directory, valid_graph.anchor)
    snapshot = _snapshot(ledger_directory, valid_graph.anchor, None)
    duplicate = _open(ledger_directory, valid_graph.anchor)
    assert type(snapshot) is ledger_v2.ActualUnsealed960LocalAttemptLedgerMechanicsV2
    assert snapshot.disposition is ledger_v2.LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_LEDGER_OPEN_VERIFIED
    assert snapshot.reason is ledger_v2.LocalAttemptLedgerReasonV2.OPEN_RECORD_VERIFIED_EXECUTION_STATE_UNKNOWN
    assert duplicate.disposition is ledger_v2.LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_ZERO_OPEN_ALREADY_PRESENT_IDENTICAL
    assert duplicate.reason is ledger_v2.LocalAttemptLedgerReasonV2.IDENTICAL_OPEN_ALREADY_PRESENT_EXECUTION_STATE_UNKNOWN
    assert snapshot.state is duplicate.state is ledger_v2.LocalAttemptLedgerStateV2.OPEN
    duplicate_again = _open(ledger_directory, valid_graph.anchor)
    assert snapshot is not duplicate and duplicate is not published
    assert duplicate_again is not duplicate
    assert snapshot.result_id != duplicate.result_id
    assert duplicate_again.result_id == duplicate.result_id
    assert snapshot.newly_published is duplicate.newly_published is False
    assert snapshot.recovered_commit_window is duplicate.recovered_commit_window is False
    assert snapshot.identical_existing_state_observed is True
    assert duplicate.identical_existing_state_observed is True
    assert snapshot.open_record_file_fsync_completed is False
    assert duplicate.open_record_file_fsync_completed is False
    assert snapshot.directory_fsync_completed is False
    assert duplicate.directory_fsync_completed is False
    _assert_result_id(snapshot)
    _assert_result_id(duplicate)
    _assert_no_authority_claims(snapshot)
    _assert_no_authority_claims(duplicate)


def test_positive_terminal_record_independently_binds_open_and_real_signed_chain(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    open_result = _open(ledger_directory, valid_graph.anchor)
    expected = _expected_terminal_record(valid_graph.anchor, valid_graph.chain)
    result = _terminalize(ledger_directory, valid_graph.anchor, valid_graph.chain)
    assert type(result) is ledger_v2.ActualUnsealed960LocalAttemptLedgerMechanicsV2
    assert result.disposition is ledger_v2.LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_ZERO_TERMINAL_PUBLISHED
    assert result.reason is ledger_v2.LocalAttemptLedgerReasonV2.TERMINAL_RECORD_FILE_AND_DIRECTORY_FSYNC_PUBLICATION_COMPLETED_LOCAL_MECHANICS_ONLY
    assert result.state is ledger_v2.LocalAttemptLedgerStateV2.TERMINAL_CONSUMED
    assert result.open_record_id == open_result.open_record_id
    assert result.terminal_record_sha256 == expected[2]
    assert result.terminal_record_id == expected[3]
    assert result.signed_chain_result_id == _expected_signed_result_id(
        valid_graph.anchor, valid_graph.chain
    )
    assert result.chain_root_sha256 == valid_graph.chain.chain_root_sha256
    assert result.chain_id == valid_graph.chain.chain_id
    assert result.raw_input_committed_payload_root_sha256 == valid_graph.chain.events[1].payload_root_sha256
    assert result.package_commitments_frozen_payload_root_sha256 == valid_graph.chain.events[2].payload_root_sha256
    assert result.newly_published is True
    assert result.test_only_signed_chain_reverified_for_terminal is True
    assert result.attempt_zero_open_record_content_address_verified is True
    assert result.attempt_zero_terminal_record_content_address_verified is True
    assert result.open_record_file_fsync_completed is False
    assert result.terminal_record_file_fsync_completed is True
    assert result.directory_fsync_completed is True
    assert result.open_before_terminal_predecessor_verified is True
    assert result.single_directory_terminal_exclusion_mechanics_verified is True
    terminal_path = ledger_directory / TERMINAL_TARGET
    assert terminal_path.read_bytes() == expected[1]
    assert json.loads(terminal_path.read_text(encoding="utf-8")) == expected[0]
    metadata = terminal_path.stat()
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_nlink == 1
    assert {item.name for item in ledger_directory.iterdir()} == {
        OPEN_TARGET,
        TERMINAL_TARGET,
    }
    _assert_result_id(result)
    _assert_no_authority_claims(result)


def test_terminal_snapshot_duplicate_and_reopen_have_closed_state_semantics(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    _open(ledger_directory, valid_graph.anchor)
    published = _terminalize(ledger_directory, valid_graph.anchor, valid_graph.chain)
    snapshot = _snapshot(ledger_directory, valid_graph.anchor, valid_graph.chain)
    duplicate = _terminalize(ledger_directory, valid_graph.anchor, valid_graph.chain)
    assert snapshot.disposition is ledger_v2.LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_LEDGER_TERMINAL_VERIFIED
    assert snapshot.reason is ledger_v2.LocalAttemptLedgerReasonV2.TERMINAL_RECORD_VERIFIED_LOCAL_MECHANICS_ONLY
    assert duplicate.disposition is ledger_v2.LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_ZERO_TERMINAL_ALREADY_PRESENT_IDENTICAL
    assert duplicate.reason is ledger_v2.LocalAttemptLedgerReasonV2.IDENTICAL_TERMINAL_ALREADY_PRESENT_LOCAL_MECHANICS_ONLY
    assert snapshot.state is duplicate.state is ledger_v2.LocalAttemptLedgerStateV2.TERMINAL_CONSUMED
    duplicate_again = _terminalize(ledger_directory, valid_graph.anchor, valid_graph.chain)
    assert duplicate_again is not duplicate
    assert snapshot.result_id != duplicate.result_id
    assert duplicate_again.result_id == duplicate.result_id
    assert snapshot.result_id != published.result_id
    assert snapshot.identical_existing_state_observed is True
    assert duplicate.identical_existing_state_observed is True
    assert snapshot.terminal_record_file_fsync_completed is False
    assert duplicate.terminal_record_file_fsync_completed is False
    assert snapshot.directory_fsync_completed is False
    assert duplicate.directory_fsync_completed is False
    reopen = _open(ledger_directory, valid_graph.anchor)
    _assert_atomic_rejection(
        reopen,
        ledger_v2.LocalAttemptLedgerReasonV2.TERMINAL_STATE_ALREADY_CLOSED,
    )
    assert reopen.filesystem_recovery_required is True
    assert {item.name for item in ledger_directory.iterdir()} == {
        OPEN_TARGET,
        TERMINAL_TARGET,
    }


def test_terminal_requires_prior_open_and_does_not_create_terminal_only_marker(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    result = _terminalize(ledger_directory, valid_graph.anchor, valid_graph.chain)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.OPEN_RECORD_REQUIRED,
    )
    assert result.filesystem_recovery_required is False
    assert set(ledger_directory.iterdir()) == set()


def test_terminal_snapshot_requires_chain_and_does_not_infer_terminal_state(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    _open(ledger_directory, valid_graph.anchor)
    _terminalize(ledger_directory, valid_graph.anchor, valid_graph.chain)
    before = {
        item.name: (item.read_bytes(), item.stat().st_ino)
        for item in ledger_directory.iterdir()
    }
    result = _snapshot(ledger_directory, valid_graph.anchor, None)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.RECOVERY_INPUT_REQUIRED,
    )
    assert result.filesystem_recovery_required is True
    after = {
        item.name: (item.read_bytes(), item.stat().st_ino)
        for item in ledger_directory.iterdir()
    }
    assert after == before


@pytest.mark.parametrize("window", ["pending_only", "target_and_pending"])
def test_verify_is_read_only_and_recover_finishes_only_exact_open_commit_windows(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
    window: str,
) -> None:
    expected = _expected_open_record(valid_graph.anchor)
    _open(ledger_directory, valid_graph.anchor)
    target = ledger_directory / OPEN_TARGET
    pending = ledger_directory / OPEN_PENDING
    if window == "pending_only":
        target.rename(pending)
        expected_inode = pending.stat().st_ino
    else:
        os.link(target, pending, follow_symlinks=False)
        expected_inode = target.stat().st_ino
        assert target.stat().st_nlink == pending.stat().st_nlink == 2
    names_before = {item.name for item in ledger_directory.iterdir()}
    snapshot = _snapshot(ledger_directory, valid_graph.anchor, None)
    _assert_atomic_rejection(
        snapshot,
        ledger_v2.LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
    )
    assert {item.name for item in ledger_directory.iterdir()} == names_before
    recovered = _recover(ledger_directory, valid_graph.anchor, None)
    assert type(recovered) is ledger_v2.ActualUnsealed960LocalAttemptLedgerMechanicsV2
    assert recovered.disposition is ledger_v2.LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_ZERO_OPEN_RECOVERED
    assert recovered.reason is ledger_v2.LocalAttemptLedgerReasonV2.OPEN_RECORD_RECOVERED_NOT_EXECUTION_AUTHORIZATION
    assert recovered.state is ledger_v2.LocalAttemptLedgerStateV2.OPEN
    assert recovered.recovered_commit_window is True
    assert recovered.newly_published is False
    assert recovered.identical_existing_state_observed is False
    assert recovered.open_record_file_fsync_completed is True
    assert recovered.directory_fsync_completed is True
    assert recovered.hardlink_no_replace_publication_verified is True
    assert target.read_bytes() == expected[1]
    assert target.stat().st_ino == expected_inode
    assert target.stat().st_nlink == 1
    assert not pending.exists()
    _assert_result_id(recovered)
    _assert_no_authority_claims(recovered)


@pytest.mark.parametrize("window", ["pending_only", "target_and_pending"])
def test_recover_finishes_only_exact_terminal_commit_windows_with_chain_reverification(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
    window: str,
) -> None:
    expected = _expected_terminal_record(valid_graph.anchor, valid_graph.chain)
    _open(ledger_directory, valid_graph.anchor)
    _terminalize(ledger_directory, valid_graph.anchor, valid_graph.chain)
    target = ledger_directory / TERMINAL_TARGET
    pending = ledger_directory / TERMINAL_PENDING
    if window == "pending_only":
        target.rename(pending)
        expected_inode = pending.stat().st_ino
    else:
        os.link(target, pending, follow_symlinks=False)
        expected_inode = target.stat().st_ino
    without_chain = _recover(ledger_directory, valid_graph.anchor, None)
    _assert_atomic_rejection(
        without_chain,
        ledger_v2.LocalAttemptLedgerReasonV2.RECOVERY_INPUT_REQUIRED,
    )
    recovered = _recover(ledger_directory, valid_graph.anchor, valid_graph.chain)
    assert type(recovered) is ledger_v2.ActualUnsealed960LocalAttemptLedgerMechanicsV2
    assert recovered.disposition is ledger_v2.LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_ZERO_TERMINAL_RECOVERED
    assert recovered.reason is ledger_v2.LocalAttemptLedgerReasonV2.TERMINAL_RECORD_RECOVERED_LOCAL_MECHANICS_ONLY
    assert recovered.state is ledger_v2.LocalAttemptLedgerStateV2.TERMINAL_CONSUMED
    assert recovered.recovered_commit_window is True
    assert recovered.test_only_signed_chain_reverified_for_terminal is True
    assert recovered.terminal_record_file_fsync_completed is True
    assert recovered.directory_fsync_completed is True
    assert target.read_bytes() == expected[1]
    assert target.stat().st_ino == expected_inode
    assert target.stat().st_nlink == 1
    assert not pending.exists()
    _assert_result_id(recovered)
    _assert_no_authority_claims(recovered)


@pytest.mark.parametrize(
    ("payload", "reason"),
    (
        (b"", ledger_v2.LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED),
        (b"{", ledger_v2.LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED),
        (b"\xff", ledger_v2.LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED),
        (b'{"record_type":"OPEN", "x":1}', ledger_v2.LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED),
        (b'{"x":1.0}', ledger_v2.LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED),
        (b'{"x":NaN}', ledger_v2.LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED),
        (b'{"x":1,"x":1}', ledger_v2.LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED),
        (b"x" * 16_385, ledger_v2.LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED),
    ),
)
def test_noncanonical_partial_duplicate_float_and_oversize_records_are_preserved_and_rejected(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
    payload: bytes,
    reason: ledger_v2.LocalAttemptLedgerReasonV2,
) -> None:
    _open(ledger_directory, valid_graph.anchor)
    target = ledger_directory / OPEN_TARGET
    target.write_bytes(payload)
    target.chmod(0o600)
    result = _recover(ledger_directory, valid_graph.anchor, None)
    _assert_atomic_rejection(result, reason)
    assert result.filesystem_recovery_required is True
    assert target.read_bytes() == payload
    assert {item.name for item in ledger_directory.iterdir()} == {OPEN_TARGET}


def test_fully_rehashed_canonical_open_tamper_is_a_different_attempt_conflict(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    _open(ledger_directory, valid_graph.anchor)
    target = ledger_directory / OPEN_TARGET
    document = dict(_expected_open_record(valid_graph.anchor)[0])
    document["attempt_index"] = 1
    pairs = tuple((name, document[name]) for name in OPEN_PAIR_FIELDS)
    record_sha = _digest(OPEN_RECORD_DOMAIN, pairs)
    document["record_sha256"] = record_sha
    document["record_id"] = OPEN_RECORD_PREFIX + record_sha
    tampered = _canonical_json(document).encode("utf-8")
    target.write_bytes(tampered)
    target.chmod(0o600)
    result = _snapshot(ledger_directory, valid_graph.anchor, None)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.DIFFERENT_ATTEMPT_OR_CHAIN_CONFLICT,
    )
    assert result.filesystem_recovery_required is True
    assert target.read_bytes() == tampered


def test_different_test_anchor_cannot_fork_an_existing_open_attempt(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    first = _open(ledger_directory, valid_graph.anchor)
    other_private = _test_private(b"HEGEL-PHASE2B-LEDGER-OTHER-TEST-ONLY-KEY")
    other_anchor = _build_anchor(other_private)
    result = _open(ledger_directory, other_anchor)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.DIFFERENT_ATTEMPT_OR_CHAIN_CONFLICT,
    )
    assert result.filesystem_recovery_required is True
    assert (ledger_directory / OPEN_TARGET).read_bytes() == _expected_open_record(
        valid_graph.anchor
    )[1]
    assert first.anchor_id == valid_graph.anchor.anchor_id


def test_different_valid_chain_cannot_fork_an_existing_terminal_attempt(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    _open(ledger_directory, valid_graph.anchor)
    first = _terminalize(ledger_directory, valid_graph.anchor, valid_graph.chain)
    payloads = list(_default_payloads(valid_graph.anchor))
    payloads[8] = hashlib.sha256(b"TEST_ONLY alternate reports commitment").hexdigest()
    other_chain = _build_chain(
        valid_graph.anchor,
        valid_graph.private,
        payloads=tuple(payloads),
    )
    result = _terminalize(ledger_directory, valid_graph.anchor, other_chain)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.DIFFERENT_ATTEMPT_OR_CHAIN_CONFLICT,
    )
    assert result.filesystem_recovery_required is True
    assert (ledger_directory / TERMINAL_TARGET).read_bytes() == _expected_terminal_record(
        valid_graph.anchor, valid_graph.chain
    )[1]
    assert first.chain_id == valid_graph.chain.chain_id


def test_invalid_signed_chain_is_internally_reverified_before_filesystem_mutation(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    _open(ledger_directory, valid_graph.anchor)
    event = valid_graph.chain.events[7]
    bad_signature = bytes([event.signature[0] ^ 1]) + event.signature[1:]
    bad_event = _unchecked_copy(event, signature=bad_signature)
    events = (*valid_graph.chain.events[:7], bad_event, *valid_graph.chain.events[8:])
    bad_chain = _unchecked_copy(valid_graph.chain, events=events)
    result = _terminalize(ledger_directory, valid_graph.anchor, bad_chain)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.SIGNED_CHAIN_REJECTED,
    )
    assert result.filesystem_recovery_required is False
    assert {item.name for item in ledger_directory.iterdir()} == {OPEN_TARGET}


def test_internal_chain_verifier_cannot_be_bypassed_by_a_success_boolean(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _open(ledger_directory, valid_graph.anchor)
    monkeypatch.setattr(
        ledger_v2,
        "verify_phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2",
        lambda **_: True,
    )
    result = _terminalize(ledger_directory, valid_graph.anchor, valid_graph.chain)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.SIGNED_CHAIN_REJECTED,
    )
    assert {item.name for item in ledger_directory.iterdir()} == {OPEN_TARGET}


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("result_id", SIGNED_RESULT_PREFIX + "f" * 64),
        ("test_only_anchor_scope_enforced", False),
        ("execution_authorized", True),
    ),
)
def test_forged_exact_signed_success_result_is_rejected_before_terminal_publication(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    _open(ledger_directory, valid_graph.anchor)
    verified = signed_v2.verify_phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2(
        anchor=valid_graph.anchor,
        chain=valid_graph.chain,
    )
    assert type(verified) is signed_v2.ActualUnsealed960SignedAttemptChainMechanicsV2
    forged = _unchecked_copy(verified, **{field: value})
    monkeypatch.setattr(
        ledger_v2,
        "verify_phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2",
        lambda **_: forged,
    )
    result = _terminalize(ledger_directory, valid_graph.anchor, valid_graph.chain)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.SIGNED_CHAIN_REJECTED,
    )
    assert {item.name for item in ledger_directory.iterdir()} == {OPEN_TARGET}


@pytest.mark.parametrize("where", ["anchor", "chain", "event"])
def test_top_and_nested_cross_version_inputs_route_before_operations(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
    where: str,
) -> None:
    anchor = valid_graph.anchor
    chain = valid_graph.chain
    drift = SIGNED_VERSION + "/future"
    if where == "anchor":
        anchor = _unchecked_copy(anchor, version=drift)
    elif where == "chain":
        chain = _unchecked_copy(chain, version=drift)
    else:
        late = _unchecked_copy(chain.events[-1], version=drift)
        chain = _unchecked_copy(chain, events=(*chain.events[:-1], late))
    result = _terminalize(ledger_directory, anchor, chain)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.CROSS_VERSION_INPUT,
    )
    assert set(ledger_directory.iterdir()) == set()


@pytest.mark.parametrize(
    ("operation", "directory", "anchor", "chain", "reason"),
    (
        ("open", "not-a-path", "valid", None, "WRONG_INPUT_TYPE"),
        ("open", "relative", "valid", None, "STATE_DIRECTORY_INVALID"),
        ("open", "root", "valid", None, "STATE_DIRECTORY_INVALID"),
        ("open", "valid", None, None, "WRONG_INPUT_TYPE"),
        ("terminal", "valid", "valid", None, "WRONG_INPUT_TYPE"),
        ("terminal", "valid", "valid", (), "WRONG_INPUT_TYPE"),
        ("recover", "valid", "valid", (), "WRONG_INPUT_TYPE"),
        ("snapshot", "valid", "valid", [], "WRONG_INPUT_TYPE"),
    ),
)
def test_wrong_input_types_reject_fresh_atomically_before_filesystem_use(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
    operation: str,
    directory: object,
    anchor: object,
    chain: object,
    reason: str,
) -> None:
    directory_value: object = {
        "valid": ledger_directory,
        "relative": PosixPath("relative-ledger"),
        "root": PosixPath("/"),
    }.get(directory, directory)
    anchor_value = valid_graph.anchor if anchor == "valid" else anchor
    if operation == "open":
        result = _open(directory_value, anchor_value)  # type: ignore[arg-type]
    elif operation == "terminal":
        result = _terminalize(
            directory_value, anchor_value, valid_graph.chain if chain == "valid" else chain
        )  # type: ignore[arg-type]
    elif operation == "recover":
        result = _recover(directory_value, anchor_value, chain)  # type: ignore[arg-type]
    else:
        result = _snapshot(directory_value, anchor_value, chain)  # type: ignore[arg-type]
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2(reason),
    )
    assert set(ledger_directory.iterdir()) == set()


def _subclass_copy(value: object, name: str) -> object:
    subtype = type(name, (type(value),), {})
    copied = object.__new__(subtype)
    for item in fields(value):
        object.__setattr__(copied, item.name, object.__getattribute__(value, item.name))
    return copied


@pytest.mark.parametrize("where", ["path", "anchor", "chain", "event"])
def test_path_and_graph_subclasses_are_not_accepted(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
    where: str,
) -> None:
    directory: object = ledger_directory
    anchor: object = valid_graph.anchor
    chain: object = valid_graph.chain
    if where == "path":
        subtype = type("LedgerPosixPathSubclass", (PosixPath,), {})
        directory = subtype(str(ledger_directory))
    elif where == "anchor":
        anchor = _subclass_copy(anchor, "AnchorSubclass")
    elif where == "chain":
        chain = _subclass_copy(chain, "ChainSubclass")
    else:
        event = _subclass_copy(valid_graph.chain.events[-1], "EventSubclass")
        chain = _unchecked_copy(valid_graph.chain, events=(*valid_graph.chain.events[:-1], event))
    result = _terminalize(directory, anchor, chain)  # type: ignore[arg-type]
    reason = (
        ledger_v2.LocalAttemptLedgerReasonV2.SIGNED_CHAIN_REJECTED
        if where == "event"
        else ledger_v2.LocalAttemptLedgerReasonV2.WRONG_INPUT_TYPE
    )
    _assert_atomic_rejection(result, reason)
    assert set(ledger_directory.iterdir()) == set()


class _PrehashBoundaryReached(BaseException):
    """Sentinel that ordinary fail-closed normalization must not catch."""


def test_global_shape_preflight_closes_late_chain_before_hash_signature_or_filesystem(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    late = _unchecked_copy(valid_graph.chain.events[-1], signature=b"short")
    malformed = _unchecked_copy(
        valid_graph.chain,
        events=(*valid_graph.chain.events[:-1], late),
    )

    def forbidden(*_: object, **__: object) -> object:
        raise _PrehashBoundaryReached("operation occurred before global close")

    monkeypatch.setattr(ledger_v2.hashlib, "sha256", forbidden)
    monkeypatch.setattr(
        ledger_v2,
        "verify_phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2",
        forbidden,
    )
    monkeypatch.setattr(ledger_v2, "_open_pinned_directory", forbidden)
    result = _terminalize(ledger_directory, valid_graph.anchor, malformed)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.SIGNED_CHAIN_REJECTED,
    )
    assert set(ledger_directory.iterdir()) == set()


def test_nested_cross_version_is_detected_before_any_content_hash(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    late = _unchecked_copy(
        valid_graph.chain.events[-1],
        version=SIGNED_VERSION + "/future",
    )
    malformed = _unchecked_copy(
        valid_graph.chain,
        events=(*valid_graph.chain.events[:-1], late),
    )

    def forbidden(*_: object, **__: object) -> object:
        raise _PrehashBoundaryReached("hash reached")

    monkeypatch.setattr(ledger_v2.hashlib, "sha256", forbidden)
    result = _terminalize(ledger_directory, valid_graph.anchor, malformed)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.CROSS_VERSION_INPUT,
    )
    assert set(ledger_directory.iterdir()) == set()


def test_injected_prehash_baseexception_escapes_and_is_not_misreported_as_atomic(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stop(*_: object, **__: object) -> object:
        raise _PrehashBoundaryReached("test-only prehash sentinel")

    monkeypatch.setattr(ledger_v2.hashlib, "sha256", stop)
    with pytest.raises(_PrehashBoundaryReached, match="prehash sentinel"):
        _open(ledger_directory, valid_graph.anchor)
    assert set(ledger_directory.iterdir()) == set()


def test_anchor_snapshot_is_closed_before_hash_time_caller_mutation(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _expected_open_record(valid_graph.anchor)
    real_digest = ledger_v2._digest
    mutated = False

    def mutate_then_digest(
        domain: bytes, pairs: tuple[tuple[str, object], ...]
    ) -> str:
        nonlocal mutated
        if not mutated:
            mutated = True
            object.__setattr__(valid_graph.anchor, "attempt_nonce_sha256", "f" * 64)
        return real_digest(domain, pairs)

    monkeypatch.setattr(ledger_v2, "_digest", mutate_then_digest)
    result = _open(ledger_directory, valid_graph.anchor)
    assert type(result) is ledger_v2.ActualUnsealed960LocalAttemptLedgerMechanicsV2
    assert mutated is True
    assert (ledger_directory / OPEN_TARGET).read_bytes() == expected[1]
    assert result.open_record_id == expected[3]


def test_chain_snapshot_is_closed_before_verification_time_caller_mutation(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _open(ledger_directory, valid_graph.anchor)
    expected = _expected_terminal_record(valid_graph.anchor, valid_graph.chain)
    real_verify = ledger_v2._verify_signed_chain

    def mutate_then_verify(anchor: object, chain: object) -> object:
        object.__setattr__(
            valid_graph.chain,
            "chain_id",
            CHAIN_PREFIX + "f" * 64,
        )
        return real_verify(anchor, chain)  # type: ignore[arg-type]

    monkeypatch.setattr(ledger_v2, "_verify_signed_chain", mutate_then_verify)
    result = _terminalize(ledger_directory, valid_graph.anchor, valid_graph.chain)
    assert type(result) is ledger_v2.ActualUnsealed960LocalAttemptLedgerMechanicsV2
    assert result.terminal_record_id == expected[3]
    assert (ledger_directory / TERMINAL_TARGET).read_bytes() == expected[1]


def test_ordinary_exception_after_hardlink_is_atomic_and_recoverable_without_cleanup(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_barrier = ledger_v2._barrier_v2

    def fail_after_link(label: str) -> None:
        if label == "after_link_before_publish_dir_fsync":
            raise RuntimeError("TEST_ONLY ordinary injected failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(ledger_v2, "_barrier_v2", fail_after_link)
        result = _open(ledger_directory, valid_graph.anchor)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.INTERNAL_ERROR,
    )
    assert result.filesystem_recovery_required is True
    target = ledger_directory / OPEN_TARGET
    pending = ledger_directory / OPEN_PENDING
    assert target.exists() and pending.exists()
    assert target.stat().st_ino == pending.stat().st_ino
    assert target.stat().st_nlink == pending.stat().st_nlink == 2
    assert ledger_v2._barrier_v2 is real_barrier
    recovered = _recover(ledger_directory, valid_graph.anchor, None)
    assert recovered.disposition is ledger_v2.LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_ZERO_OPEN_RECOVERED
    assert {item.name for item in ledger_directory.iterdir()} == {OPEN_TARGET}


def test_partial_write_oserror_preserves_pending_evidence_and_issues_no_claims(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_write = ledger_v2.os.write
    calls = 0

    def partial_then_fail(descriptor: int, payload: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, payload[:13])
        raise OSError("TEST_ONLY partial write failure")

    monkeypatch.setattr(ledger_v2.os, "write", partial_then_fail)
    result = _open(ledger_directory, valid_graph.anchor)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.FILESYSTEM_OPERATION_FAILED_RECOVERY_REQUIRED,
    )
    assert result.filesystem_recovery_required is True
    pending = ledger_directory / OPEN_PENDING
    assert pending.read_bytes() == _expected_open_record(valid_graph.anchor)[1][:13]
    assert not (ledger_directory / OPEN_TARGET).exists()


def test_successes_and_rejections_are_fresh_deterministic_private_results(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    first_success = _snapshot(ledger_directory, valid_graph.anchor, None)
    second_success = _snapshot(ledger_directory, valid_graph.anchor, None)
    first_rejection = _open(ledger_directory, None)
    second_rejection = _open(ledger_directory, None)
    assert first_success is not second_success
    assert first_success.result_id == second_success.result_id
    assert first_rejection is not second_rejection
    assert first_rejection == second_rejection
    _assert_atomic_rejection(
        first_rejection,
        ledger_v2.LocalAttemptLedgerReasonV2.WRONG_INPUT_TYPE,
    )
    _assert_atomic_rejection(
        second_rejection,
        ledger_v2.LocalAttemptLedgerReasonV2.WRONG_INPUT_TYPE,
    )


@pytest.mark.parametrize(
    ("name", "kind"),
    (
        (OPEN_TARGET, "symlink"),
        (OPEN_TARGET, "directory"),
        (OPEN_TARGET, "fifo"),
        (OPEN_TARGET, "socket"),
        (OPEN_PENDING, "symlink"),
    ),
)
def test_allowed_name_nonregular_nodes_reject_promptly_and_preserve_outside_target(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
    name: str,
    kind: str,
) -> None:
    short_directory: PosixPath | None = None
    active_directory = ledger_directory
    if kind == "socket":
        short_directory = PosixPath(tempfile.mkdtemp(prefix="hgl-", dir="/tmp"))
        short_directory.chmod(0o700)
        active_directory = short_directory
    outside = active_directory.parent / f".{active_directory.name}.outside-record"
    attack = active_directory / name
    socket_object = None
    try:
        outside.write_bytes(b"OUTSIDE MUST NOT CHANGE")
        outside.chmod(0o600)
        if kind == "symlink":
            attack.symlink_to(outside)
        elif kind == "directory":
            attack.mkdir(mode=0o700)
        elif kind == "fifo":
            os.mkfifo(attack, 0o600)
        else:
            import socket

            socket_object = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            socket_object.bind(str(attack))
        result = _recover(active_directory, valid_graph.anchor, None)
        _assert_atomic_rejection(
            result,
            ledger_v2.LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
        )
        assert result.filesystem_recovery_required is True
        assert outside.read_bytes() == b"OUTSIDE MUST NOT CHANGE"
        assert attack.exists() or attack.is_symlink()
    finally:
        if socket_object is not None:
            socket_object.close()
        if short_directory is not None:
            shutil.rmtree(short_directory)
            outside.unlink(missing_ok=True)


def test_external_hardlink_and_independent_same_bytes_link_window_are_rejected_preserved(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    expected = _expected_open_record(valid_graph.anchor)[1]
    outside = ledger_directory.parent / "outside-hardlink"
    outside.write_bytes(expected)
    outside.chmod(0o600)
    target = ledger_directory / OPEN_TARGET
    os.link(outside, target, follow_symlinks=False)
    result = _recover(ledger_directory, valid_graph.anchor, None)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
    )
    assert result.filesystem_recovery_required is True
    assert outside.stat().st_nlink == target.stat().st_nlink == 2
    target.unlink()
    outside.unlink()
    target.write_bytes(expected)
    pending = ledger_directory / OPEN_PENDING
    pending.write_bytes(expected)
    target.chmod(0o600)
    pending.chmod(0o600)
    assert target.stat().st_ino != pending.stat().st_ino
    result = _recover(ledger_directory, valid_graph.anchor, None)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
    )
    assert target.read_bytes() == pending.read_bytes() == expected
    assert target.stat().st_ino != pending.stat().st_ino


def test_unknown_namespace_and_impossible_cross_stage_graphs_are_preserved(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    unknown = ledger_directory / "unknown-entry"
    unknown.write_bytes(b"do not delete")
    result = _recover(ledger_directory, valid_graph.anchor, None)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.NAMESPACE_TAMPERED,
    )
    assert unknown.read_bytes() == b"do not delete"
    unknown.unlink()

    expected_open = _expected_open_record(valid_graph.anchor)[1]
    expected_terminal = _expected_terminal_record(valid_graph.anchor, valid_graph.chain)[1]
    (ledger_directory / OPEN_PENDING).write_bytes(expected_open)
    (ledger_directory / OPEN_PENDING).chmod(0o600)
    (ledger_directory / TERMINAL_TARGET).write_bytes(expected_terminal)
    (ledger_directory / TERMINAL_TARGET).chmod(0o600)
    before = {
        item.name: (item.read_bytes(), item.stat().st_ino)
        for item in ledger_directory.iterdir()
    }
    result = _recover(ledger_directory, valid_graph.anchor, valid_graph.chain)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
    )
    after = {
        item.name: (item.read_bytes(), item.stat().st_ino)
        for item in ledger_directory.iterdir()
    }
    assert after == before


@pytest.mark.parametrize(
    "names",
    (
        frozenset({TERMINAL_PENDING}),
        frozenset({TERMINAL_TARGET}),
        frozenset({TERMINAL_TARGET, TERMINAL_PENDING}),
        frozenset({OPEN_PENDING, TERMINAL_PENDING}),
        frozenset({OPEN_PENDING, TERMINAL_TARGET}),
        frozenset({OPEN_PENDING, TERMINAL_TARGET, TERMINAL_PENDING}),
        frozenset({OPEN_TARGET, OPEN_PENDING, TERMINAL_PENDING}),
        frozenset({OPEN_TARGET, OPEN_PENDING, TERMINAL_TARGET}),
        frozenset({OPEN_TARGET, OPEN_PENDING, TERMINAL_TARGET, TERMINAL_PENDING}),
    ),
    ids=(
        "terminal-pending-without-open",
        "terminal-target-without-open",
        "terminal-link-window-without-open",
        "both-pending-no-open",
        "open-pending-plus-terminal",
        "open-pending-plus-terminal-window",
        "open-window-plus-terminal-pending",
        "open-window-plus-terminal-target",
        "all-four-names",
    ),
)
def test_every_unreachable_four_name_subset_is_preserved_and_rejected(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
    names: frozenset[str],
) -> None:
    for name in names:
        path = ledger_directory / name
        path.write_bytes(f"TEST_ONLY impossible {name}".encode())
        path.chmod(0o600)
    before = {
        item.name: (item.read_bytes(), item.stat().st_ino)
        for item in ledger_directory.iterdir()
    }
    result = _recover(ledger_directory, valid_graph.anchor, valid_graph.chain)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
    )
    assert result.filesystem_recovery_required is True
    after = {
        item.name: (item.read_bytes(), item.stat().st_ino)
        for item in ledger_directory.iterdir()
    }
    assert after == before


def test_nominal_terminal_namespace_with_corrupt_terminal_blocks_reopen_but_requires_recovery(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    _open(ledger_directory, valid_graph.anchor)
    corrupt = ledger_directory / TERMINAL_TARGET
    corrupt.write_bytes(b"corrupt terminal")
    corrupt.chmod(0o600)
    result = _open(ledger_directory, valid_graph.anchor)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.TERMINAL_STATE_ALREADY_CLOSED,
    )
    assert result.filesystem_recovery_required is True
    assert corrupt.read_bytes() == b"corrupt terminal"


def test_directory_requires_exact_owned_mode_0700_and_symlink_ancestors_fail_closed(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    ledger_directory.chmod(0o755)
    mode_result = _open(ledger_directory, valid_graph.anchor)
    _assert_atomic_rejection(
        mode_result,
        ledger_v2.LocalAttemptLedgerReasonV2.STATE_DIRECTORY_INVALID,
    )
    ledger_directory.chmod(0o700)
    real_parent = ledger_directory.parent / "real-parent"
    real_parent.mkdir(mode=0o700)
    real_ledger = real_parent / "ledger"
    real_ledger.mkdir(mode=0o700)
    symlink_parent = ledger_directory.parent / "symlink-parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    result = _open(PosixPath(symlink_parent / "ledger"), valid_graph.anchor)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.STATE_DIRECTORY_INVALID,
    )
    assert set(real_ledger.iterdir()) == set()


def test_directory_lock_busy_for_all_public_operations_has_zero_mutation(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    descriptor = os.open(
        ledger_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        results = (
            _open(ledger_directory, valid_graph.anchor),
            _terminalize(ledger_directory, valid_graph.anchor, valid_graph.chain),
            _recover(ledger_directory, valid_graph.anchor, None),
            _snapshot(ledger_directory, valid_graph.anchor, None),
        )
    finally:
        os.close(descriptor)
    for result in results:
        _assert_atomic_rejection(
            result,
            ledger_v2.LocalAttemptLedgerReasonV2.DIRECTORY_LOCK_BUSY,
        )
        assert result.filesystem_recovery_required is False
    assert set(ledger_directory.iterdir()) == set()


def _result_tuple(result: object) -> tuple[str, str, str | None, bool]:
    if type(result) is ledger_v2.ActualUnsealed960LocalAttemptLedgerMechanicsV2:
        recovery_required = False
    elif type(result) is ledger_v2.ActualUnsealed960LocalAttemptLedgerRejectionV2:
        recovery_required = result.filesystem_recovery_required
    else:
        raise TypeError("unexpected local-ledger result type")
    return (
        result.disposition.value,
        result.reason.value,
        None if result.state is None else result.state.value,
        recovery_required,
    )


def _spawn_open_worker(
    directory: str,
    anchor: signed_v2.ActualUnsealed960TestOnlyAnchorV2,
    start: multiprocessing.synchronize.Event,
    output: multiprocessing.queues.Queue,
) -> None:
    start.wait()
    result = ledger_v2.open_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2(
        ledger_directory=PosixPath(directory),
        anchor=anchor,
    )
    output.put(_result_tuple(result))


def _spawn_terminal_worker(
    directory: str,
    anchor: signed_v2.ActualUnsealed960TestOnlyAnchorV2,
    chain: signed_v2.ActualUnsealed960SignedAttemptChainV2,
    start: multiprocessing.synchronize.Event,
    output: multiprocessing.queues.Queue,
) -> None:
    start.wait()
    result = ledger_v2.terminalize_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2(
        ledger_directory=PosixPath(directory),
        anchor=anchor,
        chain=chain,
    )
    output.put(_result_tuple(result))


def _spawn_crash_worker(
    operation: str,
    directory: str,
    anchor: signed_v2.ActualUnsealed960TestOnlyAnchorV2,
    chain: signed_v2.ActualUnsealed960SignedAttemptChainV2 | None,
    barrier: str,
    partial_write: bool,
) -> None:
    real_write = ledger_v2.os.write
    write_calls = 0

    def write_then_maybe_die(descriptor: int, payload: bytes) -> int:
        nonlocal write_calls
        write_calls += 1
        if partial_write and write_calls == 1:
            return real_write(descriptor, payload[:13])
        return real_write(descriptor, payload)

    def die_at(label: str) -> None:
        if label == barrier:
            os._exit(86)

    ledger_v2._barrier_v2 = die_at
    if partial_write:
        ledger_v2.os.write = write_then_maybe_die
    path = PosixPath(directory)
    if operation == "open":
        ledger_v2.open_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2(
            ledger_directory=path,
            anchor=anchor,
        )
    elif operation == "terminal":
        assert chain is not None
        ledger_v2.terminalize_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2(
            ledger_directory=path,
            anchor=anchor,
            chain=chain,
        )
    else:
        assert chain is not None
        ledger_v2.recover_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2(
            ledger_directory=path,
            anchor=anchor,
            chain=chain,
        )
    os._exit(87)


def _spawn_api_worker(
    operation: str,
    directory: str,
    anchor: signed_v2.ActualUnsealed960TestOnlyAnchorV2,
    chain: signed_v2.ActualUnsealed960SignedAttemptChainV2 | None,
    output: multiprocessing.queues.Queue,
) -> None:
    path = PosixPath(directory)
    if operation == "open":
        result = ledger_v2.open_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2(
            ledger_directory=path,
            anchor=anchor,
        )
    else:
        result = ledger_v2.recover_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2(
            ledger_directory=path,
            anchor=anchor,
            chain=chain,
        )
    output.put(_result_tuple(result))


def _bounded_join(process: multiprocessing.Process, timeout: float = 8.0) -> None:
    process.join(timeout)
    if process.is_alive():
        process.kill()
        process.join(2.0)
        pytest.fail(f"spawned worker hung: pid={process.pid}")


def test_true_spawn_eight_way_open_race_has_one_new_and_one_canonical_target(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    output = context.Queue()
    processes = [
        context.Process(
            target=_spawn_open_worker,
            args=(str(ledger_directory), valid_graph.anchor, start, output),
        )
        for _ in range(8)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        _bounded_join(process)
        assert process.exitcode == 0
    results = [output.get(timeout=2.0) for _ in processes]
    published = "LOCAL_ATTEMPT_ZERO_OPEN_PUBLISHED"
    busy = "REJECTED"
    identical = "LOCAL_ATTEMPT_ZERO_OPEN_ALREADY_PRESENT_IDENTICAL"
    assert sum(item[0] == published for item in results) == 1
    assert all(item[0] in {published, busy, identical} for item in results)
    assert all(
        item[1] == "DIRECTORY_LOCK_BUSY"
        for item in results
        if item[0] == busy
    )
    assert (ledger_directory / OPEN_TARGET).read_bytes() == _expected_open_record(
        valid_graph.anchor
    )[1]
    assert {item.name for item in ledger_directory.iterdir()} == {OPEN_TARGET}


def test_true_spawn_eight_way_terminal_race_has_one_new_and_no_fork(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    _open(ledger_directory, valid_graph.anchor)
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    output = context.Queue()
    processes = [
        context.Process(
            target=_spawn_terminal_worker,
            args=(
                str(ledger_directory),
                valid_graph.anchor,
                valid_graph.chain,
                start,
                output,
            ),
        )
        for _ in range(8)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        _bounded_join(process)
        assert process.exitcode == 0
    results = [output.get(timeout=2.0) for _ in processes]
    published = "LOCAL_ATTEMPT_ZERO_TERMINAL_PUBLISHED"
    busy = "REJECTED"
    identical = "LOCAL_ATTEMPT_ZERO_TERMINAL_ALREADY_PRESENT_IDENTICAL"
    assert sum(item[0] == published for item in results) == 1
    assert all(item[0] in {published, busy, identical} for item in results)
    assert all(
        item[1] == "DIRECTORY_LOCK_BUSY"
        for item in results
        if item[0] == busy
    )
    assert (ledger_directory / TERMINAL_TARGET).read_bytes() == _expected_terminal_record(
        valid_graph.anchor, valid_graph.chain
    )[1]


def test_true_spawn_different_anchor_fork_race_retains_exactly_one_attempt(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    other_private = _test_private(b"TEST_ONLY spawn fork contender")
    other_anchor = _build_anchor(other_private)
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    output = context.Queue()
    processes = [
        context.Process(
            target=_spawn_open_worker,
            args=(str(ledger_directory), anchor, start, output),
        )
        for anchor in (valid_graph.anchor, other_anchor)
        for _ in range(4)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        _bounded_join(process)
        assert process.exitcode == 0
    results = [output.get(timeout=2.0) for _ in processes]
    assert sum(item[0] == "LOCAL_ATTEMPT_ZERO_OPEN_PUBLISHED" for item in results) == 1
    payload = (ledger_directory / OPEN_TARGET).read_bytes()
    candidates = {
        _expected_open_record(valid_graph.anchor)[1],
        _expected_open_record(other_anchor)[1],
    }
    assert payload in candidates
    loser = other_anchor if payload == _expected_open_record(valid_graph.anchor)[1] else valid_graph.anchor
    conflict = _open(ledger_directory, loser)
    _assert_atomic_rejection(
        conflict,
        ledger_v2.LocalAttemptLedgerReasonV2.DIFFERENT_ATTEMPT_OR_CHAIN_CONFLICT,
    )
    assert (ledger_directory / OPEN_TARGET).read_bytes() == payload


@pytest.mark.parametrize(
    "barrier",
    (
        "after_pending_create",
        "after_full_write_before_file_fsync",
        "after_file_fsync_before_pending_dir_fsync",
        "after_pending_dir_fsync_before_link",
        "after_link_before_publish_dir_fsync",
        "after_publish_dir_fsync_before_pending_unlink",
        "after_pending_unlink_before_final_dir_fsync",
        "after_final_dir_fsync",
    ),
)
def test_process_exit_at_each_open_publication_barrier_converges_or_preserves(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
    barrier: str,
) -> None:
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_spawn_crash_worker,
        args=(
            "open",
            str(ledger_directory),
            valid_graph.anchor,
            None,
            barrier,
            False,
        ),
    )
    process.start()
    _bounded_join(process)
    assert process.exitcode == 86
    if barrier == "after_pending_create":
        pending = ledger_directory / OPEN_PENDING
        assert pending.exists() and pending.stat().st_size == 0
        result = _recover(ledger_directory, valid_graph.anchor, None)
        _assert_atomic_rejection(
            result,
            ledger_v2.LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
        )
        assert pending.exists() and pending.stat().st_size == 0
        return
    result = _recover(ledger_directory, valid_graph.anchor, None)
    assert type(result) is ledger_v2.ActualUnsealed960LocalAttemptLedgerMechanicsV2
    assert result.state is ledger_v2.LocalAttemptLedgerStateV2.OPEN
    assert (ledger_directory / OPEN_TARGET).read_bytes() == _expected_open_record(
        valid_graph.anchor
    )[1]
    assert {item.name for item in ledger_directory.iterdir()} == {OPEN_TARGET}


def test_process_exit_after_partial_write_preserves_corrupt_pending_without_inference(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_spawn_crash_worker,
        args=(
            "open",
            str(ledger_directory),
            valid_graph.anchor,
            None,
            "after_write_chunk",
            True,
        ),
    )
    process.start()
    _bounded_join(process)
    assert process.exitcode == 86
    pending = ledger_directory / OPEN_PENDING
    assert pending.read_bytes() == _expected_open_record(valid_graph.anchor)[1][:13]
    result = _recover(ledger_directory, valid_graph.anchor, None)
    _assert_atomic_rejection(
        result,
        ledger_v2.LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
    )
    assert pending.read_bytes() == _expected_open_record(valid_graph.anchor)[1][:13]


@pytest.mark.parametrize(
    "barrier",
    (
        "after_pending_create",
        "after_full_write_before_file_fsync",
        "after_file_fsync_before_pending_dir_fsync",
        "after_pending_dir_fsync_before_link",
        "after_link_before_publish_dir_fsync",
        "after_publish_dir_fsync_before_pending_unlink",
        "after_pending_unlink_before_final_dir_fsync",
        "after_final_dir_fsync",
    ),
)
def test_process_exit_at_each_terminal_publication_barrier_converges_or_preserves(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
    barrier: str,
) -> None:
    _open(ledger_directory, valid_graph.anchor)
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_spawn_crash_worker,
        args=(
            "terminal",
            str(ledger_directory),
            valid_graph.anchor,
            valid_graph.chain,
            barrier,
            False,
        ),
    )
    process.start()
    _bounded_join(process)
    assert process.exitcode == 86
    if barrier == "after_pending_create":
        pending = ledger_directory / TERMINAL_PENDING
        assert pending.exists() and pending.stat().st_size == 0
        result = _recover(ledger_directory, valid_graph.anchor, valid_graph.chain)
        _assert_atomic_rejection(
            result,
            ledger_v2.LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
        )
        assert pending.exists() and pending.stat().st_size == 0
        return
    result = _recover(ledger_directory, valid_graph.anchor, valid_graph.chain)
    assert type(result) is ledger_v2.ActualUnsealed960LocalAttemptLedgerMechanicsV2
    assert result.state is ledger_v2.LocalAttemptLedgerStateV2.TERMINAL_CONSUMED
    assert (ledger_directory / TERMINAL_TARGET).read_bytes() == _expected_terminal_record(
        valid_graph.anchor, valid_graph.chain
    )[1]
    assert {item.name for item in ledger_directory.iterdir()} == {
        OPEN_TARGET,
        TERMINAL_TARGET,
    }


def test_fifo_fixed_name_is_bounded_in_spawn_process_and_never_blocks(
    ledger_directory: PosixPath,
    valid_graph: SimpleNamespace,
) -> None:
    fifo = ledger_directory / OPEN_TARGET
    os.mkfifo(fifo, 0o600)
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(
        target=_spawn_api_worker,
        args=(
            "recover",
            str(ledger_directory),
            valid_graph.anchor,
            None,
            output,
        ),
    )
    process.start()
    _bounded_join(process, timeout=3.0)
    assert process.exitcode == 0
    result = output.get(timeout=2.0)
    assert result[0] == "REJECTED"
    assert result[1] == "CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED"
    assert result[3] is True
    assert stat.S_ISFIFO(fifo.lstat().st_mode)


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


def test_source_ast_allows_only_local_ledger_boundaries_and_no_caller_verifier() -> None:
    source = Path(ledger_v2.__file__).read_text(encoding="utf-8")
    imported, calls = _ast_imports_and_calls(source)
    forbidden_import_roots = {
        "asyncio",
        "datetime",
        "docker",
        "http",
        "multiprocessing",
        "networkx",
        "numpy",
        "queue",
        "random",
        "requests",
        "secrets",
        "shutil",
        "socket",
        "sqlite3",
        "subprocess",
        "tempfile",
        "time",
        "urllib",
    }
    assert not {
        name.split(".", 1)[0] for name in imported
    }.intersection(forbidden_import_roots)
    project_imports = {
        name
        for name in imported
        if name.startswith("hegel_machine") or name.startswith("phase2b")
    }
    forbidden_project_fragments = (
        "runner",
        "scorer",
        "scoring",
        "evaluator",
        "decoder",
        "runtime",
    )
    assert not {
        name
        for name in project_imports
        if any(fragment in name.lower() for fragment in forbidden_project_fragments)
    }
    forbidden_exact_calls = {
        "open",
        "os._exit",
        "os.makedirs",
        "os.mkdir",
        "os.popen",
        "os.remove",
        "os.rename",
        "os.replace",
        "os.rmdir",
        "os.system",
        "pathlib.Path.read_bytes",
        "pathlib.Path.read_text",
        "pathlib.Path.write_bytes",
        "pathlib.Path.write_text",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.Popen",
        "subprocess.run",
    }
    assert not calls.intersection(forbidden_exact_calls)
    forbidden_call_suffixes = (
        ".from_private_bytes",
        ".generate",
        ".generate_key",
        ".monotonic",
        ".perf_counter",
        ".private_key",
        ".sign",
        ".sleep",
        ".time",
        ".time_ns",
        ".token_bytes",
        ".urandom",
    )
    assert not {
        call for call in calls if call.lower().endswith(forbidden_call_suffixes)
    }
    required_local_calls = {
        "fcntl.flock",
        "hashlib.sha256",
        "os.fsync",
        "os.link",
        "os.open",
        "os.unlink",
    }
    assert required_local_calls.issubset(calls)
    assert any(call.endswith(".canonical_json") for call in calls)
    assert any(
        call.endswith(
            ".verify_phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2"
        )
        for call in calls
    )
    assert "os.O_EXCL" in source
    assert "os.O_NOFOLLOW" in source
    assert "os.O_NONBLOCK" in source
    assert "fcntl.LOCK_NB" in source
    assert "Callable" not in source
    assert "_barrier_v2" not in ledger_v2.__all__
    operations = (
        ledger_v2.open_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2,
        ledger_v2.terminalize_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2,
        ledger_v2.recover_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2,
        ledger_v2.verify_phase2b_actual_unsealed_960_local_attempt_ledger_snapshot_v2,
    )
    forbidden_parameter_fragments = (
        "backend",
        "barrier",
        "callback",
        "clock",
        "runner",
        "scorer",
        "verifier",
    )
    for operation in operations:
        signature = inspect.signature(operation)
        assert all(
            not any(fragment in name.lower() for fragment in forbidden_parameter_fragments)
            for name in signature.parameters
        )
