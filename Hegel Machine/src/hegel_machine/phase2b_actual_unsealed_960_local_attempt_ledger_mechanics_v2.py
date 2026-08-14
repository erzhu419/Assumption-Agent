"""Local OPEN-to-TERMINAL attempt-ledger mechanics for Phase-2B.

This module persists exactly one attempt-0 OPEN record and, later, exactly one
TERMINAL_CONSUMED record in a caller-owned directory.  It uses a pinned
directory descriptor, a cooperating-process advisory lock, file and directory
``fsync``, and hard-link no-replace publication.  The terminal transition
internally re-verifies the existing TEST_ONLY signed attempt chain.

The result is deliberately narrow.  It does not establish an external trust
anchor, custodian identity, hostile-owner tamper or rollback resistance,
hardware/remote-filesystem durability, observed time, execution, scoring,
formal gates, effect, or C1 evidence.  An OPEN receipt is not execution
authorization and an already-present OPEN must never be interpreted as proof
that execution did not occur.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import errno
import fcntl
import hashlib
import json
import os
from pathlib import PosixPath
import re
import stat
from typing import Final

from .hashing import canonical_json
from .phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2 import (
    PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_CLAIM_LEVEL,
    PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_POLICY_ID,
    PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_SCHEMA_ID,
    PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_VERSION,
    PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_TEST_ANCHOR_SCOPE_V2,
    ActualUnsealed960AttemptEventV2,
    ActualUnsealed960SignedAttemptChainMechanicsV2,
    ActualUnsealed960SignedAttemptChainV2,
    ActualUnsealed960SignerKeyV2,
    ActualUnsealed960TestOnlyAnchorV2,
    SignedAttemptChainDispositionV2,
    SignedAttemptChainReasonV2,
    SignedAttemptSignerRoleV2,
    SignedAttemptStageV2,
    verify_phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2,
)


PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_VERSION: Final = (
    "hegel-machine-phase2b-actual-unsealed-960-local-attempt-ledger-mechanics/2"
)
PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_CLAIM_LEVEL: Final = (
    "NON_AUTHORITATIVE_LOCAL_FILESYSTEM_ATTEMPT_LEDGER_MECHANICS_ONLY"
)
PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_SCOPE_V2: Final = (
    "CALLER_OWNED_SINGLE_DIRECTORY_ATTEMPT_ZERO_ONLY"
)

_MAX_TEXT_BYTES: Final = 4_096
_MAX_PATH_COMPONENT_BYTES: Final = 255
_MAX_PATH_COMPONENTS: Final = 64
_MAX_PATH_BYTES: Final = 4_096
_MAX_RECORD_BYTES: Final = 16_384
_OPEN_TARGET: Final = "attempt-000.open.json"
_OPEN_PENDING: Final = ".attempt-000.open.json.next"
_TERMINAL_TARGET: Final = "attempt-000.terminal.json"
_TERMINAL_PENDING: Final = ".attempt-000.terminal.json.next"
_ALLOWED_NAMES: Final = frozenset(
    {_OPEN_TARGET, _OPEN_PENDING, _TERMINAL_TARGET, _TERMINAL_PENDING}
)

_LEDGER_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/LOCAL_LEDGER/LEDGER/V2\x00"
_OPEN_RECORD_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/LOCAL_LEDGER/OPEN/V2\x00"
_TERMINAL_RECORD_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/LOCAL_LEDGER/TERMINAL/V2\x00"
_RESULT_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/LOCAL_LEDGER/RESULT/V2\x00"
_SCHEMA_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/LOCAL_LEDGER/SCHEMA/V2\x00"
_POLICY_DOMAIN: Final = b"HEGEL/PHASE2B/ACTUAL960/LOCAL_LEDGER/POLICY/V2\x00"

_LEDGER_PREFIX: Final = "phase2b_actual_unsealed_960_local_attempt_ledger_v2_"
_OPEN_RECORD_PREFIX: Final = "phase2b_actual_unsealed_960_local_open_record_v2_"
_TERMINAL_RECORD_PREFIX: Final = "phase2b_actual_unsealed_960_local_terminal_record_v2_"
_RESULT_PREFIX: Final = "phase2b_actual_unsealed_960_local_attempt_ledger_result_v2_"
_SCHEMA_PREFIX: Final = "phase2b_actual_unsealed_960_local_attempt_ledger_schema_v2_"
_POLICY_PREFIX: Final = "phase2b_actual_unsealed_960_local_attempt_ledger_policy_v2_"
_SIGNED_CHAIN_RESULT_DOMAIN: Final = (
    b"HEGEL/PHASE2B/ACTUAL960/SIGNED_CHAIN/RESULT/V2\x00"
)
_SIGNED_CHAIN_RESULT_PREFIX: Final = (
    "phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2_"
)


class LocalAttemptLedgerStateV2(str, Enum):
    EMPTY = "EMPTY"
    OPEN = "OPEN"
    TERMINAL_CONSUMED = "TERMINAL_CONSUMED"


class LocalAttemptLedgerDispositionV2(str, Enum):
    LOCAL_ATTEMPT_ZERO_OPEN_PUBLISHED = "LOCAL_ATTEMPT_ZERO_OPEN_PUBLISHED"
    LOCAL_ATTEMPT_ZERO_OPEN_RECOVERED = "LOCAL_ATTEMPT_ZERO_OPEN_RECOVERED"
    LOCAL_ATTEMPT_ZERO_OPEN_ALREADY_PRESENT_IDENTICAL = (
        "LOCAL_ATTEMPT_ZERO_OPEN_ALREADY_PRESENT_IDENTICAL"
    )
    LOCAL_ATTEMPT_ZERO_TERMINAL_PUBLISHED = "LOCAL_ATTEMPT_ZERO_TERMINAL_PUBLISHED"
    LOCAL_ATTEMPT_ZERO_TERMINAL_RECOVERED = "LOCAL_ATTEMPT_ZERO_TERMINAL_RECOVERED"
    LOCAL_ATTEMPT_ZERO_TERMINAL_ALREADY_PRESENT_IDENTICAL = (
        "LOCAL_ATTEMPT_ZERO_TERMINAL_ALREADY_PRESENT_IDENTICAL"
    )
    LOCAL_ATTEMPT_LEDGER_EMPTY_VERIFIED = "LOCAL_ATTEMPT_LEDGER_EMPTY_VERIFIED"
    LOCAL_ATTEMPT_LEDGER_OPEN_VERIFIED = "LOCAL_ATTEMPT_LEDGER_OPEN_VERIFIED"
    LOCAL_ATTEMPT_LEDGER_TERMINAL_VERIFIED = "LOCAL_ATTEMPT_LEDGER_TERMINAL_VERIFIED"
    REJECTED = "REJECTED"


class LocalAttemptLedgerReasonV2(str, Enum):
    OPEN_RECORD_FILE_AND_DIRECTORY_FSYNC_PUBLICATION_COMPLETED_NOT_EXECUTION_AUTHORIZATION = (
        "OPEN_RECORD_FILE_AND_DIRECTORY_FSYNC_PUBLICATION_COMPLETED_NOT_EXECUTION_AUTHORIZATION"
    )
    OPEN_RECORD_RECOVERED_NOT_EXECUTION_AUTHORIZATION = (
        "OPEN_RECORD_RECOVERED_NOT_EXECUTION_AUTHORIZATION"
    )
    IDENTICAL_OPEN_ALREADY_PRESENT_EXECUTION_STATE_UNKNOWN = (
        "IDENTICAL_OPEN_ALREADY_PRESENT_EXECUTION_STATE_UNKNOWN"
    )
    TERMINAL_RECORD_FILE_AND_DIRECTORY_FSYNC_PUBLICATION_COMPLETED_LOCAL_MECHANICS_ONLY = (
        "TERMINAL_RECORD_FILE_AND_DIRECTORY_FSYNC_PUBLICATION_COMPLETED_LOCAL_MECHANICS_ONLY"
    )
    TERMINAL_RECORD_RECOVERED_LOCAL_MECHANICS_ONLY = (
        "TERMINAL_RECORD_RECOVERED_LOCAL_MECHANICS_ONLY"
    )
    IDENTICAL_TERMINAL_ALREADY_PRESENT_LOCAL_MECHANICS_ONLY = (
        "IDENTICAL_TERMINAL_ALREADY_PRESENT_LOCAL_MECHANICS_ONLY"
    )
    EMPTY_DIRECTORY_VERIFIED = "EMPTY_DIRECTORY_VERIFIED"
    OPEN_RECORD_VERIFIED_EXECUTION_STATE_UNKNOWN = (
        "OPEN_RECORD_VERIFIED_EXECUTION_STATE_UNKNOWN"
    )
    TERMINAL_RECORD_VERIFIED_LOCAL_MECHANICS_ONLY = (
        "TERMINAL_RECORD_VERIFIED_LOCAL_MECHANICS_ONLY"
    )
    WRONG_INPUT_TYPE = "WRONG_INPUT_TYPE"
    CROSS_VERSION_INPUT = "CROSS_VERSION_INPUT"
    TEST_ANCHOR_INVALID = "TEST_ANCHOR_INVALID"
    SIGNED_CHAIN_REJECTED = "SIGNED_CHAIN_REJECTED"
    STATE_DIRECTORY_INVALID = "STATE_DIRECTORY_INVALID"
    UNSUPPORTED_PLATFORM = "UNSUPPORTED_PLATFORM"
    DIRECTORY_LOCK_BUSY = "DIRECTORY_LOCK_BUSY"
    NAMESPACE_TAMPERED = "NAMESPACE_TAMPERED"
    OPEN_RECORD_REQUIRED = "OPEN_RECORD_REQUIRED"
    TERMINAL_STATE_ALREADY_CLOSED = "TERMINAL_STATE_ALREADY_CLOSED"
    DIFFERENT_ATTEMPT_OR_CHAIN_CONFLICT = "DIFFERENT_ATTEMPT_OR_CHAIN_CONFLICT"
    RECOVERY_INPUT_REQUIRED = "RECOVERY_INPUT_REQUIRED"
    CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED = (
        "CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED"
    )
    FILESYSTEM_OPERATION_FAILED_RECOVERY_REQUIRED = (
        "FILESYSTEM_OPERATION_FAILED_RECOVERY_REQUIRED"
    )
    INTERNAL_ERROR = "INTERNAL_ERROR"


_LOCAL_CLAIMS: Final = (
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

_NONCLAIMS: Final = (
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


@dataclass(frozen=True, slots=True, init=False)
class ActualUnsealed960LocalAttemptLedgerMechanicsV2:
    disposition: LocalAttemptLedgerDispositionV2
    reason: LocalAttemptLedgerReasonV2
    version: str
    schema_id: str
    policy_id: str
    claim_level: str
    scope: str
    result_id: str
    ledger_id: str
    attempt_index: int
    state: LocalAttemptLedgerStateV2
    anchor_id: str
    attempt_id: str
    open_record_sha256: str | None
    open_record_id: str | None
    terminal_record_sha256: str | None
    terminal_record_id: str | None
    signed_chain_result_id: str | None
    chain_root_sha256: str | None
    chain_id: str | None
    raw_input_committed_payload_root_sha256: str | None
    package_commitments_frozen_payload_root_sha256: str | None
    newly_published: bool
    recovered_commit_window: bool
    identical_existing_state_observed: bool
    test_only_anchor_content_reverified: bool
    test_only_signed_chain_reverified_for_terminal: bool
    owned_nofollow_directory_descriptor_pinned: bool
    cooperating_process_directory_lock_held: bool
    fixed_namespace_allowlist_verified: bool
    attempt_zero_open_record_content_address_verified: bool
    attempt_zero_terminal_record_content_address_verified: bool
    open_record_file_fsync_completed: bool
    terminal_record_file_fsync_completed: bool
    directory_fsync_completed: bool
    hardlink_no_replace_publication_verified: bool
    open_before_terminal_predecessor_verified: bool
    single_directory_terminal_exclusion_mechanics_verified: bool
    strict_canonical_record_replay_verified: bool
    atomic_result_issuance_verified: bool
    external_trust_anchor_verified: bool
    external_signature_authority_verified: bool
    qualified_cryptographic_backend_verified: bool
    custodian_identity_verified: bool
    signer_independence_verified: bool
    externally_durable_attempt_ledger_verified: bool
    hardware_power_loss_durability_qualified: bool
    remote_filesystem_durability_qualified: bool
    hostile_same_uid_writer_tamper_resistance_verified: bool
    owner_deletion_resistance_verified: bool
    owner_rollback_resistance_verified: bool
    cross_host_global_one_shot_verified: bool
    authoritative_attempt_denominator_verified: bool
    attempt_registry_complete: bool
    retry_authorization_verified: bool
    rerun_policy_enforced: bool
    one_shot_policy_enforced: bool
    external_timestamp_authority_verified: bool
    timeline_observed: bool
    authoritative_timeline_order_verified: bool
    authoritative_attempt_terminal_state_verified: bool
    terminal_append_receipt_verified: bool
    campaign_finalization_verified: bool
    actual_evidence_inputs_accepted: bool
    admission_evidence_contract_complete: bool
    authoritative_evidence_verifier_implemented: bool
    admission_ready: bool
    execution_authorized: bool
    evidence_supplied: bool
    evidence_verified: bool
    input_archive_authority_verified: bool
    prediction_archive_authority_verified: bool
    answer_commitment_authority_verified: bool
    gate_input_commitment_authority_verified: bool
    pre_reveal_commitment_timing_verified: bool
    runtime_attestation_authority_verified: bool
    recognizer_executed: bool
    runtime_executed: bool
    actual_960_case_run_verified: bool
    origin_authenticated: bool
    scoring_performed: bool
    prediction_scored: bool
    actual_prediction_scoring_evidence: bool
    formal_gate_evaluation_performed: bool
    overall_gate_results_materialized: bool
    slice_gate_results_materialized: bool
    preservation_evaluated: bool
    scale_regret_evaluated: bool
    bootstrap_evaluated: bool
    effect_evidence: bool
    formal_c1_report_verified: bool
    c1_exit_evidence: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("local attempt-ledger mechanics results are privately issued")


@dataclass(frozen=True, slots=True, init=False)
class ActualUnsealed960LocalAttemptLedgerRejectionV2:
    disposition: LocalAttemptLedgerDispositionV2
    reason: LocalAttemptLedgerReasonV2
    version: str
    schema_id: str
    policy_id: str
    claim_level: str
    scope: str
    validation: None
    result_id: None
    ledger_id: None
    attempt_index: int
    state: None
    anchor_id: None
    attempt_id: None
    open_record_sha256: None
    open_record_id: None
    terminal_record_sha256: None
    terminal_record_id: None
    signed_chain_result_id: None
    chain_root_sha256: None
    chain_id: None
    raw_input_committed_payload_root_sha256: None
    package_commitments_frozen_payload_root_sha256: None
    newly_published: bool
    recovered_commit_window: bool
    identical_existing_state_observed: bool
    partial_result_graph_published: bool
    filesystem_recovery_required: bool
    test_only_anchor_content_reverified: bool
    test_only_signed_chain_reverified_for_terminal: bool
    owned_nofollow_directory_descriptor_pinned: bool
    cooperating_process_directory_lock_held: bool
    fixed_namespace_allowlist_verified: bool
    attempt_zero_open_record_content_address_verified: bool
    attempt_zero_terminal_record_content_address_verified: bool
    open_record_file_fsync_completed: bool
    terminal_record_file_fsync_completed: bool
    directory_fsync_completed: bool
    hardlink_no_replace_publication_verified: bool
    open_before_terminal_predecessor_verified: bool
    single_directory_terminal_exclusion_mechanics_verified: bool
    strict_canonical_record_replay_verified: bool
    atomic_result_issuance_verified: bool
    external_trust_anchor_verified: bool
    external_signature_authority_verified: bool
    qualified_cryptographic_backend_verified: bool
    custodian_identity_verified: bool
    signer_independence_verified: bool
    externally_durable_attempt_ledger_verified: bool
    hardware_power_loss_durability_qualified: bool
    remote_filesystem_durability_qualified: bool
    hostile_same_uid_writer_tamper_resistance_verified: bool
    owner_deletion_resistance_verified: bool
    owner_rollback_resistance_verified: bool
    cross_host_global_one_shot_verified: bool
    authoritative_attempt_denominator_verified: bool
    attempt_registry_complete: bool
    retry_authorization_verified: bool
    rerun_policy_enforced: bool
    one_shot_policy_enforced: bool
    external_timestamp_authority_verified: bool
    timeline_observed: bool
    authoritative_timeline_order_verified: bool
    authoritative_attempt_terminal_state_verified: bool
    terminal_append_receipt_verified: bool
    campaign_finalization_verified: bool
    actual_evidence_inputs_accepted: bool
    admission_evidence_contract_complete: bool
    authoritative_evidence_verifier_implemented: bool
    admission_ready: bool
    execution_authorized: bool
    evidence_supplied: bool
    evidence_verified: bool
    input_archive_authority_verified: bool
    prediction_archive_authority_verified: bool
    answer_commitment_authority_verified: bool
    gate_input_commitment_authority_verified: bool
    pre_reveal_commitment_timing_verified: bool
    runtime_attestation_authority_verified: bool
    recognizer_executed: bool
    runtime_executed: bool
    actual_960_case_run_verified: bool
    origin_authenticated: bool
    scoring_performed: bool
    prediction_scored: bool
    actual_prediction_scoring_evidence: bool
    formal_gate_evaluation_performed: bool
    overall_gate_results_materialized: bool
    slice_gate_results_materialized: bool
    preservation_evaluated: bool
    scale_regret_evaluated: bool
    bootstrap_evaluated: bool
    effect_evidence: bool
    formal_c1_report_verified: bool
    c1_exit_evidence: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("local attempt-ledger rejections are privately issued")


_SUCCESS_FIELDS: Final = tuple(
    item.name for item in fields(ActualUnsealed960LocalAttemptLedgerMechanicsV2)
)
_REJECTION_FIELDS: Final = tuple(
    item.name for item in fields(ActualUnsealed960LocalAttemptLedgerRejectionV2)
)
_SUCCESS_TYPE_MANIFEST: Final = tuple(
    (item.name, str(item.type))
    for item in fields(ActualUnsealed960LocalAttemptLedgerMechanicsV2)
)
_REJECTION_TYPE_MANIFEST: Final = tuple(
    (item.name, str(item.type))
    for item in fields(ActualUnsealed960LocalAttemptLedgerRejectionV2)
)
_SIGNED_CHAIN_RESULT_FIELDS: Final = tuple(
    item.name for item in fields(ActualUnsealed960SignedAttemptChainMechanicsV2)
)
_SIGNED_CHAIN_TRUE_CLAIMS: Final = (
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
_SIGNED_CHAIN_NONBOOL_FIELDS: Final = (
    "disposition", "reason", "version", "schema_id", "policy_id",
    "claim_level", "result_id", "anchor_id", "attempt_id", "signer_key_id",
    "raw_input_committed_payload_root_sha256",
    "package_commitments_frozen_payload_root_sha256", "chain_root_sha256",
    "chain_id", "event_count", "terminal_stage",
)
_INPUT_TYPE_MANIFEST: Final = (
    ("ledger_directory", "exact_pathlib.PosixPath_absolute_existing_directory"),
    ("anchor", "exact_ActualUnsealed960TestOnlyAnchorV2"),
    ("chain", "exact_ActualUnsealed960SignedAttemptChainV2_or_exact_None"),
)


def _text(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact string")
    encoded = value.encode("utf-8")
    if not 1 <= len(encoded) <= _MAX_TEXT_BYTES:
        raise ValueError(f"{label} UTF-8 size drift")
    return value


def _sha(value: object, label: str) -> str:
    text = _text(value, label)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return text


def _id(value: object, prefix: str, label: str) -> str:
    text = _text(value, label)
    if not text.startswith(prefix):
        raise ValueError(f"{label} prefix drift")
    _sha(text[len(prefix) :], f"{label} suffix")
    return text


def _digest(domain: bytes, pairs: tuple[tuple[str, object], ...]) -> str:
    return hashlib.sha256(domain + canonical_json(pairs).encode("utf-8")).hexdigest()


def _preflight_directory(value: object) -> tuple[str, ...]:
    if type(value) is not PosixPath:
        raise TypeError("ledger_directory must be an exact pathlib.PosixPath")
    if not value.is_absolute():
        raise ValueError("ledger_directory must be absolute")
    parts = value.parts
    if len(parts) <= 1 or parts[0] != "/":
        raise ValueError("ledger_directory must name a non-root directory")
    if len(parts) - 1 > _MAX_PATH_COMPONENTS:
        raise ValueError("ledger_directory component-count cap exceeded")
    closed: list[str] = []
    total_bytes = 1
    for component in parts[1:]:
        if type(component) is not str or component in {"", ".", ".."}:
            raise ValueError("ledger_directory component drift")
        encoded = component.encode("utf-8")
        if not 1 <= len(encoded) <= _MAX_PATH_COMPONENT_BYTES or "/" in component:
            raise ValueError("ledger_directory component size drift")
        total_bytes += len(encoded) + 1
        if total_bytes > _MAX_PATH_BYTES:
            raise ValueError("ledger_directory total UTF-8 cap exceeded")
        closed.append(component)
    return tuple(closed)


def _close_anchor_primitives(anchor: object) -> dict[str, object]:
    if type(anchor) is not ActualUnsealed960TestOnlyAnchorV2:
        raise TypeError("anchor exact type drift")
    names = (
        "version", "claim_level", "anchor_scope", "schema_id", "policy_id",
        "protocol_id", "exact_freeze_id", "execution_freeze_manifest_id",
        "attempt_nonce_sha256", "attempt_id", "signer_keys",
        "signer_key_ids_root", "anchor_root_sha256", "anchor_id",
    )
    raw = {name: object.__getattribute__(anchor, name) for name in names}
    for name in names:
        if name == "signer_keys":
            continue
        if type(raw[name]) is not str:
            raise TypeError(f"anchor {name} exact string drift")
        _text(raw[name], f"anchor {name}")
    if type(raw["signer_keys"]) is not tuple or len(raw["signer_keys"]) != 1:
        raise TypeError("anchor signer_keys exact tuple drift")
    signer = raw["signer_keys"][0]
    if type(signer) is not ActualUnsealed960SignerKeyV2:
        raise TypeError("anchor signer key exact type drift")
    role = object.__getattribute__(signer, "role")
    public_key = object.__getattribute__(signer, "public_key")
    key_id = object.__getattribute__(signer, "key_id")
    if type(role) is not SignedAttemptSignerRoleV2:
        raise TypeError("anchor signer role exact type drift")
    if type(public_key) is not bytes or len(public_key) != 32:
        raise TypeError("anchor public key exact bytes drift")
    if type(key_id) is not str:
        raise TypeError("anchor key ID exact string drift")
    _text(key_id, "anchor signer key ID")
    raw["signer_keys"] = ((role, bytes(public_key), key_id),)
    return raw


def _materialize_anchor(raw: dict[str, object]) -> ActualUnsealed960TestOnlyAnchorV2:
    role, public_key, key_id = raw["signer_keys"][0]
    signer = ActualUnsealed960SignerKeyV2(
        role=role, public_key=public_key, key_id=key_id
    )
    return ActualUnsealed960TestOnlyAnchorV2(
        version=raw["version"], claim_level=raw["claim_level"],
        anchor_scope=raw["anchor_scope"], schema_id=raw["schema_id"],
        policy_id=raw["policy_id"], protocol_id=raw["protocol_id"],
        exact_freeze_id=raw["exact_freeze_id"],
        execution_freeze_manifest_id=raw["execution_freeze_manifest_id"],
        attempt_nonce_sha256=raw["attempt_nonce_sha256"],
        attempt_id=raw["attempt_id"], signer_keys=(signer,),
        signer_key_ids_root=raw["signer_key_ids_root"],
        anchor_root_sha256=raw["anchor_root_sha256"], anchor_id=raw["anchor_id"],
    )


def _close_chain_primitives(chain: object) -> dict[str, object]:
    if type(chain) is not ActualUnsealed960SignedAttemptChainV2:
        raise TypeError("chain exact type drift")
    raw = {
        name: object.__getattribute__(chain, name)
        for name in ("version", "anchor_id", "attempt_id", "events", "chain_root_sha256", "chain_id")
    }
    for name in ("version", "anchor_id", "attempt_id", "chain_root_sha256", "chain_id"):
        if type(raw[name]) is not str:
            raise TypeError(f"chain {name} exact string drift")
        _text(raw[name], f"chain {name}")
    if type(raw["events"]) is not tuple or len(raw["events"]) != 10:
        raise TypeError("chain events exact tuple drift")
    closed_events: list[tuple[object, ...]] = []
    event_names = (
        "version", "anchor_id", "attempt_id", "event_index", "stage",
        "predecessor_event_id", "payload_root_sha256", "signer_key_id",
        "signature", "event_id",
    )
    for event in raw["events"]:
        if type(event) is not ActualUnsealed960AttemptEventV2:
            raise TypeError("chain event exact type drift")
        values = tuple(object.__getattribute__(event, name) for name in event_names)
        version, anchor_id, attempt_id, index, stage, predecessor, payload, key_id, signature, event_id = values
        for label, value in (
            ("version", version), ("anchor_id", anchor_id),
            ("attempt_id", attempt_id), ("payload_root", payload),
            ("signer_key_id", key_id), ("event_id", event_id),
        ):
            if type(value) is not str:
                raise TypeError(f"event {label} exact string drift")
            _text(value, f"event {label}")
        if type(index) is not int or not 0 <= index <= 9:
            raise TypeError("event index exact integer drift")
        if type(stage) is not SignedAttemptStageV2:
            raise TypeError("event stage exact enum drift")
        if predecessor is not None and type(predecessor) is not str:
            raise TypeError("event predecessor exact optional-string drift")
        if predecessor is not None:
            _text(predecessor, "event predecessor")
        if type(signature) is not bytes or len(signature) != 64:
            raise TypeError("event signature exact bytes drift")
        closed_events.append(
            (version, anchor_id, attempt_id, index, stage, predecessor, payload,
             key_id, bytes(signature), event_id)
        )
    raw["events"] = tuple(closed_events)
    return raw


def _materialize_chain(raw: dict[str, object]) -> ActualUnsealed960SignedAttemptChainV2:
    events = tuple(
        ActualUnsealed960AttemptEventV2(
            version=item[0], anchor_id=item[1], attempt_id=item[2],
            event_index=item[3], stage=item[4], predecessor_event_id=item[5],
            payload_root_sha256=item[6], signer_key_id=item[7],
            signature=item[8], event_id=item[9],
        )
        for item in raw["events"]
    )
    return ActualUnsealed960SignedAttemptChainV2(
        version=raw["version"], anchor_id=raw["anchor_id"],
        attempt_id=raw["attempt_id"], events=events,
        chain_root_sha256=raw["chain_root_sha256"], chain_id=raw["chain_id"],
    )


def _ledger_pairs(anchor: ActualUnsealed960TestOnlyAnchorV2) -> tuple[tuple[str, object], ...]:
    return (
        ("version", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_VERSION),
        ("schema_id", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_SCHEMA_ID),
        ("policy_id", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_POLICY_ID),
        ("scope", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_SCOPE_V2),
        ("signed_chain_schema_id", anchor.schema_id),
        ("signed_chain_policy_id", anchor.policy_id),
        ("protocol_id", anchor.protocol_id),
        ("exact_freeze_id", anchor.exact_freeze_id),
        ("execution_freeze_manifest_id", anchor.execution_freeze_manifest_id),
        ("anchor_id", anchor.anchor_id),
        ("attempt_id", anchor.attempt_id),
        ("attempt_index", 0),
    )


def _ledger_id(anchor: ActualUnsealed960TestOnlyAnchorV2) -> str:
    return _LEDGER_PREFIX + _digest(_LEDGER_DOMAIN, _ledger_pairs(anchor))


_OPEN_PAIR_FIELDS: Final = (
    "record_type", "version", "schema_id", "policy_id", "claim_level", "scope",
    "ledger_id", "record_sequence", "predecessor_record_id", "attempt_index",
    "state", "signed_chain_schema_id", "signed_chain_policy_id", "protocol_id",
    "exact_freeze_id", "execution_freeze_manifest_id", "attempt_nonce_sha256",
    "anchor_id", "attempt_id",
)
_TERMINAL_PAIR_FIELDS: Final = (
    "record_type", "version", "schema_id", "policy_id", "claim_level", "scope",
    "ledger_id", "record_sequence", "predecessor_record_id", "attempt_index",
    "state", "signed_chain_schema_id", "signed_chain_policy_id", "protocol_id",
    "exact_freeze_id", "execution_freeze_manifest_id", "anchor_id", "attempt_id",
    "signed_chain_result_id", "chain_root_sha256", "chain_id", "terminal_event_id",
    "raw_input_committed_payload_root_sha256",
    "package_commitments_frozen_payload_root_sha256",
)


def _record_document(
    pairs: tuple[tuple[str, object], ...], domain: bytes, prefix: str
) -> tuple[dict[str, object], bytes, str, str]:
    record_sha = _digest(domain, pairs)
    record_id = prefix + record_sha
    document = dict(pairs)
    document["record_sha256"] = record_sha
    document["record_id"] = record_id
    payload = canonical_json(document).encode("utf-8")
    if len(payload) > _MAX_RECORD_BYTES:
        raise ValueError("record payload cap exceeded")
    return document, payload, record_sha, record_id


def _open_record(anchor: ActualUnsealed960TestOnlyAnchorV2) -> tuple[dict[str, object], bytes, str, str]:
    ledger_id = _ledger_id(anchor)
    pairs = (
        ("record_type", "OPEN"),
        ("version", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_VERSION),
        ("schema_id", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_SCHEMA_ID),
        ("policy_id", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_POLICY_ID),
        ("claim_level", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_CLAIM_LEVEL),
        ("scope", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_SCOPE_V2),
        ("ledger_id", ledger_id), ("record_sequence", 0),
        ("predecessor_record_id", None), ("attempt_index", 0),
        ("state", LocalAttemptLedgerStateV2.OPEN.value),
        ("signed_chain_schema_id", anchor.schema_id),
        ("signed_chain_policy_id", anchor.policy_id),
        ("protocol_id", anchor.protocol_id),
        ("exact_freeze_id", anchor.exact_freeze_id),
        ("execution_freeze_manifest_id", anchor.execution_freeze_manifest_id),
        ("attempt_nonce_sha256", anchor.attempt_nonce_sha256),
        ("anchor_id", anchor.anchor_id), ("attempt_id", anchor.attempt_id),
    )
    if tuple(name for name, _ in pairs) != _OPEN_PAIR_FIELDS:
        raise RuntimeError("open record field drift")
    return _record_document(pairs, _OPEN_RECORD_DOMAIN, _OPEN_RECORD_PREFIX)


def _terminal_record(
    anchor: ActualUnsealed960TestOnlyAnchorV2,
    chain: ActualUnsealed960SignedAttemptChainV2,
    verified: ActualUnsealed960SignedAttemptChainMechanicsV2,
    open_record_id: str,
) -> tuple[dict[str, object], bytes, str, str]:
    pairs = (
        ("record_type", "TERMINAL"),
        ("version", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_VERSION),
        ("schema_id", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_SCHEMA_ID),
        ("policy_id", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_POLICY_ID),
        ("claim_level", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_CLAIM_LEVEL),
        ("scope", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_SCOPE_V2),
        ("ledger_id", _ledger_id(anchor)), ("record_sequence", 1),
        ("predecessor_record_id", open_record_id), ("attempt_index", 0),
        ("state", LocalAttemptLedgerStateV2.TERMINAL_CONSUMED.value),
        ("signed_chain_schema_id", anchor.schema_id),
        ("signed_chain_policy_id", anchor.policy_id),
        ("protocol_id", anchor.protocol_id),
        ("exact_freeze_id", anchor.exact_freeze_id),
        ("execution_freeze_manifest_id", anchor.execution_freeze_manifest_id),
        ("anchor_id", anchor.anchor_id), ("attempt_id", anchor.attempt_id),
        ("signed_chain_result_id", verified.result_id),
        ("chain_root_sha256", verified.chain_root_sha256),
        ("chain_id", verified.chain_id),
        ("terminal_event_id", chain.events[-1].event_id),
        ("raw_input_committed_payload_root_sha256", verified.raw_input_committed_payload_root_sha256),
        ("package_commitments_frozen_payload_root_sha256", verified.package_commitments_frozen_payload_root_sha256),
    )
    if tuple(name for name, _ in pairs) != _TERMINAL_PAIR_FIELDS:
        raise RuntimeError("terminal record field drift")
    return _record_document(pairs, _TERMINAL_RECORD_DOMAIN, _TERMINAL_RECORD_PREFIX)


def _reject_float(_: str) -> object:
    raise ValueError("floating-point JSON is forbidden")


def _reject_constant(_: str) -> object:
    raise ValueError("non-finite JSON is forbidden")


def _pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise ValueError("duplicate or non-string JSON key")
        result[key] = value
    return result


def _strict_document(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes or not 1 <= len(payload) <= _MAX_RECORD_BYTES:
        raise ValueError("record byte-size drift")
    text = payload.decode("utf-8")
    value = json.loads(
        text, object_pairs_hook=_pairs_hook, parse_float=_reject_float,
        parse_constant=_reject_constant,
    )
    if type(value) is not dict or canonical_json(value).encode("utf-8") != payload:
        raise ValueError("record is not exact canonical JSON")
    return value


def _metadata_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid,
        value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns,
    )


def _open_pinned_directory(parts: tuple[str, ...]) -> int:
    required = ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC", "O_NONBLOCK")
    if any(not hasattr(os, name) for name in required):
        raise NotImplementedError("required POSIX open flags unavailable")
    if (
        os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.unlink not in os.supports_dir_fd
        or os.link not in os.supports_dir_fd
    ):
        raise NotImplementedError("required dirfd operations unavailable")
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NONBLOCK
        | os.O_NOFOLLOW
        | os.O_CLOEXEC
    )
    current = os.open("/", flags)
    try:
        for component in parts:
            before = os.stat(component, dir_fd=current, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError("ledger_directory ancestor is not a directory")
            child = os.open(component, flags, dir_fd=current)
            after = os.fstat(child)
            if (
                not stat.S_ISDIR(after.st_mode)
                or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            ):
                os.close(child)
                raise ValueError("ledger_directory component identity drift")
            os.close(current)
            current = child
        final = os.fstat(current)
        if final.st_uid != os.geteuid() or stat.S_IMODE(final.st_mode) != 0o700:
            raise ValueError("ledger_directory must be current-euid-owned mode 0700")
        return current
    except BaseException:
        os.close(current)
        raise


def _acquire_directory_lock(directory_fd: int) -> None:
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.DIRECTORY_LOCK_BUSY) from exc


def _scan_names(directory_fd: int) -> frozenset[str]:
    names: list[str] = []
    with os.scandir(directory_fd) as iterator:
        for entry in iterator:
            if len(names) == len(_ALLOWED_NAMES):
                raise _LedgerFailure(
                    LocalAttemptLedgerReasonV2.NAMESPACE_TAMPERED, recovery=True
                )
            if type(entry.name) is not str:
                raise _LedgerFailure(
                    LocalAttemptLedgerReasonV2.NAMESPACE_TAMPERED, recovery=True
                )
            names.append(entry.name)
    present = frozenset(names)
    if len(present) != len(names):
        raise _LedgerFailure(
            LocalAttemptLedgerReasonV2.NAMESPACE_TAMPERED, recovery=True
        )
    if not present.issubset(_ALLOWED_NAMES):
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.NAMESPACE_TAMPERED, recovery=True)
    return present


def _classify_namespace(names: frozenset[str]) -> str:
    """Classify the complete four-name graph before any recovery mutation."""

    if not names:
        return "EMPTY"
    if names == {_OPEN_PENDING}:
        return "OPEN_PENDING"
    if names == {_OPEN_TARGET}:
        return "OPEN_CLEAN"
    if names == {_OPEN_TARGET, _OPEN_PENDING}:
        return "OPEN_LINK_WINDOW"
    if _TERMINAL_TARGET in names or _TERMINAL_PENDING in names:
        # Terminal publication is reachable only from a completely clean OPEN.
        if _OPEN_TARGET not in names or _OPEN_PENDING in names:
            raise _LedgerFailure(
                LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
                recovery=True,
            )
        terminal_names = names - {_OPEN_TARGET}
        if terminal_names == {_TERMINAL_PENDING}:
            return "TERMINAL_PENDING"
        if terminal_names == {_TERMINAL_TARGET}:
            return "TERMINAL_CLEAN"
        if terminal_names == {_TERMINAL_TARGET, _TERMINAL_PENDING}:
            return "TERMINAL_LINK_WINDOW"
    raise _LedgerFailure(
        LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
        recovery=True,
    )


def _validate_namespace_metadata(directory_fd: int, state: str) -> None:
    expected_links = {
        "OPEN_PENDING": ((_OPEN_PENDING, 1),),
        "OPEN_CLEAN": ((_OPEN_TARGET, 1),),
        "OPEN_LINK_WINDOW": ((_OPEN_TARGET, 2), (_OPEN_PENDING, 2)),
        "TERMINAL_PENDING": ((_OPEN_TARGET, 1), (_TERMINAL_PENDING, 1)),
        "TERMINAL_CLEAN": ((_OPEN_TARGET, 1), (_TERMINAL_TARGET, 1)),
        "TERMINAL_LINK_WINDOW": (
            (_OPEN_TARGET, 1), (_TERMINAL_TARGET, 2), (_TERMINAL_PENDING, 2),
        ),
        "EMPTY": (),
    }[state]
    metadata: dict[str, os.stat_result] = {}
    for name, expected_nlink in expected_links:
        item = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(item.st_mode)
            or item.st_uid != os.geteuid()
            or stat.S_IMODE(item.st_mode) != 0o600
            or item.st_nlink != expected_nlink
            or not 1 <= item.st_size <= _MAX_RECORD_BYTES
        ):
            raise _LedgerFailure(
                LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
                recovery=True,
            )
        metadata[name] = item
    if state == "OPEN_LINK_WINDOW" and (
        metadata[_OPEN_TARGET].st_dev,
        metadata[_OPEN_TARGET].st_ino,
    ) != (
        metadata[_OPEN_PENDING].st_dev,
        metadata[_OPEN_PENDING].st_ino,
    ):
        raise _LedgerFailure(
            LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
            recovery=True,
        )
    if state == "TERMINAL_LINK_WINDOW" and (
        metadata[_TERMINAL_TARGET].st_dev,
        metadata[_TERMINAL_TARGET].st_ino,
    ) != (
        metadata[_TERMINAL_PENDING].st_dev,
        metadata[_TERMINAL_PENDING].st_ino,
    ):
        raise _LedgerFailure(
            LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
            recovery=True,
        )


def _read_regular(
    directory_fd: int, name: str, *, expected_nlink: int
) -> bytes:
    before_path = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(before_path.st_mode)
        or before_path.st_uid != os.geteuid()
        or stat.S_IMODE(before_path.st_mode) != 0o600
        or before_path.st_nlink != expected_nlink
        or not 1 <= before_path.st_size <= _MAX_RECORD_BYTES
    ):
        raise _LedgerFailure(
            LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
            recovery=True,
        )
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != expected_nlink
            or not 1 <= before.st_size <= _MAX_RECORD_BYTES
        ):
            raise _LedgerFailure(
                LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
                recovery=True,
            )
        if _metadata_identity(before_path) != _metadata_identity(before):
            raise _LedgerFailure(
                LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
                recovery=True,
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 65_536)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_RECORD_BYTES:
                raise _LedgerFailure(
                    LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
                    recovery=True,
                )
            chunks.append(chunk)
        after = os.fstat(descriptor)
        namespace = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _metadata_identity(before) != _metadata_identity(after) or _metadata_identity(after) != _metadata_identity(namespace):
            raise _LedgerFailure(
                LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
                recovery=True,
            )
        payload = b"".join(chunks)
        try:
            _strict_document(payload)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise _LedgerFailure(
                LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
                recovery=True,
            ) from exc
        return payload
    finally:
        os.close(descriptor)


def _barrier_v2(_: str) -> None:
    """Private no-op fault-injection seam; never supplied through public API."""


def _fsync_regular_name(
    directory_fd: int, name: str, *, expected_nlink: int
) -> None:
    before_path = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(before_path.st_mode)
        or before_path.st_uid != os.geteuid()
        or stat.S_IMODE(before_path.st_mode) != 0o600
        or before_path.st_nlink != expected_nlink
    ):
        raise _LedgerFailure(
            LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
            recovery=True,
        )
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=directory_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != expected_nlink
        ):
            raise _LedgerFailure(
                LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
                recovery=True,
            )
        if _metadata_identity(before_path) != _metadata_identity(metadata):
            raise _LedgerFailure(
                LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
                recovery=True,
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_record(
    directory_fd: int,
    *,
    target: str,
    pending: str,
    payload: bytes,
    create_if_absent: bool,
) -> str:
    names = _scan_names(directory_fd)
    target_present = target in names
    pending_present = pending in names

    if target_present and pending_present:
        target_meta = os.stat(target, dir_fd=directory_fd, follow_symlinks=False)
        pending_meta = os.stat(pending, dir_fd=directory_fd, follow_symlinks=False)
        if (
            target_meta.st_nlink != 2
            or pending_meta.st_nlink != 2
            or (target_meta.st_dev, target_meta.st_ino)
            != (pending_meta.st_dev, pending_meta.st_ino)
            or _read_regular(directory_fd, target, expected_nlink=2) != payload
            or _read_regular(directory_fd, pending, expected_nlink=2) != payload
        ):
            raise _LedgerFailure(
                LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
                recovery=True,
            )
        _fsync_regular_name(directory_fd, target, expected_nlink=2)
        os.fsync(directory_fd)
        _barrier_v2("after_recovery_publish_dir_fsync_before_pending_unlink")
        os.unlink(pending, dir_fd=directory_fd)
        _barrier_v2("after_recovery_pending_unlink_before_final_dir_fsync")
        os.fsync(directory_fd)
        if _read_regular(directory_fd, target, expected_nlink=1) != payload:
            raise _LedgerFailure(
                LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
                recovery=True,
            )
        return "RECOVERED"

    if target_present:
        if _read_regular(directory_fd, target, expected_nlink=1) != payload:
            raise _LedgerFailure(
                LocalAttemptLedgerReasonV2.DIFFERENT_ATTEMPT_OR_CHAIN_CONFLICT,
                recovery=True,
            )
        # This is also the deterministic recovery action for a crash after
        # pending unlink but before the final directory fsync.  Repeating the
        # directory fsync is harmless for an already-complete publication.
        os.fsync(directory_fd)
        return "IDENTICAL"

    if pending_present:
        if _read_regular(directory_fd, pending, expected_nlink=1) != payload:
            raise _LedgerFailure(
                LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
                recovery=True,
            )
        _fsync_regular_name(directory_fd, pending, expected_nlink=1)
        os.fsync(directory_fd)
        publication = "RECOVERED"
    elif create_if_absent:
        descriptor = os.open(
            pending,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            os.fchmod(descriptor, 0o600)
            _barrier_v2("after_pending_create")
            written = 0
            while written < len(payload):
                count = os.write(descriptor, payload[written:])
                if count <= 0:
                    raise OSError(errno.EIO, "record write made no progress")
                written += count
                _barrier_v2("after_write_chunk")
            _barrier_v2("after_full_write_before_file_fsync")
            os.fsync(descriptor)
            _barrier_v2("after_file_fsync_before_pending_dir_fsync")
        finally:
            os.close(descriptor)
        if _read_regular(directory_fd, pending, expected_nlink=1) != payload:
            raise _LedgerFailure(
                LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
                recovery=True,
            )
        os.fsync(directory_fd)
        _barrier_v2("after_pending_dir_fsync_before_link")
        publication = "NEW"
    else:
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.RECOVERY_INPUT_REQUIRED)

    os.link(
        pending, target, src_dir_fd=directory_fd, dst_dir_fd=directory_fd,
        follow_symlinks=False,
    )
    _barrier_v2("after_link_before_publish_dir_fsync")
    target_meta = os.stat(target, dir_fd=directory_fd, follow_symlinks=False)
    pending_meta = os.stat(pending, dir_fd=directory_fd, follow_symlinks=False)
    if (
        target_meta.st_nlink != 2
        or pending_meta.st_nlink != 2
        or (target_meta.st_dev, target_meta.st_ino)
        != (pending_meta.st_dev, pending_meta.st_ino)
        or _read_regular(directory_fd, target, expected_nlink=2) != payload
        or _read_regular(directory_fd, pending, expected_nlink=2) != payload
    ):
        raise _LedgerFailure(
            LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
            recovery=True,
        )
    os.fsync(directory_fd)
    _barrier_v2("after_publish_dir_fsync_before_pending_unlink")
    os.unlink(pending, dir_fd=directory_fd)
    _barrier_v2("after_pending_unlink_before_final_dir_fsync")
    os.fsync(directory_fd)
    _barrier_v2("after_final_dir_fsync")
    if _read_regular(directory_fd, target, expected_nlink=1) != payload:
        raise _LedgerFailure(
            LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
            recovery=True,
        )
    return publication


class _LedgerFailure(Exception):
    def __init__(self, reason: LocalAttemptLedgerReasonV2, *, recovery: bool = False):
        super().__init__(reason.value)
        self.reason = reason
        self.recovery = recovery


def _result_pairs(values: dict[str, object]) -> tuple[tuple[str, object], ...]:
    return tuple((name, values[name]) for name in _SUCCESS_FIELDS if name != "result_id")


def _issue_success(
    *,
    disposition: LocalAttemptLedgerDispositionV2,
    reason: LocalAttemptLedgerReasonV2,
    state: LocalAttemptLedgerStateV2,
    anchor: ActualUnsealed960TestOnlyAnchorV2,
    open_record: tuple[dict[str, object], bytes, str, str] | None,
    terminal_record: tuple[dict[str, object], bytes, str, str] | None,
    verified: ActualUnsealed960SignedAttemptChainMechanicsV2 | None,
    publication: str,
) -> ActualUnsealed960LocalAttemptLedgerMechanicsV2:
    values: dict[str, object] = {
        "disposition": disposition, "reason": reason,
        "version": PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_VERSION,
        "schema_id": PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_SCHEMA_ID,
        "policy_id": PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_POLICY_ID,
        "claim_level": PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_CLAIM_LEVEL,
        "scope": PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_SCOPE_V2,
        "result_id": "",
        "ledger_id": _ledger_id(anchor), "attempt_index": 0, "state": state,
        "anchor_id": anchor.anchor_id, "attempt_id": anchor.attempt_id,
        "open_record_sha256": None if open_record is None else open_record[2],
        "open_record_id": None if open_record is None else open_record[3],
        "terminal_record_sha256": None if terminal_record is None else terminal_record[2],
        "terminal_record_id": None if terminal_record is None else terminal_record[3],
        "signed_chain_result_id": None if verified is None else verified.result_id,
        "chain_root_sha256": None if verified is None else verified.chain_root_sha256,
        "chain_id": None if verified is None else verified.chain_id,
        "raw_input_committed_payload_root_sha256": None if verified is None else verified.raw_input_committed_payload_root_sha256,
        "package_commitments_frozen_payload_root_sha256": None if verified is None else verified.package_commitments_frozen_payload_root_sha256,
        "newly_published": publication == "NEW",
        "recovered_commit_window": publication == "RECOVERED",
        "identical_existing_state_observed": publication == "IDENTICAL",
    }
    terminal = state is LocalAttemptLedgerStateV2.TERMINAL_CONSUMED
    nonempty = state is not LocalAttemptLedgerStateV2.EMPTY
    local_values = {
        "test_only_anchor_content_reverified": True,
        "test_only_signed_chain_reverified_for_terminal": terminal,
        "owned_nofollow_directory_descriptor_pinned": True,
        "cooperating_process_directory_lock_held": True,
        "fixed_namespace_allowlist_verified": True,
        "attempt_zero_open_record_content_address_verified": nonempty,
        "attempt_zero_terminal_record_content_address_verified": terminal,
        "open_record_file_fsync_completed": (
            state is LocalAttemptLedgerStateV2.OPEN
            and publication in {"NEW", "RECOVERED"}
        ),
        "terminal_record_file_fsync_completed": (
            terminal and publication in {"NEW", "RECOVERED"}
        ),
        "directory_fsync_completed": publication in {"NEW", "RECOVERED"},
        "hardlink_no_replace_publication_verified": publication in {"NEW", "RECOVERED"},
        "open_before_terminal_predecessor_verified": terminal,
        "single_directory_terminal_exclusion_mechanics_verified": terminal,
        "strict_canonical_record_replay_verified": nonempty,
        "atomic_result_issuance_verified": True,
    }
    values.update(local_values)
    values.update((name, False) for name in _NONCLAIMS)
    values["result_id"] = _RESULT_PREFIX + _digest(_RESULT_DOMAIN, _result_pairs(values))
    if tuple(values) != _SUCCESS_FIELDS:
        raise RuntimeError("local ledger success field drift")
    result = object.__new__(ActualUnsealed960LocalAttemptLedgerMechanicsV2)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def _issue_rejection(
    reason: LocalAttemptLedgerReasonV2, *, recovery: bool = False
) -> ActualUnsealed960LocalAttemptLedgerRejectionV2:
    values: dict[str, object] = {
        "disposition": LocalAttemptLedgerDispositionV2.REJECTED, "reason": reason,
        "version": PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_VERSION,
        "schema_id": PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_SCHEMA_ID,
        "policy_id": PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_POLICY_ID,
        "claim_level": PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_CLAIM_LEVEL,
        "scope": PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_SCOPE_V2,
        "validation": None, "result_id": None, "ledger_id": None,
        "attempt_index": 0, "state": None, "anchor_id": None, "attempt_id": None,
        "open_record_sha256": None, "open_record_id": None,
        "terminal_record_sha256": None, "terminal_record_id": None,
        "signed_chain_result_id": None, "chain_root_sha256": None, "chain_id": None,
        "raw_input_committed_payload_root_sha256": None,
        "package_commitments_frozen_payload_root_sha256": None,
        "newly_published": False, "recovered_commit_window": False,
        "identical_existing_state_observed": False,
        "partial_result_graph_published": False,
        "filesystem_recovery_required": recovery,
    }
    values.update((name, False) for name in (*_LOCAL_CLAIMS, *_NONCLAIMS))
    if tuple(values) != _REJECTION_FIELDS:
        raise RuntimeError("local ledger rejection field drift")
    result = object.__new__(ActualUnsealed960LocalAttemptLedgerRejectionV2)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


_SCHEMA_PREIMAGE: Final = (
    ("version", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_VERSION),
    ("claim_level", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_CLAIM_LEVEL),
    ("scope", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_SCOPE_V2),
    ("success_fields", _SUCCESS_FIELDS), ("rejection_fields", _REJECTION_FIELDS),
    ("success_type_manifest", _SUCCESS_TYPE_MANIFEST),
    ("rejection_type_manifest", _REJECTION_TYPE_MANIFEST),
    ("input_type_manifest", _INPUT_TYPE_MANIFEST),
    ("signed_chain_result_fields", _SIGNED_CHAIN_RESULT_FIELDS),
    ("signed_chain_result_nonbool_fields", _SIGNED_CHAIN_NONBOOL_FIELDS),
    ("signed_chain_result_true_claims", _SIGNED_CHAIN_TRUE_CLAIMS),
    ("state_values", tuple(item.value for item in LocalAttemptLedgerStateV2)),
    ("disposition_values", tuple(item.value for item in LocalAttemptLedgerDispositionV2)),
    ("reason_values", tuple(item.value for item in LocalAttemptLedgerReasonV2)),
    ("open_record_pair_fields", _OPEN_PAIR_FIELDS),
    ("terminal_record_pair_fields", _TERMINAL_PAIR_FIELDS),
    ("fixed_names", tuple(sorted(_ALLOWED_NAMES))),
    ("max_text_bytes", _MAX_TEXT_BYTES),
    ("max_path_component_bytes", _MAX_PATH_COMPONENT_BYTES),
    ("max_path_components", _MAX_PATH_COMPONENTS),
    ("max_path_bytes", _MAX_PATH_BYTES),
    ("max_namespace_entries", len(_ALLOWED_NAMES)),
    ("max_record_bytes", _MAX_RECORD_BYTES),
)
PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_SCHEMA_ID: Final = (
    _SCHEMA_PREFIX + _digest(_SCHEMA_DOMAIN, _SCHEMA_PREIMAGE)
)

_POLICY_PREIMAGE: Final = (
    ("version", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_VERSION),
    ("claim_level", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_CLAIM_LEVEL),
    ("scope", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_SCOPE_V2),
    ("schema_id", PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_SCHEMA_ID),
    ("signed_chain_version", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_VERSION),
    ("signed_chain_schema_id", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_SCHEMA_ID),
    ("signed_chain_policy_id", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_POLICY_ID),
    ("signed_chain_claim_level", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_CLAIM_LEVEL),
    ("signed_chain_anchor_scope", PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_TEST_ANCHOR_SCOPE_V2),
    ("content_address_parameters", (
        ("ledger_domain_hex", _LEDGER_DOMAIN.hex()),
        ("open_record_domain_hex", _OPEN_RECORD_DOMAIN.hex()),
        ("terminal_record_domain_hex", _TERMINAL_RECORD_DOMAIN.hex()),
        ("result_domain_hex", _RESULT_DOMAIN.hex()),
        ("signed_chain_result_domain_hex", _SIGNED_CHAIN_RESULT_DOMAIN.hex()),
        ("schema_domain_hex", _SCHEMA_DOMAIN.hex()),
        ("policy_domain_hex", _POLICY_DOMAIN.hex()),
        ("ledger_prefix", _LEDGER_PREFIX),
        ("open_record_prefix", _OPEN_RECORD_PREFIX),
        ("terminal_record_prefix", _TERMINAL_RECORD_PREFIX),
        ("result_prefix", _RESULT_PREFIX),
        ("signed_chain_result_prefix", _SIGNED_CHAIN_RESULT_PREFIX),
        ("schema_prefix", _SCHEMA_PREFIX),
        ("policy_prefix", _POLICY_PREFIX),
    )),
    ("public_API", (
        "open_keyword_only_ledger_directory_anchor",
        "terminalize_keyword_only_ledger_directory_anchor_chain",
        "recover_keyword_only_ledger_directory_anchor_optional_chain",
        "verify_snapshot_keyword_only_ledger_directory_anchor_optional_chain",
    )),
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
    ("namespace_rule", tuple(sorted(_ALLOWED_NAMES))),
    ("namespace_state_profiles", (
        ("EMPTY", ()),
        ("OPEN_PENDING", (_OPEN_PENDING,)),
        ("OPEN_CLEAN", (_OPEN_TARGET,)),
        ("OPEN_LINK_WINDOW", (_OPEN_TARGET, _OPEN_PENDING)),
        ("TERMINAL_PENDING", (_OPEN_TARGET, _TERMINAL_PENDING)),
        ("TERMINAL_CLEAN", (_OPEN_TARGET, _TERMINAL_TARGET)),
        ("TERMINAL_LINK_WINDOW", (_OPEN_TARGET, _TERMINAL_TARGET, _TERMINAL_PENDING)),
    )),
    ("namespace_classification_rule", "entire_initial_name_and_inode_graph_validated_before_any_recovery_mutation"),
    ("publication_order", (
        "pending_O_CREAT_O_EXCL_O_NOFOLLOW", "full_write", "file_fsync",
        "pending_directory_fsync", "dirfd_hardlink_no_replace",
        "published_directory_fsync", "verified_pending_unlink", "final_directory_fsync",
    )),
    ("recovery_rule", "only_exact_pending_or_exact_target_pending_same_inode_is_recoverable"),
    ("corruption_rule", "unknown_partial_noncanonical_or_ambiguous_state_preserved_and_rejected"),
    ("record_formula", "prefix_plus_sha256(domain_plus_canonical_json_declared_order_named_pairs)"),
    ("strict_record_rule", "bounded_UTF8_duplicate_free_integer_only_exact_canonical_JSON"),
    ("ordinary_failure_rule", "no_partial_result_ids_or_true_claims_and_filesystem_evidence_preserved"),
    ("BaseException_and_process_termination_rule", "outside_ordinary_atomic_result_claim"),
    ("local_claims", _LOCAL_CLAIMS), ("nonclaims", _NONCLAIMS),
    ("forbidden_operations", (
        "os.replace", "os.rename", "filesystem_cleanup_api", "clock", "datetime",
        "random", "secrets", "network", "subprocess", "docker", "runner", "scorer",
        "decoder", "key_generation", "signature_generation", "caller_supplied_verifier",
    )),
)
PHASE2B_ACTUAL_UNSEALED_960_LOCAL_ATTEMPT_LEDGER_MECHANICS_V2_POLICY_ID: Final = (
    _POLICY_PREFIX + _digest(_POLICY_DOMAIN, _POLICY_PREIMAGE)
)


def _prepare_anchor(
    ledger_directory: object, anchor: object
) -> tuple[tuple[str, ...], dict[str, object]]:
    try:
        parts = _preflight_directory(ledger_directory)
    except TypeError:
        raise
    except (ValueError, OverflowError, UnicodeError) as exc:
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.STATE_DIRECTORY_INVALID) from exc
    if type(anchor) is not ActualUnsealed960TestOnlyAnchorV2:
        raise TypeError("anchor exact type drift")
    try:
        raw_anchor = _close_anchor_primitives(anchor)
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.TEST_ANCHOR_INVALID) from exc
    if raw_anchor["version"] != PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_VERSION:
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.CROSS_VERSION_INPUT)
    return parts, raw_anchor


def _prepare_anchor_and_chain_globally(
    ledger_directory: object,
    anchor: object,
    chain: object,
) -> tuple[
    tuple[str, ...],
    dict[str, object],
    dict[str, object],
]:
    """Close every caller-controlled primitive before the first content hash."""

    try:
        parts = _preflight_directory(ledger_directory)
    except TypeError:
        raise
    except (ValueError, OverflowError, UnicodeError) as exc:
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.STATE_DIRECTORY_INVALID) from exc
    if type(anchor) is not ActualUnsealed960TestOnlyAnchorV2 or type(chain) is not ActualUnsealed960SignedAttemptChainV2:
        raise TypeError("anchor/chain exact type drift")
    try:
        raw_anchor = _close_anchor_primitives(anchor)
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.TEST_ANCHOR_INVALID) from exc
    try:
        raw_chain = _close_chain_primitives(chain)
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.SIGNED_CHAIN_REJECTED) from exc
    versions = (
        raw_anchor["version"],
        raw_chain["version"],
        *(event[0] for event in raw_chain["events"]),
    )
    if any(
        value
        != PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_VERSION
        for value in versions
    ):
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.CROSS_VERSION_INPUT)
    return parts, raw_anchor, raw_chain


def _materialize_preflighted_anchor(
    raw_anchor: dict[str, object],
) -> ActualUnsealed960TestOnlyAnchorV2:
    try:
        return _materialize_anchor(raw_anchor)
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.TEST_ANCHOR_INVALID) from exc


def _materialize_preflighted_chain(
    raw_chain: dict[str, object],
) -> ActualUnsealed960SignedAttemptChainV2:
    try:
        return _materialize_chain(raw_chain)
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.SIGNED_CHAIN_REJECTED) from exc


def _verify_signed_chain(
    anchor: ActualUnsealed960TestOnlyAnchorV2,
    chain: ActualUnsealed960SignedAttemptChainV2,
) -> ActualUnsealed960SignedAttemptChainMechanicsV2:
    result = verify_phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2(
        anchor=anchor, chain=chain
    )
    if type(result) is not ActualUnsealed960SignedAttemptChainMechanicsV2:
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.SIGNED_CHAIN_REJECTED)
    raw = {
        name: object.__getattribute__(result, name)
        for name in _SIGNED_CHAIN_RESULT_FIELDS
    }
    if tuple(raw) != _SIGNED_CHAIN_RESULT_FIELDS:
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.SIGNED_CHAIN_REJECTED)
    expected_nonbool = {
        "disposition": SignedAttemptChainDispositionV2.SIGNED_ATTEMPT_CHAIN_MECHANICS_VERIFIED_NOT_DURABLE_AUTHORITY,
        "reason": SignedAttemptChainReasonV2.TEST_ONLY_TEN_STAGE_ED25519_CHAIN_VERIFIED,
        "version": PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_VERSION,
        "schema_id": PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_SCHEMA_ID,
        "policy_id": PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_POLICY_ID,
        "claim_level": PHASE2B_ACTUAL_UNSEALED_960_SIGNED_ATTEMPT_CHAIN_MECHANICS_V2_CLAIM_LEVEL,
        "anchor_id": anchor.anchor_id,
        "attempt_id": anchor.attempt_id,
        "signer_key_id": anchor.signer_keys[0].key_id,
        "raw_input_committed_payload_root_sha256": chain.events[1].payload_root_sha256,
        "package_commitments_frozen_payload_root_sha256": chain.events[2].payload_root_sha256,
        "chain_root_sha256": chain.chain_root_sha256,
        "chain_id": chain.chain_id,
        "event_count": 10,
        "terminal_stage": SignedAttemptStageV2.TERMINAL_CONSUMED,
    }
    preimage = (
        ("version", expected_nonbool["version"]),
        ("schema_id", expected_nonbool["schema_id"]),
        ("policy_id", expected_nonbool["policy_id"]),
        ("anchor_id", expected_nonbool["anchor_id"]),
        ("attempt_id", expected_nonbool["attempt_id"]),
        ("signer_key_id", expected_nonbool["signer_key_id"]),
        ("raw_input_committed_payload_root_sha256", expected_nonbool["raw_input_committed_payload_root_sha256"]),
        ("package_commitments_frozen_payload_root_sha256", expected_nonbool["package_commitments_frozen_payload_root_sha256"]),
        ("chain_root_sha256", expected_nonbool["chain_root_sha256"]),
        ("chain_id", expected_nonbool["chain_id"]),
        ("event_count", 10),
        ("terminal_stage", SignedAttemptStageV2.TERMINAL_CONSUMED.value),
    )
    expected_nonbool["result_id"] = _SIGNED_CHAIN_RESULT_PREFIX + _digest(
        _SIGNED_CHAIN_RESULT_DOMAIN, preimage
    )
    if type(raw["disposition"]) is not SignedAttemptChainDispositionV2:
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.SIGNED_CHAIN_REJECTED)
    if type(raw["reason"]) is not SignedAttemptChainReasonV2:
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.SIGNED_CHAIN_REJECTED)
    for name in (
        "version", "schema_id", "policy_id", "claim_level", "result_id",
        "anchor_id", "attempt_id", "signer_key_id",
        "raw_input_committed_payload_root_sha256",
        "package_commitments_frozen_payload_root_sha256", "chain_root_sha256",
        "chain_id",
    ):
        if type(raw[name]) is not str:
            raise _LedgerFailure(LocalAttemptLedgerReasonV2.SIGNED_CHAIN_REJECTED)
    if type(raw["event_count"]) is not int:
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.SIGNED_CHAIN_REJECTED)
    if type(raw["terminal_stage"]) is not SignedAttemptStageV2:
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.SIGNED_CHAIN_REJECTED)
    if any(raw[name] != value for name, value in expected_nonbool.items()):
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.SIGNED_CHAIN_REJECTED)
    bool_fields = tuple(
        name for name in _SIGNED_CHAIN_RESULT_FIELDS
        if name not in _SIGNED_CHAIN_NONBOOL_FIELDS
    )
    if any(type(raw[name]) is not bool for name in bool_fields):
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.SIGNED_CHAIN_REJECTED)
    if any(raw[name] is not True for name in _SIGNED_CHAIN_TRUE_CLAIMS):
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.SIGNED_CHAIN_REJECTED)
    if any(
        raw[name] is not False
        for name in bool_fields
        if name not in _SIGNED_CHAIN_TRUE_CLAIMS
    ):
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.SIGNED_CHAIN_REJECTED)
    return result


def _locked_directory(parts: tuple[str, ...]) -> int:
    try:
        directory_fd = _open_pinned_directory(parts)
    except NotImplementedError as exc:
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.UNSUPPORTED_PLATFORM) from exc
    except (OSError, ValueError) as exc:
        raise _LedgerFailure(LocalAttemptLedgerReasonV2.STATE_DIRECTORY_INVALID) from exc
    try:
        _acquire_directory_lock(directory_fd)
        return directory_fd
    except BaseException:
        os.close(directory_fd)
        raise


def _assert_directory_path_identity(parts: tuple[str, ...], directory_fd: int) -> None:
    """Reject a caller-visible path that was rebound while the dirfd was held."""

    try:
        rebound_fd = _open_pinned_directory(parts)
    except (NotImplementedError, OSError, ValueError) as exc:
        raise _LedgerFailure(
            LocalAttemptLedgerReasonV2.STATE_DIRECTORY_INVALID, recovery=True
        ) from exc
    try:
        held = os.fstat(directory_fd)
        rebound = os.fstat(rebound_fd)
        if (held.st_dev, held.st_ino) != (rebound.st_dev, rebound.st_ino):
            raise _LedgerFailure(
                LocalAttemptLedgerReasonV2.STATE_DIRECTORY_INVALID, recovery=True
            )
    finally:
        os.close(rebound_fd)


def open_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2(
    *, ledger_directory: PosixPath, anchor: ActualUnsealed960TestOnlyAnchorV2
) -> ActualUnsealed960LocalAttemptLedgerMechanicsV2 | ActualUnsealed960LocalAttemptLedgerRejectionV2:
    """Persist attempt-0 OPEN; this receipt is never execution authorization."""

    try:
        parts, raw_anchor = _prepare_anchor(ledger_directory, anchor)
        directory_fd = _locked_directory(parts)
        try:
            names = _scan_names(directory_fd)
            namespace_state = _classify_namespace(names)
            _validate_namespace_metadata(directory_fd, namespace_state)
            if namespace_state.startswith("TERMINAL_"):
                # OPEN never consumes or verifies a terminal chain.  Even a
                # nominally clean terminal filename is therefore an opaque
                # closed state requiring the dedicated verifier for recovery
                # classification.
                raise _LedgerFailure(
                    LocalAttemptLedgerReasonV2.TERMINAL_STATE_ALREADY_CLOSED,
                    recovery=True,
                )
            closed_anchor = _materialize_preflighted_anchor(raw_anchor)
            record = _open_record(closed_anchor)
            publication = _publish_record(
                directory_fd, target=_OPEN_TARGET, pending=_OPEN_PENDING,
                payload=record[1], create_if_absent=True,
            )
            disposition, reason = {
                "NEW": (
                    LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_ZERO_OPEN_PUBLISHED,
                    LocalAttemptLedgerReasonV2.OPEN_RECORD_FILE_AND_DIRECTORY_FSYNC_PUBLICATION_COMPLETED_NOT_EXECUTION_AUTHORIZATION,
                ),
                "RECOVERED": (
                    LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_ZERO_OPEN_RECOVERED,
                    LocalAttemptLedgerReasonV2.OPEN_RECORD_RECOVERED_NOT_EXECUTION_AUTHORIZATION,
                ),
                "IDENTICAL": (
                    LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_ZERO_OPEN_ALREADY_PRESENT_IDENTICAL,
                    LocalAttemptLedgerReasonV2.IDENTICAL_OPEN_ALREADY_PRESENT_EXECUTION_STATE_UNKNOWN,
                ),
            }[publication]
            _assert_directory_path_identity(parts, directory_fd)
            return _issue_success(
                disposition=disposition, reason=reason, state=LocalAttemptLedgerStateV2.OPEN,
                anchor=closed_anchor, open_record=record, terminal_record=None,
                verified=None, publication=publication,
            )
        finally:
            os.close(directory_fd)
    except _LedgerFailure as exc:
        return _issue_rejection(exc.reason, recovery=exc.recovery)
    except (TypeError, ValueError, OverflowError, UnicodeError):
        return _issue_rejection(LocalAttemptLedgerReasonV2.WRONG_INPUT_TYPE)
    except OSError:
        return _issue_rejection(
            LocalAttemptLedgerReasonV2.FILESYSTEM_OPERATION_FAILED_RECOVERY_REQUIRED,
            recovery=True,
        )
    except Exception:
        return _issue_rejection(LocalAttemptLedgerReasonV2.INTERNAL_ERROR, recovery=True)


def terminalize_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2(
    *,
    ledger_directory: PosixPath,
    anchor: ActualUnsealed960TestOnlyAnchorV2,
    chain: ActualUnsealed960SignedAttemptChainV2,
) -> ActualUnsealed960LocalAttemptLedgerMechanicsV2 | ActualUnsealed960LocalAttemptLedgerRejectionV2:
    """Append TERMINAL_CONSUMED after exact OPEN and signed-chain verification."""

    try:
        parts, raw_anchor, raw_chain = _prepare_anchor_and_chain_globally(
            ledger_directory, anchor, chain
        )
        directory_fd = _locked_directory(parts)
        try:
            names = _scan_names(directory_fd)
            namespace_state = _classify_namespace(names)
            _validate_namespace_metadata(directory_fd, namespace_state)
            if namespace_state not in {
                "OPEN_CLEAN", "TERMINAL_PENDING", "TERMINAL_CLEAN",
                "TERMINAL_LINK_WINDOW",
            }:
                raise _LedgerFailure(
                    LocalAttemptLedgerReasonV2.OPEN_RECORD_REQUIRED,
                    recovery=namespace_state != "EMPTY",
                )
            closed_anchor = _materialize_preflighted_anchor(raw_anchor)
            closed_chain = _materialize_preflighted_chain(raw_chain)
            open_record = _open_record(closed_anchor)
            if _read_regular(directory_fd, _OPEN_TARGET, expected_nlink=1) != open_record[1]:
                raise _LedgerFailure(
                    LocalAttemptLedgerReasonV2.DIFFERENT_ATTEMPT_OR_CHAIN_CONFLICT,
                    recovery=True,
                )
            verified = _verify_signed_chain(closed_anchor, closed_chain)
            terminal_record = _terminal_record(
                closed_anchor, closed_chain, verified, open_record[3]
            )
            publication = _publish_record(
                directory_fd, target=_TERMINAL_TARGET, pending=_TERMINAL_PENDING,
                payload=terminal_record[1], create_if_absent=True,
            )
            disposition, reason = {
                "NEW": (
                    LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_ZERO_TERMINAL_PUBLISHED,
                    LocalAttemptLedgerReasonV2.TERMINAL_RECORD_FILE_AND_DIRECTORY_FSYNC_PUBLICATION_COMPLETED_LOCAL_MECHANICS_ONLY,
                ),
                "RECOVERED": (
                    LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_ZERO_TERMINAL_RECOVERED,
                    LocalAttemptLedgerReasonV2.TERMINAL_RECORD_RECOVERED_LOCAL_MECHANICS_ONLY,
                ),
                "IDENTICAL": (
                    LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_ZERO_TERMINAL_ALREADY_PRESENT_IDENTICAL,
                    LocalAttemptLedgerReasonV2.IDENTICAL_TERMINAL_ALREADY_PRESENT_LOCAL_MECHANICS_ONLY,
                ),
            }[publication]
            _assert_directory_path_identity(parts, directory_fd)
            return _issue_success(
                disposition=disposition, reason=reason,
                state=LocalAttemptLedgerStateV2.TERMINAL_CONSUMED,
                anchor=closed_anchor, open_record=open_record,
                terminal_record=terminal_record, verified=verified,
                publication=publication,
            )
        finally:
            os.close(directory_fd)
    except _LedgerFailure as exc:
        return _issue_rejection(exc.reason, recovery=exc.recovery)
    except (TypeError, ValueError, OverflowError, UnicodeError):
        return _issue_rejection(LocalAttemptLedgerReasonV2.WRONG_INPUT_TYPE)
    except OSError:
        return _issue_rejection(
            LocalAttemptLedgerReasonV2.FILESYSTEM_OPERATION_FAILED_RECOVERY_REQUIRED,
            recovery=True,
        )
    except Exception:
        return _issue_rejection(LocalAttemptLedgerReasonV2.INTERNAL_ERROR, recovery=True)


def _inspect_or_recover(
    *,
    ledger_directory: object,
    anchor: object,
    chain: object,
    recover: bool,
) -> ActualUnsealed960LocalAttemptLedgerMechanicsV2 | ActualUnsealed960LocalAttemptLedgerRejectionV2:
    try:
        if chain is None:
            parts, raw_anchor = _prepare_anchor(ledger_directory, anchor)
            raw_chain = None
        else:
            parts, raw_anchor, raw_chain = _prepare_anchor_and_chain_globally(
                ledger_directory, anchor, chain
            )
        directory_fd = _locked_directory(parts)
        try:
            names = _scan_names(directory_fd)
            namespace_state = _classify_namespace(names)
            _validate_namespace_metadata(directory_fd, namespace_state)
            closed_anchor = _materialize_preflighted_anchor(raw_anchor)
            closed_chain = (
                None
                if raw_chain is None
                else _materialize_preflighted_chain(raw_chain)
            )
            open_record = _open_record(closed_anchor)
            if namespace_state == "EMPTY":
                _assert_directory_path_identity(parts, directory_fd)
                return _issue_success(
                    disposition=LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_LEDGER_EMPTY_VERIFIED,
                    reason=LocalAttemptLedgerReasonV2.EMPTY_DIRECTORY_VERIFIED,
                    state=LocalAttemptLedgerStateV2.EMPTY, anchor=closed_anchor,
                    open_record=None, terminal_record=None, verified=None,
                    publication="NONE",
                )

            if namespace_state in {"OPEN_PENDING", "OPEN_LINK_WINDOW"}:
                if not recover:
                    raise _LedgerFailure(
                        LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
                        recovery=True,
                    )
                publication = _publish_record(
                    directory_fd, target=_OPEN_TARGET, pending=_OPEN_PENDING,
                    payload=open_record[1], create_if_absent=False,
                )
                _assert_directory_path_identity(parts, directory_fd)
                return _issue_success(
                    disposition=LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_ZERO_OPEN_RECOVERED,
                    reason=LocalAttemptLedgerReasonV2.OPEN_RECORD_RECOVERED_NOT_EXECUTION_AUTHORIZATION,
                    state=LocalAttemptLedgerStateV2.OPEN, anchor=closed_anchor,
                    open_record=open_record, terminal_record=None, verified=None,
                    publication=publication,
                )

            # Every remaining valid state contains a completely clean OPEN.
            if _read_regular(directory_fd, _OPEN_TARGET, expected_nlink=1) != open_record[1]:
                raise _LedgerFailure(
                    LocalAttemptLedgerReasonV2.DIFFERENT_ATTEMPT_OR_CHAIN_CONFLICT,
                    recovery=True,
                )
            if namespace_state == "OPEN_CLEAN":
                publication = "IDENTICAL"
                if recover:
                    publication = _publish_record(
                        directory_fd, target=_OPEN_TARGET, pending=_OPEN_PENDING,
                        payload=open_record[1], create_if_absent=False,
                    )
                _assert_directory_path_identity(parts, directory_fd)
                return _issue_success(
                    disposition=LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_LEDGER_OPEN_VERIFIED,
                    reason=LocalAttemptLedgerReasonV2.OPEN_RECORD_VERIFIED_EXECUTION_STATE_UNKNOWN,
                    state=LocalAttemptLedgerStateV2.OPEN, anchor=closed_anchor,
                    open_record=open_record, terminal_record=None, verified=None,
                    publication=publication,
                )

            if closed_chain is None:
                raise _LedgerFailure(LocalAttemptLedgerReasonV2.RECOVERY_INPUT_REQUIRED, recovery=True)
            # Signature work remains under the same locked directory inode.
            verified = _verify_signed_chain(closed_anchor, closed_chain)
            terminal_record = _terminal_record(
                closed_anchor, closed_chain, verified, open_record[3]
            )
            publication = "IDENTICAL"
            if namespace_state in {"TERMINAL_PENDING", "TERMINAL_LINK_WINDOW"}:
                if not recover:
                    raise _LedgerFailure(
                        LocalAttemptLedgerReasonV2.CORRUPT_OR_AMBIGUOUS_STATE_RECOVERY_REQUIRED,
                        recovery=True,
                    )
                publication = _publish_record(
                    directory_fd, target=_TERMINAL_TARGET,
                    pending=_TERMINAL_PENDING, payload=terminal_record[1],
                    create_if_absent=False,
                )
            elif recover:
                publication = _publish_record(
                    directory_fd, target=_TERMINAL_TARGET,
                    pending=_TERMINAL_PENDING, payload=terminal_record[1],
                    create_if_absent=False,
                )
            elif _read_regular(directory_fd, _TERMINAL_TARGET, expected_nlink=1) != terminal_record[1]:
                raise _LedgerFailure(
                    LocalAttemptLedgerReasonV2.DIFFERENT_ATTEMPT_OR_CHAIN_CONFLICT,
                    recovery=True,
                )
            _assert_directory_path_identity(parts, directory_fd)
            return _issue_success(
                disposition=(
                    LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_ZERO_TERMINAL_RECOVERED
                    if publication == "RECOVERED"
                    else LocalAttemptLedgerDispositionV2.LOCAL_ATTEMPT_LEDGER_TERMINAL_VERIFIED
                ),
                reason=(
                    LocalAttemptLedgerReasonV2.TERMINAL_RECORD_RECOVERED_LOCAL_MECHANICS_ONLY
                    if publication == "RECOVERED"
                    else LocalAttemptLedgerReasonV2.TERMINAL_RECORD_VERIFIED_LOCAL_MECHANICS_ONLY
                ),
                state=LocalAttemptLedgerStateV2.TERMINAL_CONSUMED,
                anchor=closed_anchor, open_record=open_record,
                terminal_record=terminal_record, verified=verified,
                publication=publication,
            )
        finally:
            os.close(directory_fd)
    except _LedgerFailure as exc:
        return _issue_rejection(exc.reason, recovery=exc.recovery)
    except (TypeError, ValueError, OverflowError, UnicodeError):
        return _issue_rejection(LocalAttemptLedgerReasonV2.WRONG_INPUT_TYPE)
    except OSError:
        return _issue_rejection(
            LocalAttemptLedgerReasonV2.FILESYSTEM_OPERATION_FAILED_RECOVERY_REQUIRED,
            recovery=True,
        )
    except Exception:
        return _issue_rejection(LocalAttemptLedgerReasonV2.INTERNAL_ERROR, recovery=True)


def recover_phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2(
    *,
    ledger_directory: PosixPath,
    anchor: ActualUnsealed960TestOnlyAnchorV2,
    chain: ActualUnsealed960SignedAttemptChainV2 | None,
) -> ActualUnsealed960LocalAttemptLedgerMechanicsV2 | ActualUnsealed960LocalAttemptLedgerRejectionV2:
    """Recover only an exact recognized commit window; never infer execution."""

    return _inspect_or_recover(
        ledger_directory=ledger_directory, anchor=anchor, chain=chain, recover=True
    )


def verify_phase2b_actual_unsealed_960_local_attempt_ledger_snapshot_v2(
    *,
    ledger_directory: PosixPath,
    anchor: ActualUnsealed960TestOnlyAnchorV2,
    chain: ActualUnsealed960SignedAttemptChainV2 | None,
) -> ActualUnsealed960LocalAttemptLedgerMechanicsV2 | ActualUnsealed960LocalAttemptLedgerRejectionV2:
    """Read and independently replay a clean local ledger snapshot."""

    return _inspect_or_recover(
        ledger_directory=ledger_directory, anchor=anchor, chain=chain, recover=False
    )


for _result_type, _manifest in (
    (ActualUnsealed960LocalAttemptLedgerMechanicsV2, _SUCCESS_FIELDS),
    (ActualUnsealed960LocalAttemptLedgerRejectionV2, _REJECTION_FIELDS),
):
    if tuple(item.name for item in fields(_result_type)) != _manifest:
        raise RuntimeError("local attempt-ledger result field drift")
if set(_LOCAL_CLAIMS) & set(_NONCLAIMS):
    raise RuntimeError("local attempt-ledger claim overlap")
if not set((*_LOCAL_CLAIMS, *_NONCLAIMS)).issubset(_SUCCESS_FIELDS):
    raise RuntimeError("local attempt-ledger success claim coverage drift")
if not set((*_LOCAL_CLAIMS, *_NONCLAIMS)).issubset(_REJECTION_FIELDS):
    raise RuntimeError("local attempt-ledger rejection claim coverage drift")
if len(_ALLOWED_NAMES) != 4:
    raise RuntimeError("local attempt-ledger namespace drift")
if len({_LEDGER_DOMAIN, _OPEN_RECORD_DOMAIN, _TERMINAL_RECORD_DOMAIN, _RESULT_DOMAIN, _SCHEMA_DOMAIN, _POLICY_DOMAIN}) != 6:
    raise RuntimeError("local attempt-ledger domain separation drift")
if len({_LEDGER_PREFIX, _OPEN_RECORD_PREFIX, _TERMINAL_RECORD_PREFIX, _RESULT_PREFIX, _SCHEMA_PREFIX, _POLICY_PREFIX}) != 6:
    raise RuntimeError("local attempt-ledger prefix separation drift")


__all__ = (
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
