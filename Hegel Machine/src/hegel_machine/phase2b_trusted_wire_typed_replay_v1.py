"""Strict typed replay for keyed Phase-2B trusted-wire batches.

The diagnostic envelope entry point proves only public framing, closed-profile
decoding, lossless re-encoding, and direct exact-transform replay.  The batch
entry point first replays the supplied custodian inputs through the keyed batch
builder and then requires every emitted envelope to pass that typed replay.
Only a complete batch receives a private-issued receipt; a data or semantic
failure returns an abstaining object with no decoded rows or partial roots.

This is still mechanics evidence.  It does not authenticate the source of the
three IKM values, prove independent custody or one-shot consumption, execute a
formal covert-channel audit, or establish sealed-holdout/C1 exit evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Final

from .hashing import stable_hash
from .phase2b_exact_transform_semantics_v1 import (
    EXACT_TRANSFORM_POLICY_ID,
    EXACT_TRANSFORM_PROVENANCE_COMPILER_POLICY_ID,
    EXACT_TRANSFORM_PROVENANCE_COMPILER_VERSION,
    ExactTransformCompilation,
    PublicTransformEvidenceBundleV2,
    TransformCompilationDisposition,
    run_exact_transform_semantics,
)
from .phase2b_trusted_wire_batch_v1 import (
    MAXIMUM_BATCH_AUTHORITIES,
    TRUSTED_WIRE_BATCH_PAYLOAD_FIELDS,
    TRUSTED_WIRE_BATCH_PAYLOAD_SCHEMA_VERSION,
    TRUSTED_WIRE_BATCH_POLICY_ID,
    TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION,
    TrustedWireBatchV1,
    TrustedWireKeySourcesV1,
    TrustedWireReplayReceiptV1,
    decode_and_audit_trusted_envelope_v1,
    verify_trusted_wire_batch_replay_v1,
)
from .phase2b_trusted_wire_typed_authority_v1 import (
    TYPED_AUTHORITY_CODEC_VERSION,
    TYPED_AUTHORITY_CODEC_POLICY_ID,
    TYPED_AUTHORITY_SCHEMA_ID,
    decode_typed_transform_authority_profile_v1,
    encode_typed_transform_authority_profile_v1,
)
from .phase2b_trusted_wire_v1 import (
    ENVELOPE_HEADER_BYTES,
    FIELD_MANIFEST_ID,
    JCS_PROFILE_ID,
    NON_AUTHORITATIVE_CLAIM_LEVEL,
    decode_phase2b_jcs_profile_v1,
    encode_phase2b_jcs_profile_v1,
)


TYPED_TRUSTED_WIRE_REPLAY_VERSION: Final = (
    "hegel-machine-phase2b-trusted-wire-typed-replay/1"
)
_RAW_ROW_FIELD_MANIFEST: Final = (
    "envelope",
    "envelope_id",
    "payload_sha256",
    "authority",
    "authority_content_id",
    "transform_result_id",
    "replay_policy_id",
    "typed_authority_schema_id",
    "batch_policy_membership_verified",
    "secret_padding_replay_verified",
    "direct_payload_authority_transform_replay_verified",
    "lossless_typed_profile_roundtrip_verified",
    "origin_authenticated",
    "formal_covert_audit",
    "sealed_holdout_eligible",
    "c1_exit_evidence",
)
_BATCH_RECEIPT_FIELD_MANIFEST: Final = (
    "disposition",
    "batch",
    "batch_id",
    "batch_policy_id",
    "run_id_commitment",
    "rows",
    "source_authorities",
    "source_authority_content_ids",
    "secret_replay_receipt",
    "secret_replay_receipt_id",
    "authority_content_ids",
    "transform_result_ids",
    "replay_policy_id",
    "claim_level",
    "batch_policy_membership_verified",
    "whole_batch_atomic_typed_replay_verified",
    "secret_custodian_replay_verified",
    "direct_payload_authority_transform_replay_verified",
    "origin_authenticated",
    "formal_covert_audit",
    "sealed_holdout_eligible",
    "c1_exit_evidence",
)
_REJECTION_FIELD_MANIFEST: Final = (
    "disposition",
    "reason",
    "authority_count",
    "batch_id",
    "rows",
    "source_authority_content_ids",
    "authority_content_ids",
    "transform_result_ids",
    "replay_policy_id",
    "claim_level",
)
_CLAIM_MANIFEST: Final = {
    "batch_complete_true": (
        "batch_policy_membership_verified",
        "whole_batch_atomic_typed_replay_verified",
        "secret_custodian_replay_verified",
        "direct_payload_authority_transform_replay_verified",
    ),
    "raw_row_true": (
        "direct_payload_authority_transform_replay_verified",
        "lossless_typed_profile_roundtrip_verified",
    ),
    "always_false": (
        "origin_authenticated",
        "formal_covert_audit",
        "sealed_holdout_eligible",
        "c1_exit_evidence",
    ),
    "raw_row_false": (
        "batch_policy_membership_verified",
        "secret_padding_replay_verified",
    ),
}
_ROW_ID_FIELD_MANIFEST: Final = (
    "authority_content_id",
    "envelope_id",
    "payload_sha256",
    "policy_id",
    "transform_result_id",
)
_RECEIPT_ID_FIELD_MANIFEST: Final = (
    "authority_content_ids",
    "batch_id",
    "batch_policy_id",
    "replay_policy_id",
    "row_ids",
    "run_id_commitment",
    "secret_replay_receipt_id",
    "source_authority_content_ids",
    "transform_result_ids",
)
_REPLAY_POLICY_VALUE: Final = {
    "batch_payload_fields": list(TRUSTED_WIRE_BATCH_PAYLOAD_FIELDS),
    "batch_payload_schema_version": TRUSTED_WIRE_BATCH_PAYLOAD_SCHEMA_VERSION,
    "batch_policy_id": TRUSTED_WIRE_BATCH_POLICY_ID,
    "batch_receipt_field_manifest": list(_BATCH_RECEIPT_FIELD_MANIFEST),
    "claim_manifest": {
        key: list(value) for key, value in sorted(_CLAIM_MANIFEST.items())
    },
    "claim_value_type": "exact_bool",
    "direct_payload_authority_transform_replay_required": True,
    "exact_transform_provenance_compiler_version": (
        EXACT_TRANSFORM_PROVENANCE_COMPILER_VERSION
    ),
    "exact_transform_policy_id": EXACT_TRANSFORM_POLICY_ID,
    "exact_transform_provenance_compiler_policy_id": (
        EXACT_TRANSFORM_PROVENANCE_COMPILER_POLICY_ID
    ),
    "formal_claims_enabled": False,
    "jcs_profile_id": JCS_PROFILE_ID,
    "field_manifest_id": FIELD_MANIFEST_ID,
    "raw_row_field_manifest": list(_RAW_ROW_FIELD_MANIFEST),
    "receipt_id_formula": {
        "fields": list(_RECEIPT_ID_FIELD_MANIFEST),
        "prefix": "phase2b_typed_trusted_wire_batch_replay_",
        "validate_before_hash": True,
    },
    "rejection_field_manifest": list(_REJECTION_FIELD_MANIFEST),
    "rejection_empty_root_contract": "all_partial_fields_exact_empty_tuple",
    "receipt_scalar_type_contract": (
        "all_identity_roots_and_claim_strings_use_exact_builtin_types"
    ),
    "row_id_formula": {
        "fields": list(_ROW_ID_FIELD_MANIFEST),
        "prefix": "phase2b_typed_trusted_wire_row_",
        "validate_before_hash": True,
    },
    "semantic_failure_contract": (
        "abstain_without_partial_rows_or_source_authority_or_transform_roots;"
        "validated_input_batch_id_may_be_reported"
    ),
    "source_authority_binding": (
        "ordered_exact_source_authorities_and_content_ids_after_secret_replay"
    ),
    "stage_b_secret_replay_receipt_contract": (
        "exact_type_validate_batch_id_run_commitment_authority_count_"
        "ordered_source_roots_and_stored_receipt_id"
    ),
    "typed_authority_codec_version": TYPED_AUTHORITY_CODEC_VERSION,
    "public_provenance_version": TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION,
    "typed_authority_codec_policy_id": TYPED_AUTHORITY_CODEC_POLICY_ID,
    "typed_authority_schema_id": TYPED_AUTHORITY_SCHEMA_ID,
    "version": TYPED_TRUSTED_WIRE_REPLAY_VERSION,
    "whole_batch_atomic": True,
}
TYPED_TRUSTED_WIRE_REPLAY_POLICY_ID: Final = stable_hash(
    _REPLAY_POLICY_VALUE,
    prefix="phase2b_typed_trusted_wire_replay_policy_",
)


class TypedTrustedWireReplayDisposition(str, Enum):
    COMPLETE = "COMPLETE"
    ABSTAIN = "ABSTAIN"


def _require_prefixed_digest(value: object, prefix: str, name: str) -> str:
    if type(value) is not str or not value.startswith(prefix):
        raise TypeError(f"{name} must use the frozen content-ID prefix")
    suffix = value[len(prefix) :]
    if len(suffix) != 64 or any(item not in "0123456789abcdef" for item in suffix):
        raise ValueError(f"{name} must end in lowercase SHA-256")
    return value


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        item not in "0123456789abcdef" for item in value
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _require_exact_bool_claims(*values: object) -> None:
    if any(type(value) is not bool for value in values):
        raise TypeError("typed replay claim fields must use exact bool values")


def _typed_payload_parts(
    envelope: bytes,
) -> tuple[
    str,
    str,
    PublicTransformEvidenceBundleV2,
    ExactTransformCompilation,
]:
    structural = decode_and_audit_trusted_envelope_v1(envelope)
    start = ENVELOPE_HEADER_BYTES
    stop = start + structural.payload_bytes
    payload = envelope[start:stop]
    decoded = decode_phase2b_jcs_profile_v1(payload)
    if type(decoded) is not dict or tuple(sorted(decoded)) != (
        TRUSTED_WIRE_BATCH_PAYLOAD_FIELDS
    ):
        raise ValueError("typed replay payload field manifest drift")
    if (
        decoded["schema_version"] != TRUSTED_WIRE_BATCH_PAYLOAD_SCHEMA_VERSION
        or decoded["public_provenance_version"]
        != TRUSTED_WIRE_PUBLIC_PROVENANCE_VERSION
        or decoded["typed_authority_schema_id"] != TYPED_AUTHORITY_SCHEMA_ID
        or decoded["jcs_profile_id"] != JCS_PROFILE_ID
        or decoded["field_manifest_id"] != FIELD_MANIFEST_ID
    ):
        raise ValueError("typed replay payload identity drift")
    authority = decode_typed_transform_authority_profile_v1(decoded["authority"])
    rebuilt = dict(decoded)
    rebuilt["authority"] = encode_typed_transform_authority_profile_v1(authority)
    if encode_phase2b_jcs_profile_v1(rebuilt) != payload:
        raise ValueError("typed replay authority is not losslessly encoded")
    result = run_exact_transform_semantics(authority)
    if (
        type(result) is not ExactTransformCompilation
        or result.disposition is not TransformCompilationDisposition.COMPLETE
    ):
        raise ValueError("typed replay authority is not exact-transform complete")
    if result.wrapper_content_id != authority.content_id:
        raise ValueError("typed replay authority and transform root disagree")
    return structural.envelope_id, structural.payload_sha256, authority, result


@dataclass(frozen=True, slots=True)
class TypedTrustedEnvelopeReplayV1:
    envelope: bytes
    envelope_id: str
    payload_sha256: str
    authority: PublicTransformEvidenceBundleV2
    authority_content_id: str
    transform_result_id: str
    replay_policy_id: str = TYPED_TRUSTED_WIRE_REPLAY_POLICY_ID
    typed_authority_schema_id: str = TYPED_AUTHORITY_SCHEMA_ID
    batch_policy_membership_verified: bool = False
    secret_padding_replay_verified: bool = False
    direct_payload_authority_transform_replay_verified: bool = True
    lossless_typed_profile_roundtrip_verified: bool = True
    origin_authenticated: bool = False
    formal_covert_audit: bool = False
    sealed_holdout_eligible: bool = False
    c1_exit_evidence: bool = False

    def __post_init__(self) -> None:
        if type(self) is not TypedTrustedEnvelopeReplayV1:
            raise TypeError("typed envelope replay must use the exact type")
        if type(self.envelope) is not bytes:
            raise TypeError("typed envelope replay needs exact bytes")
        _require_prefixed_digest(
            self.envelope_id,
            "phase2b_trusted_envelope_",
            "typed replay envelope ID",
        )
        _require_sha256(self.payload_sha256, "typed replay payload SHA-256")
        _require_prefixed_digest(
            self.authority_content_id,
            "phase2b_public_transform_evidence_",
            "typed replay authority content ID",
        )
        _require_prefixed_digest(
            self.transform_result_id,
            "phase2b_exact_transform_result_",
            "typed replay transform result ID",
        )
        _require_prefixed_digest(
            self.replay_policy_id,
            "phase2b_typed_trusted_wire_replay_policy_",
            "typed replay policy ID",
        )
        _require_prefixed_digest(
            self.typed_authority_schema_id,
            "phase2b_trusted_wire_typed_authority_schema_",
            "typed replay authority schema ID",
        )
        encoded_authority = encode_typed_transform_authority_profile_v1(
            self.authority
        )
        envelope_id, payload_sha256, authority, result = _typed_payload_parts(
            self.envelope
        )
        if (
            self.envelope_id != envelope_id
            or self.payload_sha256 != payload_sha256
            or type(self.authority) is not PublicTransformEvidenceBundleV2
            or self.authority != authority
            or encoded_authority
            != encode_typed_transform_authority_profile_v1(authority)
            or self.authority_content_id != authority.content_id
            or self.transform_result_id != result.result_id
            or self.replay_policy_id != TYPED_TRUSTED_WIRE_REPLAY_POLICY_ID
            or self.typed_authority_schema_id != TYPED_AUTHORITY_SCHEMA_ID
        ):
            raise ValueError("typed envelope replay receipt drift")
        _require_exact_bool_claims(
            self.batch_policy_membership_verified,
            self.secret_padding_replay_verified,
            self.direct_payload_authority_transform_replay_verified,
            self.lossless_typed_profile_roundtrip_verified,
            self.origin_authenticated,
            self.formal_covert_audit,
            self.sealed_holdout_eligible,
            self.c1_exit_evidence,
        )
        if not all(
            (
                self.direct_payload_authority_transform_replay_verified,
                self.lossless_typed_profile_roundtrip_verified,
            )
        ) or any(
            (
                self.batch_policy_membership_verified,
                self.secret_padding_replay_verified,
                self.origin_authenticated,
                self.formal_covert_audit,
                self.sealed_holdout_eligible,
                self.c1_exit_evidence,
            )
        ):
            raise ValueError("typed envelope replay claim boundary drift")

    @property
    def transform_result(self) -> ExactTransformCompilation:
        self.__post_init__()
        result = run_exact_transform_semantics(self.authority)
        if (
            type(result) is not ExactTransformCompilation
            or result.disposition is not TransformCompilationDisposition.COMPLETE
            or result.result_id != self.transform_result_id
        ):
            raise ValueError("typed envelope replay transform result drift")
        return result

    @property
    def row_id(self) -> str:
        self.__post_init__()
        return stable_hash(
            {
                "authority_content_id": self.authority_content_id,
                "envelope_id": self.envelope_id,
                "payload_sha256": self.payload_sha256,
                "policy_id": self.replay_policy_id,
                "transform_result_id": self.transform_result_id,
            },
            prefix="phase2b_typed_trusted_wire_row_",
        )


def decode_and_replay_typed_trusted_envelope_v1(
    envelope: bytes,
) -> TypedTrustedEnvelopeReplayV1:
    """Strictly decode and directly replay one envelope without batch claims."""

    if type(envelope) is not bytes:
        raise TypeError("typed trusted envelope input must use exact bytes")
    envelope_id, payload_sha256, authority, result = _typed_payload_parts(envelope)
    return TypedTrustedEnvelopeReplayV1(
        envelope=envelope,
        envelope_id=envelope_id,
        payload_sha256=payload_sha256,
        authority=authority,
        authority_content_id=authority.content_id,
        transform_result_id=result.result_id,
    )


@dataclass(frozen=True, slots=True)
class TypedTrustedWireBatchReplayRejectionV1:
    disposition: TypedTrustedWireReplayDisposition
    reason: str
    authority_count: int
    batch_id: str | None
    rows: tuple[()] = ()
    source_authority_content_ids: tuple[()] = ()
    authority_content_ids: tuple[()] = ()
    transform_result_ids: tuple[()] = ()
    replay_policy_id: str = TYPED_TRUSTED_WIRE_REPLAY_POLICY_ID
    claim_level: str = NON_AUTHORITATIVE_CLAIM_LEVEL

    def __post_init__(self) -> None:
        if type(self) is not TypedTrustedWireBatchReplayRejectionV1:
            raise TypeError("typed replay rejection must use the exact type")
        if self.disposition is not TypedTrustedWireReplayDisposition.ABSTAIN:
            raise ValueError("typed replay rejection must abstain")
        if type(self.reason) is not str or not self.reason or not self.reason.isascii():
            raise ValueError("typed replay rejection reason must be nonempty ASCII")
        if type(self.authority_count) is not int or self.authority_count < 0:
            raise ValueError("typed replay rejection count is invalid")
        if self.batch_id is not None:
            _require_prefixed_digest(
                self.batch_id,
                "phase2b_trusted_wire_batch_",
                "typed replay rejection batch ID",
            )
        _require_prefixed_digest(
            self.replay_policy_id,
            "phase2b_typed_trusted_wire_replay_policy_",
            "typed replay rejection policy ID",
        )
        partial_values = (
            self.rows,
            self.source_authority_content_ids,
            self.authority_content_ids,
            self.transform_result_ids,
        )
        if any(type(value) is not tuple or value != () for value in partial_values):
            raise ValueError("typed replay rejection cannot expose partial roots")
        if (
            self.replay_policy_id != TYPED_TRUSTED_WIRE_REPLAY_POLICY_ID
            or type(self.claim_level) is not str
            or self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
        ):
            raise ValueError("typed replay rejection identity drift")


_BATCH_REPLAY_ISSUE_TOKEN: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class TypedTrustedWireBatchReplayV1:
    disposition: TypedTrustedWireReplayDisposition
    batch: TrustedWireBatchV1
    batch_id: str
    batch_policy_id: str
    run_id_commitment: str
    rows: tuple[TypedTrustedEnvelopeReplayV1, ...]
    source_authorities: tuple[PublicTransformEvidenceBundleV2, ...]
    source_authority_content_ids: tuple[str, ...]
    secret_replay_receipt: TrustedWireReplayReceiptV1
    secret_replay_receipt_id: str
    authority_content_ids: tuple[str, ...]
    transform_result_ids: tuple[str, ...]
    replay_policy_id: str
    claim_level: str
    batch_policy_membership_verified: bool
    whole_batch_atomic_typed_replay_verified: bool
    secret_custodian_replay_verified: bool
    direct_payload_authority_transform_replay_verified: bool
    origin_authenticated: bool
    formal_covert_audit: bool
    sealed_holdout_eligible: bool
    c1_exit_evidence: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("typed batch replay receipts are issued only by exact replay")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        batch: TrustedWireBatchV1,
        rows: tuple[TypedTrustedEnvelopeReplayV1, ...],
        source_authorities: tuple[PublicTransformEvidenceBundleV2, ...],
        secret_replay_receipt: TrustedWireReplayReceiptV1,
    ) -> "TypedTrustedWireBatchReplayV1":
        if token is not _BATCH_REPLAY_ISSUE_TOKEN:
            raise TypeError("typed batch replay issuer token mismatch")
        value = object.__new__(cls)
        frozen: tuple[tuple[str, object], ...] = (
            ("disposition", TypedTrustedWireReplayDisposition.COMPLETE),
            ("batch", batch),
            ("batch_id", batch.batch_id),
            ("batch_policy_id", TRUSTED_WIRE_BATCH_POLICY_ID),
            ("run_id_commitment", batch.run_id_commitment),
            ("rows", rows),
            ("source_authorities", source_authorities),
            (
                "source_authority_content_ids",
                tuple(item.content_id for item in source_authorities),
            ),
            ("secret_replay_receipt", secret_replay_receipt),
            (
                "secret_replay_receipt_id",
                secret_replay_receipt.replay_receipt_id,
            ),
            (
                "authority_content_ids",
                tuple(item.authority_content_id for item in rows),
            ),
            (
                "transform_result_ids",
                tuple(item.transform_result_id for item in rows),
            ),
            ("replay_policy_id", TYPED_TRUSTED_WIRE_REPLAY_POLICY_ID),
            ("claim_level", NON_AUTHORITATIVE_CLAIM_LEVEL),
            ("batch_policy_membership_verified", True),
            ("whole_batch_atomic_typed_replay_verified", True),
            ("secret_custodian_replay_verified", True),
            ("direct_payload_authority_transform_replay_verified", True),
            ("origin_authenticated", False),
            ("formal_covert_audit", False),
            ("sealed_holdout_eligible", False),
            ("c1_exit_evidence", False),
        )
        for name, item in frozen:
            object.__setattr__(value, name, item)
        value._validate()
        return value

    def _validate(self) -> None:
        if type(self) is not TypedTrustedWireBatchReplayV1:
            raise TypeError("typed batch replay receipt must use the exact type")
        if type(self.batch) is not TrustedWireBatchV1:
            raise TypeError("typed batch replay needs the exact batch")
        self.batch._validate()
        _require_prefixed_digest(
            self.batch_id,
            "phase2b_trusted_wire_batch_",
            "typed batch replay batch ID",
        )
        _require_prefixed_digest(
            self.batch_policy_id,
            "phase2b_trusted_wire_batch_policy_",
            "typed batch replay batch policy ID",
        )
        _require_prefixed_digest(
            self.run_id_commitment,
            "phase2b_trusted_wire_run_",
            "typed batch replay run commitment",
        )
        _require_prefixed_digest(
            self.replay_policy_id,
            "phase2b_typed_trusted_wire_replay_policy_",
            "typed batch replay policy ID",
        )
        if (
            self.disposition is not TypedTrustedWireReplayDisposition.COMPLETE
            or self.batch_id != self.batch.batch_id
            or self.batch_policy_id != TRUSTED_WIRE_BATCH_POLICY_ID
            or self.run_id_commitment != self.batch.run_id_commitment
            or self.replay_policy_id != TYPED_TRUSTED_WIRE_REPLAY_POLICY_ID
            or type(self.claim_level) is not str
            or self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
        ):
            raise ValueError("typed batch replay identity drift")
        if (
            type(self.rows) is not tuple
            or not self.rows
            or len(self.rows) != len(self.batch.envelopes)
            or any(type(item) is not TypedTrustedEnvelopeReplayV1 for item in self.rows)
        ):
            raise TypeError("typed batch replay needs all exact rows")
        for item in self.rows:
            item.__post_init__()
        if (
            type(self.source_authorities) is not tuple
            or len(self.source_authorities) != len(self.rows)
            or any(
                type(item) is not PublicTransformEvidenceBundleV2
                for item in self.source_authorities
            )
        ):
            raise TypeError("typed batch replay needs all exact source authorities")
        for item in self.source_authorities:
            encode_typed_transform_authority_profile_v1(item)
        if (
            type(self.source_authority_content_ids) is not tuple
            or len(self.source_authority_content_ids)
            != len(self.source_authorities)
        ):
            raise TypeError("typed batch replay source roots must be a full tuple")
        for item in self.source_authority_content_ids:
            _require_prefixed_digest(
                item,
                "phase2b_public_transform_evidence_",
                "typed batch replay source authority content ID",
            )
        if self.source_authority_content_ids != tuple(
            item.content_id for item in self.source_authorities
        ):
            raise ValueError("typed batch replay source authority roots drift")
        if type(self.secret_replay_receipt) is not TrustedWireReplayReceiptV1:
            raise TypeError("typed batch replay needs exact secret replay receipt")
        self.secret_replay_receipt._validate()
        _require_prefixed_digest(
            self.secret_replay_receipt_id,
            "phase2b_trusted_wire_secret_replay_",
            "typed batch replay secret replay receipt ID",
        )
        if (
            self.secret_replay_receipt_id
            != self.secret_replay_receipt.replay_receipt_id
            or self.secret_replay_receipt.batch_id != self.batch_id
            or self.secret_replay_receipt.run_id_commitment
            != self.run_id_commitment
            or self.secret_replay_receipt.authority_count != len(self.rows)
            or self.secret_replay_receipt.source_authority_content_ids
            != self.source_authority_content_ids
        ):
            raise ValueError("typed batch replay secret receipt binding drift")
        if tuple(item.envelope_id for item in self.rows) != tuple(
            item.envelope_id for item in self.batch.envelopes
        ):
            raise ValueError("typed batch replay row membership drift")
        if (
            type(self.authority_content_ids) is not tuple
            or len(self.authority_content_ids) != len(self.rows)
            or type(self.transform_result_ids) is not tuple
            or len(self.transform_result_ids) != len(self.rows)
        ):
            raise TypeError("typed batch replay row roots must be full tuples")
        for item in self.authority_content_ids:
            _require_prefixed_digest(
                item,
                "phase2b_public_transform_evidence_",
                "typed batch replay authority content ID",
            )
        for item in self.transform_result_ids:
            _require_prefixed_digest(
                item,
                "phase2b_exact_transform_result_",
                "typed batch replay transform result ID",
            )
        if self.authority_content_ids != tuple(
            item.authority_content_id for item in self.rows
        ) or self.transform_result_ids != tuple(
            item.transform_result_id for item in self.rows
        ):
            raise ValueError("typed batch replay row roots drift")
        _require_exact_bool_claims(
            self.batch_policy_membership_verified,
            self.whole_batch_atomic_typed_replay_verified,
            self.secret_custodian_replay_verified,
            self.direct_payload_authority_transform_replay_verified,
            self.origin_authenticated,
            self.formal_covert_audit,
            self.sealed_holdout_eligible,
            self.c1_exit_evidence,
        )
        if not all(
            (
                self.batch_policy_membership_verified,
                self.whole_batch_atomic_typed_replay_verified,
                self.secret_custodian_replay_verified,
                self.direct_payload_authority_transform_replay_verified,
            )
        ) or any(
            (
                self.origin_authenticated,
                self.formal_covert_audit,
                self.sealed_holdout_eligible,
                self.c1_exit_evidence,
            )
        ):
            raise ValueError("typed batch replay claim boundary drift")

    @property
    def receipt_id(self) -> str:
        self._validate()
        return stable_hash(
            {
                "authority_content_ids": self.authority_content_ids,
                "batch_id": self.batch_id,
                "batch_policy_id": self.batch_policy_id,
                "replay_policy_id": self.replay_policy_id,
                "row_ids": tuple(item.row_id for item in self.rows),
                "run_id_commitment": self.run_id_commitment,
                "secret_replay_receipt_id": self.secret_replay_receipt_id,
                "source_authority_content_ids": (
                    self.source_authority_content_ids
                ),
                "transform_result_ids": self.transform_result_ids,
            },
            prefix="phase2b_typed_trusted_wire_batch_replay_",
        )


def _assert_receipt_field_manifests() -> None:
    actual = (
        tuple(item.name for item in fields(TypedTrustedEnvelopeReplayV1)),
        tuple(
            item.name
            for item in fields(TypedTrustedWireBatchReplayRejectionV1)
        ),
        tuple(item.name for item in fields(TypedTrustedWireBatchReplayV1)),
    )
    expected = (
        _RAW_ROW_FIELD_MANIFEST,
        _REJECTION_FIELD_MANIFEST,
        _BATCH_RECEIPT_FIELD_MANIFEST,
    )
    if actual != expected:
        raise RuntimeError("typed replay receipt field manifest drift")


_assert_receipt_field_manifests()


def replay_typed_trusted_wire_batch_v1(
    *,
    batch: TrustedWireBatchV1,
    run_id: bytes,
    key_sources: TrustedWireKeySourcesV1,
    authorities: tuple[PublicTransformEvidenceBundleV2, ...],
) -> TypedTrustedWireBatchReplayV1 | TypedTrustedWireBatchReplayRejectionV1:
    """Replay supplied custodian inputs and atomically decode the whole batch."""

    if type(batch) is not TrustedWireBatchV1:
        raise TypeError("typed replay requires the exact trusted-wire batch")
    if type(run_id) is not bytes:
        raise TypeError("typed replay run ID must use exact bytes")
    if type(key_sources) is not TrustedWireKeySourcesV1:
        raise TypeError("typed replay key sources must use the exact type")
    if type(authorities) is not tuple:
        raise TypeError("typed replay authorities must use an exact immutable tuple")
    if not 1 <= len(authorities) <= MAXIMUM_BATCH_AUTHORITIES:
        return TypedTrustedWireBatchReplayRejectionV1(
            TypedTrustedWireReplayDisposition.ABSTAIN,
            "authority_count_out_of_range",
            len(authorities),
            None,
        )
    if any(type(item) is not PublicTransformEvidenceBundleV2 for item in authorities):
        raise TypeError("typed replay authorities must use an exact immutable tuple")
    try:
        batch._validate()
    except (AttributeError, KeyError, TypeError, ValueError):
        return TypedTrustedWireBatchReplayRejectionV1(
            TypedTrustedWireReplayDisposition.ABSTAIN,
            "batch_validation_failed",
            len(authorities),
            None,
        )
    batch_id = batch.batch_id
    try:
        secret_replay = verify_trusted_wire_batch_replay_v1(
            batch=batch,
            run_id=run_id,
            key_sources=key_sources,
            authorities=authorities,
        )
        if type(secret_replay) is not TrustedWireReplayReceiptV1:
            raise TypeError("trusted-wire secret replay receipt type drift")
        secret_replay._validate()
        if (
            secret_replay.batch_id != batch_id
            or secret_replay.run_id_commitment != batch.run_id_commitment
            or secret_replay.authority_count != len(authorities)
            or secret_replay.source_authority_content_ids
            != tuple(authority.content_id for authority in authorities)
        ):
            raise ValueError("trusted-wire secret replay receipt identity drift")
        rows = tuple(
            decode_and_replay_typed_trusted_envelope_v1(item.envelope)
            for item in batch.envelopes
        )
        receipt = TypedTrustedWireBatchReplayV1._issue(
            _BATCH_REPLAY_ISSUE_TOKEN,
            batch=batch,
            rows=rows,
            source_authorities=authorities,
            secret_replay_receipt=secret_replay,
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return TypedTrustedWireBatchReplayRejectionV1(
            TypedTrustedWireReplayDisposition.ABSTAIN,
            "custodian_or_typed_replay_failed",
            len(authorities),
            batch_id,
        )
    return receipt


__all__ = (
    "TYPED_TRUSTED_WIRE_REPLAY_POLICY_ID",
    "TYPED_TRUSTED_WIRE_REPLAY_VERSION",
    "TypedTrustedEnvelopeReplayV1",
    "TypedTrustedWireBatchReplayRejectionV1",
    "TypedTrustedWireBatchReplayV1",
    "TypedTrustedWireReplayDisposition",
    "decode_and_replay_typed_trusted_envelope_v1",
    "replay_typed_trusted_wire_batch_v1",
)
