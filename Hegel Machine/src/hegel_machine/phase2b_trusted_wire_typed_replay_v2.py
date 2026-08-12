"""Public typed replay for compact Phase-2B trusted-wire V2 batches.

The public entry points consume only fixed V2 envelope bytes or an already
issued V2 batch.  They replay the public frame, compact canonical authority,
public provenance, and direct exact-transform semantics.  They do not rebuild
the batch and make no claim about custodian inputs, shuffle or allocation
mechanics, secret HMAC material, origin, formal audit, recognizer capacity, or
C1 exit evidence.

The private core is an integration seam for a future recognizer-input archive.
It performs exactly one custodian build, optionally checks that build against
an expected exact batch, and then reduces the result to the same safe public
receipt exposed by the public API.  Private projections are returned only when
the entire build and public replay complete atomically.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Callable, Final, Mapping

from .hashing import stable_hash
from .phase2b_exact_transform_semantics_v1 import PublicTransformEvidenceBundleV2
from . import phase2b_trusted_wire_batch_v2 as _batch_v2
from .phase2b_trusted_wire_batch_v2 import (
    MAXIMUM_BATCH_V2_AUTHORITIES,
    TRUSTED_WIRE_BATCH_V2_POLICY_ID,
    DecodedTrustedEnvelopeV2,
    TrustedWireBatchDispositionV2,
    TrustedWireBatchRejectionV2,
    TrustedWireBatchV2,
    TrustedWireKeySourcesV2,
    decode_and_audit_trusted_envelope_v2,
)
from .phase2b_trusted_wire_v1 import NON_AUTHORITATIVE_CLAIM_LEVEL


TYPED_TRUSTED_WIRE_REPLAY_V2_VERSION: Final = (
    "hegel-machine-phase2b-trusted-wire-typed-replay/2"
)

_REJECTION_FIELD_MANIFEST_V2: Final = (
    "disposition",
    "reason",
    "authority_count",
    "batch_id",
    "rows",
    "row_ids",
    "authority_content_ids",
    "transform_result_ids",
    "replay_policy_id",
    "claim_level",
)
_RECEIPT_FIELD_MANIFEST_V2: Final = (
    "disposition",
    "batch",
    "batch_id",
    "batch_policy_id",
    "rows",
    "row_ids",
    "authority_content_ids",
    "transform_result_ids",
    "replay_policy_id",
    "claim_level",
    "batch_policy_membership_verified",
    "whole_batch_atomic_typed_replay_verified",
    "compact_authority_canonical_replay_verified",
    "public_provenance_verified",
    "direct_exact_transform_replay_verified",
    "secret_custodian_replay_verified",
    "whole_batch_shuffle_publicly_verified",
    "purpose_separated_keys_publicly_verified",
    "post_shuffle_hmac_uuidv4_publicly_verified",
    "secret_hmac_padding_publicly_verified",
    "source_authority_binding_verified",
    "live_allocation_schedule_verified",
    "recognizer_capacity_evidence",
    "origin_authenticated",
    "formal_uuid_audit",
    "formal_covert_audit",
    "sealed_holdout_eligible",
    "c1_exit_evidence",
)
_TRUE_CLAIMS_V2: Final = (
    "batch_policy_membership_verified",
    "whole_batch_atomic_typed_replay_verified",
    "compact_authority_canonical_replay_verified",
    "public_provenance_verified",
    "direct_exact_transform_replay_verified",
)
_FALSE_CLAIMS_V2: Final = (
    "secret_custodian_replay_verified",
    "whole_batch_shuffle_publicly_verified",
    "purpose_separated_keys_publicly_verified",
    "post_shuffle_hmac_uuidv4_publicly_verified",
    "secret_hmac_padding_publicly_verified",
    "source_authority_binding_verified",
    "live_allocation_schedule_verified",
    "recognizer_capacity_evidence",
    "origin_authenticated",
    "formal_uuid_audit",
    "formal_covert_audit",
    "sealed_holdout_eligible",
    "c1_exit_evidence",
)
_ROW_ID_FIELD_MANIFEST_V2: Final = (
    "authority_content_id",
    "batch_id",
    "envelope_id",
    "namespace_audit_id",
    "padding_sha256",
    "payload_sha256",
    "replay_policy_id",
    "transform_result_id",
    "typed_authority_codec_policy_id",
    "typed_authority_codec_version",
    "typed_authority_schema_id",
)
_RECEIPT_ID_FIELD_MANIFEST_V2: Final = (
    "authority_content_ids",
    "batch_id",
    "batch_policy_id",
    "replay_policy_id",
    "row_ids",
    "transform_result_ids",
)
_REPLAY_POLICY_VALUE_V2: Final = {
    "batch_policy_id": TRUSTED_WIRE_BATCH_V2_POLICY_ID,
    "compact_typed_authority_codec_policy_id": (
        _batch_v2.COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID
    ),
    "compact_typed_authority_codec_version": (
        _batch_v2.COMPACT_TYPED_AUTHORITY_CODEC_VERSION
    ),
    "compact_typed_authority_schema_id": (
        _batch_v2.COMPACT_TYPED_AUTHORITY_SCHEMA_ID
    ),
    "claim_manifest": {
        "always_false": list(_FALSE_CLAIMS_V2),
        "public_complete_true": list(_TRUE_CLAIMS_V2),
    },
    "claim_value_type": "exact_builtin_bool",
    "custodian_material_in_public_api": False,
    "decode_api_result_type": "DecodedTrustedEnvelopeV2",
    "formal_claims_enabled": False,
    "private_core_contract": {
        "atomic_projection_return": True,
        "builder_call_count_on_build_path": 1,
        "expected_batch_exact_validation_and_equality": True,
        "public_replay_after_build": True,
    },
    "public_replay_contract": (
        "exact_TrustedWireBatchV2_zero_argument_validation_then_all_rows_"
        "decoded_from_fixed_envelope_bytes"
    ),
    "receipt_field_manifest": list(_RECEIPT_FIELD_MANIFEST_V2),
    "receipt_id_formula": {
        "fields": list(_RECEIPT_ID_FIELD_MANIFEST_V2),
        "prefix": "phase2b_typed_trusted_wire_batch_replay_v2_",
        "validate_before_hash": True,
    },
    "rejection_empty_root_contract": (
        "rows_row_ids_authority_content_ids_transform_result_ids_exact_empty_tuple"
    ),
    "rejection_field_manifest": list(_REJECTION_FIELD_MANIFEST_V2),
    "row_id_formula": {
        "fields": list(_ROW_ID_FIELD_MANIFEST_V2),
        "prefix": "phase2b_typed_trusted_wire_row_v2_",
        "validate_before_hash": True,
    },
    "version": TYPED_TRUSTED_WIRE_REPLAY_V2_VERSION,
}
TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID: Final = stable_hash(
    _REPLAY_POLICY_VALUE_V2,
    prefix="phase2b_typed_trusted_wire_replay_policy_v2_",
)


class TypedTrustedWireReplayDispositionV2(str, Enum):
    COMPLETE = "COMPLETE"
    ABSTAIN = "ABSTAIN"


def _require_digest_v2(value: object, prefix: str, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must use an exact string")
    expected_length = len(prefix) + 64
    if len(value) != expected_length:
        raise ValueError(f"{name} has the wrong content-ID length")
    if not value.startswith(prefix):
        raise ValueError(f"{name} has the wrong content-ID prefix")
    suffix = value[len(prefix) :]
    if any(character not in "0123456789abcdef" for character in suffix):
        raise ValueError(f"{name} must end in lowercase SHA-256")
    return value


def _require_hex64_v2(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be an exact lowercase SHA-256 string")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _require_exact_bool_claims_v2(*values: object) -> None:
    if any(type(value) is not bool for value in values):
        raise TypeError("V2 typed replay claims must use exact bool values")


def decode_and_replay_typed_trusted_envelope_v2(
    envelope: bytes,
) -> DecodedTrustedEnvelopeV2:
    """Replay one public V2 envelope without claiming batch membership."""

    if type(envelope) is not bytes:
        raise TypeError("V2 typed replay envelope must use exact bytes")
    decoded = decode_and_audit_trusted_envelope_v2(envelope)
    if type(decoded) is not DecodedTrustedEnvelopeV2:
        raise TypeError("V2 typed replay decoder result type drift")
    decoded._validate()
    return decoded


def _validated_batch_id_v2(batch: TrustedWireBatchV2) -> str:
    if type(batch) is not TrustedWireBatchV2:
        raise TypeError("V2 typed replay requires the exact trusted-wire batch")
    batch._validate()
    return _require_digest_v2(
        batch.batch_id,
        "phase2b_trusted_wire_batch_v2_",
        "V2 typed replay batch ID",
    )


def _row_id_v2(
    row: DecodedTrustedEnvelopeV2,
    batch_id: str,
    *,
    validated: bool = False,
) -> str:
    if type(row) is not DecodedTrustedEnvelopeV2:
        raise TypeError("V2 typed replay row must use the exact decoded type")
    if type(validated) is not bool:
        raise TypeError("V2 typed replay row validation flag must be exact bool")
    if not validated:
        row._validate()
    _require_digest_v2(
        batch_id,
        "phase2b_trusted_wire_batch_v2_",
        "V2 typed replay row batch ID",
    )
    _require_digest_v2(
        row.envelope_id,
        "phase2b_trusted_envelope_v2_",
        "V2 typed replay row envelope ID",
    )
    _require_hex64_v2(row.payload_sha256, "V2 typed replay row payload SHA-256")
    _require_hex64_v2(row.padding_sha256, "V2 typed replay row padding SHA-256")
    _require_digest_v2(
        row.namespace_audit_id,
        "phase2b_namespace_audit_v2_",
        "V2 typed replay row namespace audit ID",
    )
    _require_digest_v2(
        row.authority_content_id,
        "phase2b_public_transform_evidence_",
        "V2 typed replay row authority content ID",
    )
    _require_digest_v2(
        row.transform_result_id,
        "phase2b_exact_transform_result_",
        "V2 typed replay row transform result ID",
    )
    for value, expected, name in (
        (
            row.typed_authority_schema_id,
            _batch_v2.COMPACT_TYPED_AUTHORITY_SCHEMA_ID,
            "V2 typed replay row compact schema ID",
        ),
        (
            row.typed_authority_codec_version,
            _batch_v2.COMPACT_TYPED_AUTHORITY_CODEC_VERSION,
            "V2 typed replay row compact codec version",
        ),
        (
            row.typed_authority_codec_policy_id,
            _batch_v2.COMPACT_TYPED_AUTHORITY_CODEC_POLICY_ID,
            "V2 typed replay row compact codec policy ID",
        ),
    ):
        if type(value) is not str or value != expected:
            raise ValueError(f"{name} drift")
    return stable_hash(
        {
            "authority_content_id": row.authority_content_id,
            "batch_id": batch_id,
            "envelope_id": row.envelope_id,
            "namespace_audit_id": row.namespace_audit_id,
            "padding_sha256": row.padding_sha256,
            "payload_sha256": row.payload_sha256,
            "replay_policy_id": TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID,
            "transform_result_id": row.transform_result_id,
            "typed_authority_codec_policy_id": (
                row.typed_authority_codec_policy_id
            ),
            "typed_authority_codec_version": row.typed_authority_codec_version,
            "typed_authority_schema_id": row.typed_authority_schema_id,
        },
        prefix="phase2b_typed_trusted_wire_row_v2_",
    )


@dataclass(frozen=True, slots=True)
class TypedTrustedWireBatchReplayRejectionV2:
    disposition: TypedTrustedWireReplayDispositionV2
    reason: str
    authority_count: int
    batch_id: str | None
    rows: tuple[()] = ()
    row_ids: tuple[()] = ()
    authority_content_ids: tuple[()] = ()
    transform_result_ids: tuple[()] = ()
    replay_policy_id: str = TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID
    claim_level: str = NON_AUTHORITATIVE_CLAIM_LEVEL

    def __post_init__(self) -> None:
        if type(self) is not TypedTrustedWireBatchReplayRejectionV2:
            raise TypeError("V2 typed replay rejection must use the exact type")
        if self.disposition is not TypedTrustedWireReplayDispositionV2.ABSTAIN:
            raise ValueError("V2 typed replay rejection must abstain")
        if type(self.reason) is not str or not 1 <= len(self.reason) <= 2048:
            raise ValueError("V2 typed replay rejection reason length is invalid")
        if not self.reason.isascii():
            raise ValueError("V2 typed replay rejection reason must be ASCII")
        if type(self.authority_count) is not int or self.authority_count < 0:
            raise ValueError("V2 typed replay rejection count is invalid")
        if self.batch_id is not None:
            _require_digest_v2(
                self.batch_id,
                "phase2b_trusted_wire_batch_v2_",
                "V2 typed replay rejection batch ID",
            )
        for value in (
            self.rows,
            self.row_ids,
            self.authority_content_ids,
            self.transform_result_ids,
        ):
            if type(value) is not tuple or value != ():
                raise ValueError("V2 typed replay rejection exposes partial roots")
        if (
            type(self.replay_policy_id) is not str
            or self.replay_policy_id != TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID
            or type(self.claim_level) is not str
            or self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
        ):
            raise ValueError("V2 typed replay rejection identity drift")


_RECEIPT_ISSUE_TOKEN_V2: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class TypedTrustedWireBatchReplayV2:
    disposition: TypedTrustedWireReplayDispositionV2
    batch: TrustedWireBatchV2
    batch_id: str
    batch_policy_id: str
    rows: tuple[DecodedTrustedEnvelopeV2, ...]
    row_ids: tuple[str, ...]
    authority_content_ids: tuple[str, ...]
    transform_result_ids: tuple[str, ...]
    replay_policy_id: str
    claim_level: str
    batch_policy_membership_verified: bool
    whole_batch_atomic_typed_replay_verified: bool
    compact_authority_canonical_replay_verified: bool
    public_provenance_verified: bool
    direct_exact_transform_replay_verified: bool
    secret_custodian_replay_verified: bool
    whole_batch_shuffle_publicly_verified: bool
    purpose_separated_keys_publicly_verified: bool
    post_shuffle_hmac_uuidv4_publicly_verified: bool
    secret_hmac_padding_publicly_verified: bool
    source_authority_binding_verified: bool
    live_allocation_schedule_verified: bool
    recognizer_capacity_evidence: bool
    origin_authenticated: bool
    formal_uuid_audit: bool
    formal_covert_audit: bool
    sealed_holdout_eligible: bool
    c1_exit_evidence: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("V2 typed replay receipts are issued only by public replay")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        batch: TrustedWireBatchV2,
        batch_id: str,
        rows: tuple[DecodedTrustedEnvelopeV2, ...],
    ) -> "TypedTrustedWireBatchReplayV2":
        if token is not _RECEIPT_ISSUE_TOKEN_V2:
            raise TypeError("V2 typed replay receipt issuer token mismatch")
        if type(batch) is not TrustedWireBatchV2:
            raise TypeError("V2 typed replay issuer needs the exact batch")
        _require_digest_v2(
            batch_id,
            "phase2b_trusted_wire_batch_v2_",
            "V2 typed replay issuer batch ID",
        )
        if (
            type(rows) is not tuple
            or not 1 <= len(rows) <= MAXIMUM_BATCH_V2_AUTHORITIES
            or any(type(row) is not DecodedTrustedEnvelopeV2 for row in rows)
        ):
            raise TypeError("V2 typed replay issuer needs all exact rows")
        value = object.__new__(cls)
        frozen: tuple[tuple[str, object], ...] = (
            ("disposition", TypedTrustedWireReplayDispositionV2.COMPLETE),
            ("batch", batch),
            ("batch_id", batch_id),
            ("batch_policy_id", TRUSTED_WIRE_BATCH_V2_POLICY_ID),
            ("rows", rows),
            (
                "row_ids",
                tuple(_row_id_v2(row, batch_id, validated=True) for row in rows),
            ),
            (
                "authority_content_ids",
                tuple(row.authority_content_id for row in rows),
            ),
            (
                "transform_result_ids",
                tuple(row.transform_result_id for row in rows),
            ),
            ("replay_policy_id", TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID),
            ("claim_level", NON_AUTHORITATIVE_CLAIM_LEVEL),
        )
        for name, item in frozen:
            object.__setattr__(value, name, item)
        for name in _TRUE_CLAIMS_V2:
            object.__setattr__(value, name, True)
        for name in _FALSE_CLAIMS_V2:
            object.__setattr__(value, name, False)
        value._validate(validated_rows=rows, context_token=_RECEIPT_ISSUE_TOKEN_V2)
        return value

    def _validate(
        self,
        *,
        validated_rows: tuple[DecodedTrustedEnvelopeV2, ...] | None = None,
        context_token: object | None = None,
    ) -> None:
        if type(self) is not TypedTrustedWireBatchReplayV2:
            raise TypeError("V2 typed replay receipt must use the exact type")
        if (validated_rows is None) is not (context_token is None):
            raise TypeError("V2 typed replay receipt context token mismatch")
        if validated_rows is not None and context_token is not _RECEIPT_ISSUE_TOKEN_V2:
            raise TypeError("V2 typed replay receipt context is private")
        if validated_rows is not None and (
            type(validated_rows) is not tuple
            or any(type(row) is not DecodedTrustedEnvelopeV2 for row in validated_rows)
        ):
            raise TypeError("V2 typed replay validated rows must be exact")

        if type(self.batch) is not TrustedWireBatchV2:
            raise TypeError("V2 typed replay receipt needs the exact batch")
        _require_digest_v2(
            self.batch_id,
            "phase2b_trusted_wire_batch_v2_",
            "V2 typed replay stored batch ID",
        )
        _require_digest_v2(
            self.batch_policy_id,
            "phase2b_trusted_wire_batch_v2_policy_",
            "V2 typed replay batch policy ID",
        )
        _require_digest_v2(
            self.replay_policy_id,
            "phase2b_typed_trusted_wire_replay_policy_v2_",
            "V2 typed replay policy ID",
        )
        if (
            self.disposition is not TypedTrustedWireReplayDispositionV2.COMPLETE
            or self.batch_policy_id != TRUSTED_WIRE_BATCH_V2_POLICY_ID
            or self.replay_policy_id != TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID
            or type(self.claim_level) is not str
            or self.claim_level != NON_AUTHORITATIVE_CLAIM_LEVEL
        ):
            raise ValueError("V2 typed replay receipt identity drift")
        _require_exact_bool_claims_v2(
            *(getattr(self, name) for name in (*_TRUE_CLAIMS_V2, *_FALSE_CLAIMS_V2))
        )
        if not all(getattr(self, name) for name in _TRUE_CLAIMS_V2) or any(
            getattr(self, name) for name in _FALSE_CLAIMS_V2
        ):
            raise ValueError("V2 typed replay claim boundary drift")
        if (
            type(self.rows) is not tuple
            or not 1 <= len(self.rows) <= MAXIMUM_BATCH_V2_AUTHORITIES
            or any(type(row) is not DecodedTrustedEnvelopeV2 for row in self.rows)
        ):
            raise TypeError("V2 typed replay receipt needs all exact rows")
        column_specs = (
            (
                self.row_ids,
                "phase2b_typed_trusted_wire_row_v2_",
                "row ID",
            ),
            (
                self.authority_content_ids,
                "phase2b_public_transform_evidence_",
                "authority content ID",
            ),
            (
                self.transform_result_ids,
                "phase2b_exact_transform_result_",
                "transform result ID",
            ),
        )
        for actual, prefix, name in column_specs:
            if type(actual) is not tuple or len(actual) != len(self.rows):
                raise TypeError(f"V2 typed replay {name} column shape drift")
            for item in actual:
                _require_digest_v2(item, prefix, f"V2 typed replay {name}")

        if validated_rows is None:
            batch_id = _validated_batch_id_v2(self.batch)
        else:
            batch_id = self.batch_id
            if self.batch.batch_id != batch_id:
                raise ValueError("V2 typed replay private batch context drift")
        if self.batch_id != batch_id:
            raise ValueError("V2 typed replay stored batch ID drift")
        if len(self.rows) != len(self.batch.envelopes):
            raise ValueError("V2 typed replay row count drifts from batch")

        if validated_rows is None:
            expected_rows = tuple(
                decode_and_replay_typed_trusted_envelope_v2(item.envelope)
                for item in self.batch.envelopes
            )
        else:
            if len(validated_rows) != len(self.rows) or not all(
                validated is stored
                for validated, stored in zip(validated_rows, self.rows)
            ):
                raise ValueError("V2 typed replay validated row context drift")
            expected_rows = validated_rows
        if validated_rows is None:
            for row in self.rows:
                row._validate()
        if self.rows != expected_rows:
            raise ValueError("V2 typed replay rows drift from batch bytes")
        if tuple(row.envelope_id for row in self.rows) != self.batch.envelope_ids:
            raise ValueError("V2 typed replay batch membership drift")

        expected_row_ids = tuple(
            _row_id_v2(row, batch_id, validated=True) for row in expected_rows
        )
        expected_authority_ids = tuple(
            row.authority_content_id for row in self.rows
        )
        expected_transform_ids = tuple(row.transform_result_id for row in self.rows)
        columns = (
            (
                self.row_ids,
                expected_row_ids,
                "phase2b_typed_trusted_wire_row_v2_",
                "row ID",
            ),
            (
                self.authority_content_ids,
                expected_authority_ids,
                "phase2b_public_transform_evidence_",
                "authority content ID",
            ),
            (
                self.transform_result_ids,
                expected_transform_ids,
                "phase2b_exact_transform_result_",
                "transform result ID",
            ),
        )
        for actual, expected, prefix, name in columns:
            if actual != expected:
                raise ValueError(f"V2 typed replay {name} column drift")
        if (
            self.authority_content_ids != self.batch.authority_content_ids
            or self.transform_result_ids != self.batch.transform_result_ids
        ):
            raise ValueError("V2 typed replay public batch roots drift")


    @property
    def receipt_id(self) -> str:
        self._validate()
        return stable_hash(
            {
                "authority_content_ids": self.authority_content_ids,
                "batch_id": self.batch_id,
                "batch_policy_id": self.batch_policy_id,
                "replay_policy_id": self.replay_policy_id,
                "row_ids": self.row_ids,
                "transform_result_ids": self.transform_result_ids,
            },
            prefix="phase2b_typed_trusted_wire_batch_replay_v2_",
        )


def _rejection_v2(
    reason: str,
    authority_count: int,
    batch_id: str | None,
) -> TypedTrustedWireBatchReplayRejectionV2:
    return TypedTrustedWireBatchReplayRejectionV2(
        disposition=TypedTrustedWireReplayDispositionV2.ABSTAIN,
        reason=reason,
        authority_count=authority_count,
        batch_id=batch_id,
    )


def _safe_batch_shape_v2(batch: TrustedWireBatchV2) -> tuple[int, str | None]:
    count = 0
    batch_id: str | None = None
    try:
        if type(batch.envelopes) is tuple:
            count = len(batch.envelopes)
        if type(batch.batch_id) is str:
            _require_digest_v2(
                batch.batch_id,
                "phase2b_trusted_wire_batch_v2_",
                "V2 typed replay candidate batch ID",
            )
            batch_id = batch.batch_id
    except (AttributeError, TypeError, ValueError):
        pass
    return count, batch_id


def _replay_public_batch_raw_v2(
    batch: TrustedWireBatchV2,
) -> tuple[str, tuple[DecodedTrustedEnvelopeV2, ...]]:
    """Validate one exact batch and replay each row exactly once."""

    batch_id = _validated_batch_id_v2(batch)
    rows = tuple(
        decode_and_replay_typed_trusted_envelope_v2(item.envelope)
        for item in batch.envelopes
    )
    if (
        tuple(row.envelope_id for row in rows) != batch.envelope_ids
        or tuple(row.authority_content_id for row in rows)
        != batch.authority_content_ids
        or tuple(row.transform_result_id for row in rows)
        != batch.transform_result_ids
    ):
        raise ValueError("V2 raw public replay drifts from batch roots")
    return batch_id, rows


def replay_typed_trusted_wire_batch_v2(
    *,
    batch: TrustedWireBatchV2,
) -> TypedTrustedWireBatchReplayV2 | TypedTrustedWireBatchReplayRejectionV2:
    """Atomically replay all public rows of an already issued V2 batch."""

    if type(batch) is not TrustedWireBatchV2:
        raise TypeError("V2 typed replay requires the exact trusted-wire batch")
    authority_count, candidate_batch_id = _safe_batch_shape_v2(batch)
    try:
        batch_id, rows = _replay_public_batch_raw_v2(batch)
        receipt = TypedTrustedWireBatchReplayV2._issue(
            _RECEIPT_ISSUE_TOKEN_V2,
            batch=batch,
            batch_id=batch_id,
            rows=rows,
        )
    except (
        AttributeError,
        KeyError,
        OverflowError,
        RecursionError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return _rejection_v2(
            "batch_or_public_typed_replay_failed",
            authority_count,
            candidate_batch_id,
        )
    if receipt.batch_id != batch_id:
        raise RuntimeError("V2 typed replay issued a mismatched receipt")
    return receipt


def _replay_typed_trusted_wire_batch_core_v2(
    *,
    authorities: tuple[PublicTransformEvidenceBundleV2, ...],
    run_id: bytes,
    key_sources: TrustedWireKeySourcesV2,
    expected_batch: TrustedWireBatchV2 | None = None,
    per_case_projection_compiler: Callable[
        [
            int,
            Mapping[tuple[str, str], str],
            PublicTransformEvidenceBundleV2,
        ],
        object,
    ]
    | None = None,
) -> tuple[
    TypedTrustedWireBatchReplayV2 | TypedTrustedWireBatchReplayRejectionV2,
    tuple[object, ...],
]:
    """Build exactly once, then expose only public replay plus aligned projections."""

    if expected_batch is not None and type(expected_batch) is not TrustedWireBatchV2:
        raise TypeError("V2 typed replay expected batch must use the exact type")
    if expected_batch is not None:
        try:
            _validated_batch_id_v2(expected_batch)
        except (
            AttributeError,
            KeyError,
            OverflowError,
            RecursionError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return _rejection_v2(
                "expected_batch_validation_failed",
                len(authorities) if type(authorities) is tuple else 0,
                None,
            ), ()

    try:
        built, projections = _batch_v2._build_trusted_wire_batch_core_v2(
            authorities=authorities,
            run_id=run_id,
            key_sources=key_sources,
            per_case_projection_compiler=per_case_projection_compiler,
        )
    except (
        AttributeError,
        KeyError,
        OverflowError,
        RecursionError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return _rejection_v2(
            "custodian_batch_build_failed",
            len(authorities) if type(authorities) is tuple else 0,
            None,
        ), ()

    if type(built) is TrustedWireBatchRejectionV2:
        try:
            built.__post_init__()
        except (AttributeError, TypeError, ValueError):
            return _rejection_v2(
                "custodian_batch_build_rejection_validation_failed",
                len(authorities) if type(authorities) is tuple else 0,
                None,
            ), ()
        return _rejection_v2(
            "custodian_batch_build_abstained",
            built.authority_count,
            None,
        ), ()
    if type(built) is not TrustedWireBatchV2 or type(projections) is not tuple:
        return _rejection_v2(
            "custodian_batch_build_result_type_drift",
            len(authorities) if type(authorities) is tuple else 0,
            None,
        ), ()

    try:
        built_id = _validated_batch_id_v2(built)
        if expected_batch is not None:
            if built != expected_batch:
                raise ValueError("rebuilt V2 batch differs from expected batch")
        replay = replay_typed_trusted_wire_batch_v2(batch=built)
        if type(replay) is not TypedTrustedWireBatchReplayV2:
            raise ValueError("rebuilt V2 batch did not complete public replay")
        if (
            per_case_projection_compiler is None
            and projections != ()
        ) or (
            per_case_projection_compiler is not None
            and len(projections) != len(replay.rows)
        ):
            raise ValueError("V2 typed replay projection alignment drift")
    except (
        AttributeError,
        KeyError,
        OverflowError,
        RecursionError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return _rejection_v2(
            "custodian_rebuild_or_public_replay_failed",
            len(built.envelopes),
            built_id if "built_id" in locals() else None,
        ), ()
    return replay, projections


def _assert_field_manifests_v2() -> None:
    actual = (
        tuple(field.name for field in fields(TypedTrustedWireBatchReplayRejectionV2)),
        tuple(field.name for field in fields(TypedTrustedWireBatchReplayV2)),
    )
    expected = (_REJECTION_FIELD_MANIFEST_V2, _RECEIPT_FIELD_MANIFEST_V2)
    if actual != expected:
        raise RuntimeError("V2 typed replay field manifest drift")


_assert_field_manifests_v2()


__all__ = (
    "TYPED_TRUSTED_WIRE_REPLAY_V2_POLICY_ID",
    "TYPED_TRUSTED_WIRE_REPLAY_V2_VERSION",
    "TypedTrustedWireBatchReplayRejectionV2",
    "TypedTrustedWireBatchReplayV2",
    "TypedTrustedWireReplayDispositionV2",
    "decode_and_replay_typed_trusted_envelope_v2",
    "replay_typed_trusted_wire_batch_v2",
)
