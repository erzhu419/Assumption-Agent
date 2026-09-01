"""Producer-free wire and verification for Phase-3A-Q6 issuance.

This module deliberately imports only the strict CBOR core.  It replays the
sealed Q4 capability and Q5b trust-amendment artifacts as public bytes, builds
the pinned-status-bound Q6 signature preimage, supplies the pure-Python
Ed25519 verifier used once per signer by the controller, and emits one
structural Q4-compatible seven-field envelope inside a stricter Q6 issuance
receipt.  It has no filesystem, process, network, private-key, or signing
surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256, sha512
from typing import Final, NoReturn

from .strict_cbor_v1 import (
    canonical_cbor_decode,
    canonical_cbor_encode,
    content_hash,
    rfc6962_root,
)


# Exact sealed Q4 projection.
Q4_BODY_TAG: Final = 0x3E00
Q4_BODY_SCHEMA_ID: Final = b"hegel-q4-formal-quotient-certificate-body/1"
Q4_BODY_ROOT_DOMAIN: Final = "HEGEL/Q4/FORMAL_QUOTIENT_CERTIFICATE_BODY/V1"
Q4_RECEIPT_TAG: Final = 0x3E01
Q4_RECEIPT_SCHEMA_ID: Final = (
    b"hegel-q4-formal-quotient-certificate-capability-receipt/1"
)
Q4_RECEIPT_ROOT_DOMAIN: Final = (
    "HEGEL/Q4/FORMAL_QUOTIENT_CERTIFICATE_CAPABILITY_RECEIPT/V1"
)
Q4_CLAIM_KIND: Final = b"OUTSIDE_FROZEN_QUOTIENT_CLOSURE"
Q4_DSL_VERSION: Final = b"hegel-old-dsl-v1.6.0"
Q4_CLOSURE_SEMANTICS_VERSION: Final = b"hegel-quotient-closure-v1.0.1"
Q4_EQUIVALENCE_ID: Final = b"exact_behavior_vector_with_bottom"
Q4_CLAIM_LEVEL: Final = b"LOCAL_DISCLOSED_DEVELOPMENT_CONTROL"
Q4_CAPABILITY_DISPOSITION: Final = b"NOT_READY_FOR_ISSUANCE"
Q4_SUCCESS_STATUS: Final = (
    b"Q4_FORMAL_QUOTIENT_CERTIFICATE_CAPABILITY_ADMITTED_NOT_ISSUED"
)
Q4_MISSING_PREREQUISITES: Final = (
    b"PURPOSE5_TRUST_GENESIS_PIN",
    b"PURPOSE5_CURRENT_KEY_STATUS_BUNDLE",
    b"PURPOSE5_THREE_ACTIVE_SIGNERS",
    b"THREE_OF_THREE_SIGNATURES",
)

# Exact sealed Q5a/Q5b projection.
Q5A_SIGNER_MANIFEST_TAG: Final = 0x3F00
Q5A_SIGNER_MANIFEST_SCHEMA_ID: Final = (
    b"hegel-q5a-purpose5-signer-manifest/1"
)
Q5A_SIGNER_MANIFEST_ROOT_DOMAIN: Final = (
    "HEGEL/Q5A/PURPOSE5_SIGNER_MANIFEST/V1"
)
Q5A_TRUST_GENESIS_TAG: Final = 0x3F01
Q5A_TRUST_GENESIS_SCHEMA_ID: Final = (
    b"hegel-q5a-purpose5-unpinned-trust-genesis-candidate/1"
)
Q5A_TRUST_GENESIS_ROOT_DOMAIN: Final = (
    "HEGEL/Q5A/PURPOSE5_UNPINNED_TRUST_GENESIS_CANDIDATE/V1"
)
Q5A_KEY_STATUS_TAG: Final = 0x3F02
Q5A_KEY_STATUS_SCHEMA_ID: Final = (
    b"hegel-q5a-purpose5-genesis-only-status-candidate/1"
)
Q5A_KEY_STATUS_ROOT_DOMAIN: Final = (
    "HEGEL/Q5A/PURPOSE5_GENESIS_ONLY_STATUS_CANDIDATE/V1"
)
Q5A_CANDIDATE_TAG: Final = 0x3F03
Q5A_CANDIDATE_SCHEMA_ID: Final = (
    b"hegel-q5a-purpose5-signer-genesis-candidate/1"
)
Q5A_CANDIDATE_ROOT_DOMAIN: Final = (
    "HEGEL/Q5A/PURPOSE5_SIGNER_GENESIS_CANDIDATE/V1"
)
Q5A_ACTION_ID: Final = b"Q5A_PURPOSE5_SIGNER_GENESIS_CANDIDATE_V1"
Q5A_SUCCESS_STATUS: Final = (
    b"Q5A_PURPOSE5_SIGNER_GENESIS_CANDIDATE_COMPLETE_NOT_PINNED_NOT_SIGNED"
)
Q5A_CLAIM_LEVEL: Final = b"LOCAL_DISCLOSED_DEVELOPMENT_CONTROL"

Q5B_PINNED_GENESIS_TAG: Final = 0x3F10
Q5B_PINNED_GENESIS_SCHEMA_ID: Final = (
    b"hegel-q5b-purpose5-pinned-trust-genesis/1"
)
Q5B_PINNED_GENESIS_ROOT_DOMAIN: Final = (
    "HEGEL/Q5B/PURPOSE5_PINNED_TRUST_GENESIS/V1"
)
Q5B_CURRENT_STATUS_TAG: Final = 0x3F11
Q5B_CURRENT_STATUS_SCHEMA_ID: Final = (
    b"hegel-q5b-purpose5-current-key-status-bundle/1"
)
Q5B_CURRENT_STATUS_ROOT_DOMAIN: Final = (
    "HEGEL/Q5B/PURPOSE5_CURRENT_KEY_STATUS_BUNDLE/V1"
)
Q5B_RECEIPT_TAG: Final = 0x3F12
Q5B_RECEIPT_SCHEMA_ID: Final = (
    b"hegel-q5b-purpose5-trust-amendment-receipt/1"
)
Q5B_RECEIPT_ROOT_DOMAIN: Final = (
    "HEGEL/Q5B/PURPOSE5_TRUST_AMENDMENT_RECEIPT/V1"
)
Q5B_ACTION_ID: Final = (
    b"Q5B_PURPOSE5_TRUST_GENESIS_CURRENT_STATUS_PIN_AMENDMENT_V1"
)
Q5B_SUCCESS_STATUS: Final = (
    b"Q5B_PURPOSE5_TRUST_AMENDMENT_ADMITTED_CURRENT_NOT_SIGNED_NOT_ISSUED"
)
Q5B_CLAIM_LEVEL: Final = b"LOCAL_DISCLOSED_DEVELOPMENT_CONTROL"

PURPOSE5_FINAL_CERTIFICATE_SIGNER: Final = 5
INITIAL_KEY_EPOCH: Final = 1
QUOTIENT_CERTIFICATE_KIND_ID: Final = 2
REQUIRED_SIGNATURE_COUNT: Final = 3
SIGNER_ORDINALS: Final = (1, 2, 3)
SIGNER_ROLE_NAMES: Final = (
    "FINAL_CERTIFICATE_CUSTODIAN",
    "FINAL_CERTIFICATE_PYTHON_REPLAY_ATTESTER",
    "FINAL_CERTIFICATE_RUST_REPLAY_ATTESTER",
)
FUTURE_STATUS_TRANSITION_THRESHOLD: Final = 2
FUTURE_STATUS_TRANSITION_ELIGIBLE_COUNT: Final = 3
Q6_SIGNATURE_PREIMAGE_POLICY_ID: Final = (
    b"hegel-q6-purpose5-pinned-status-bound-signature-preimage/1"
)

# New Q6 wire.
Q6_PREIMAGE_TAG: Final = 0x3F20
Q6_PREIMAGE_SCHEMA_ID: Final = (
    b"hegel-q6-purpose5-formal-quotient-certificate-signature-preimage/1"
)
Q6_PREIMAGE_ROOT_DOMAIN: Final = (
    "HEGEL/Q6/PURPOSE5_FORMAL_QUOTIENT_CERTIFICATE_SIGNATURE_PREIMAGE/V1"
)
Q6_SIGNATURE_MESSAGE_PREFIX: Final = (
    b"HEGEL/Q6/PURPOSE5_FORMAL_QUOTIENT_CERTIFICATE_SIGNATURE/V1\x00"
)
Q6_RECEIPT_TAG: Final = 0x3F21
Q6_RECEIPT_SCHEMA_ID: Final = (
    b"hegel-q6-purpose5-formal-quotient-certificate-issuance-receipt/1"
)
Q6_RECEIPT_ROOT_DOMAIN: Final = (
    "HEGEL/Q6/PURPOSE5_FORMAL_QUOTIENT_CERTIFICATE_ISSUANCE_RECEIPT/V1"
)
Q6_ACTION_ID: Final = "Q6_PURPOSE5_THREE_OF_THREE_FORMAL_QUOTIENT_CERTIFICATE_ISSUANCE_V1"
Q6_SUCCESS_STATUS: Final = (
    "Q6_PURPOSE5_THREE_OF_THREE_FORMAL_QUOTIENT_CERTIFICATE_ISSUED"
)
Q6_CLAIM_LEVEL: Final = "LOCAL_DISCLOSED_DEVELOPMENT_CONTROL"


class Q6FormalIssuanceWireError(ValueError):
    """One public predecessor, signature, or Q6 wire object differs."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q6FormalIssuanceWireError(code, detail)


def _bytes(value: object, length: int, label: str) -> bytes:
    if type(value) is not bytes or len(value) != length:
        _fail("REJECT_Q6_BYTES", f"{label} must be {length} bytes")
    return value


def _bytes32(value: object, label: str) -> bytes:
    return _bytes(value, 32, label)


def _exact_int(value: object, expected: int) -> bool:
    return type(value) is int and value == expected


def _exact_bool(value: object, expected: bool) -> bool:
    return type(value) is bool and value is expected


def _decode(payload: bytes, label: str) -> object:
    if type(payload) is not bytes:
        raise TypeError(f"{label} payload must be exact bytes")
    try:
        return canonical_cbor_decode(payload)
    except ValueError as error:
        _fail("REJECT_Q6_CBOR", f"{label}: {error}")


def _repository_commit(value: object, label: str) -> bytes:
    if (
        type(value) is not tuple
        or len(value) != 2
        or not _exact_int(value[0], 1)
        or type(value[1]) is not bytes
        or len(value[1]) != 20
    ):
        _fail("REJECT_Q6_REPOSITORY_COMMIT", f"{label} wire differs")
    return value[1]


@dataclass(frozen=True, slots=True)
class SealedQ4QuotientCertificateBodyV1:
    canonical_object: tuple[object, ...]

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object)

    @property
    def body_hash(self) -> bytes:
        return content_hash(Q4_BODY_ROOT_DOMAIN, self.canonical_object)

    @property
    def q3_candidate_root(self) -> bytes:
        return self.canonical_object[28]  # type: ignore[return-value]

    @property
    def q4_source_commit_raw(self) -> bytes:
        return self.canonical_object[31][1]  # type: ignore[index,return-value]

    @property
    def q4_source_closure_root(self) -> bytes:
        return self.canonical_object[32]  # type: ignore[return-value]


def decode_sealed_q4_body_v1(payload: bytes) -> SealedQ4QuotientCertificateBodyV1:
    value = _decode(payload, "sealed Q4 quotient-certificate body")
    if (
        type(value) is not tuple
        or len(value) != 34
        or value[:7]
        != (
            1,
            Q4_BODY_TAG,
            Q4_BODY_SCHEMA_ID,
            Q4_CLAIM_KIND,
            Q4_DSL_VERSION,
            Q4_CLOSURE_SEMANTICS_VERSION,
            1,
        )
        or value[9] != Q4_EQUIVALENCE_ID
        or value[20:23] != (418, 0, rfc6962_root(()))
        or type(value[23]) is not tuple
        or len(value[23]) != 7
        or not _exact_int(value[23][0], 2)
        or value[23][4:6] != (89, 1)
        or type(value[23][6]) is not tuple
        or len(value[23][6]) != 1
        or value[33] != Q4_CLAIM_LEVEL
    ):
        _fail("REJECT_Q6_Q4_BODY", "sealed Q4 body wire differs")
    for index in (*range(7, 9), *range(10, 20), *range(24, 29), 30, 32):
        _bytes32(value[index], f"sealed Q4 body field {index}")
    for nested, label in (
        (value[23][1], "null universe root"),
        (value[23][2], "null truth root"),
        (value[23][3], "null role result root"),
        (value[23][6][0], "null matching class id"),
    ):
        _bytes32(nested, label)
    _repository_commit(value[29], "Q3 source commit")
    _repository_commit(value[31], "Q4 source commit")
    body = SealedQ4QuotientCertificateBodyV1(value)
    if body.canonical_bytes != payload:
        _fail("REJECT_Q6_Q4_BODY", "sealed Q4 body replay differs")
    return body


@dataclass(frozen=True, slots=True)
class SealedQ4CapabilityArtifactV1:
    canonical_object: tuple[object, ...]
    body: SealedQ4QuotientCertificateBodyV1

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object)

    @property
    def receipt_root(self) -> bytes:
        return content_hash(Q4_RECEIPT_ROOT_DOMAIN, self.canonical_object)


def decode_sealed_q4_capability_artifact_v1(
    payload: bytes,
) -> SealedQ4CapabilityArtifactV1:
    value = _decode(payload, "sealed Q4 capability artifact")
    fixed_tail = (
        PURPOSE5_FINAL_CERTIFICATE_SIGNER,
        Q4_MISSING_PREREQUISITES,
        0,
        REQUIRED_SIGNATURE_COUNT,
        0,
        REQUIRED_SIGNATURE_COUNT,
        Q4_CAPABILITY_DISPOSITION,
        Q4_SUCCESS_STATUS,
        True,
        False,
        False,
        False,
        False,
        False,
    )
    if (
        type(value) is not tuple
        or len(value) != 20
        or value[:3] != (1, Q4_RECEIPT_TAG, Q4_RECEIPT_SCHEMA_ID)
        or type(value[3]) is not tuple
        or value[6:] != fixed_tail
    ):
        _fail("REJECT_Q6_Q4_ARTIFACT", "sealed Q4 capability wire differs")
    body = decode_sealed_q4_body_v1(canonical_cbor_encode(value[3]))
    if value[4] != body.body_hash or value[5] != body.q3_candidate_root:
        _fail("REJECT_Q6_Q4_ARTIFACT", "sealed Q4 body binding differs")
    artifact = SealedQ4CapabilityArtifactV1(value, body)
    if artifact.canonical_bytes != payload:
        _fail("REJECT_Q6_Q4_ARTIFACT", "sealed Q4 capability replay differs")
    return artifact


@dataclass(frozen=True, slots=True)
class SealedPurpose5SignerManifestV1:
    signer_ordinal: int
    key_id: bytes
    public_key: bytes

    def __post_init__(self) -> None:
        if type(self.signer_ordinal) is not int or self.signer_ordinal not in SIGNER_ORDINALS:
            _fail("REJECT_Q6_SIGNER", "signer ordinal differs")
        _bytes(self.key_id, 16, "Purpose-5 key id")
        _bytes32(self.public_key, "Purpose-5 public key")
        if self.key_id != sha256(self.public_key).digest()[:16]:
            _fail("REJECT_Q6_SIGNER", "Purpose-5 key id differs")

    @property
    def signer_role(self) -> str:
        return SIGNER_ROLE_NAMES[self.signer_ordinal - 1]

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q5A_SIGNER_MANIFEST_TAG,
            Q5A_SIGNER_MANIFEST_SCHEMA_ID,
            PURPOSE5_FINAL_CERTIFICATE_SIGNER,
            self.signer_ordinal,
            self.signer_role.encode("ascii"),
            INITIAL_KEY_EPOCH,
            self.key_id,
            self.public_key,
        )

    @property
    def manifest_root(self) -> bytes:
        return content_hash(Q5A_SIGNER_MANIFEST_ROOT_DOMAIN, self.canonical_object())


def _decode_signer_manifest(value: object) -> SealedPurpose5SignerManifestV1:
    if (
        type(value) is not tuple
        or len(value) != 9
        or value[:4]
        != (
            1,
            Q5A_SIGNER_MANIFEST_TAG,
            Q5A_SIGNER_MANIFEST_SCHEMA_ID,
            PURPOSE5_FINAL_CERTIFICATE_SIGNER,
        )
        or type(value[4]) is not int
        or value[4] not in SIGNER_ORDINALS
        or value[5] != SIGNER_ROLE_NAMES[value[4] - 1].encode("ascii")
        or not _exact_int(value[6], INITIAL_KEY_EPOCH)
    ):
        _fail("REJECT_Q6_SIGNER", "sealed Purpose-5 signer wire differs")
    manifest = SealedPurpose5SignerManifestV1(value[4], value[7], value[8])
    if manifest.canonical_object() != value:
        _fail("REJECT_Q6_SIGNER", "sealed Purpose-5 signer replay differs")
    return manifest


def _validate_manifests(
    manifests: object,
) -> tuple[SealedPurpose5SignerManifestV1, ...]:
    if (
        type(manifests) is not tuple
        or len(manifests) != REQUIRED_SIGNATURE_COUNT
        or any(type(item) is not SealedPurpose5SignerManifestV1 for item in manifests)
        or tuple(item.signer_ordinal for item in manifests) != SIGNER_ORDINALS
        or len({item.key_id for item in manifests}) != REQUIRED_SIGNATURE_COUNT
        or len({item.public_key for item in manifests}) != REQUIRED_SIGNATURE_COUNT
        or len({item.manifest_root for item in manifests}) != REQUIRED_SIGNATURE_COUNT
    ):
        _fail("REJECT_Q6_SIGNER_SET", "three ordered distinct Purpose-5 signers required")
    return manifests


@dataclass(frozen=True, slots=True)
class SealedQ5aCandidateProjectionV1:
    canonical_object: tuple[object, ...]
    q4_capability_receipt_root: bytes
    q4_body_hash: bytes
    signer_manifests: tuple[SealedPurpose5SignerManifestV1, ...]
    unpinned_genesis_root: bytes
    genesis_only_status_root: bytes
    q4_source_commit_raw: bytes
    q4_source_closure_root: bytes
    q5a_source_commit_raw: bytes
    q5a_source_closure_root: bytes

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object)

    @property
    def candidate_root(self) -> bytes:
        return content_hash(Q5A_CANDIDATE_ROOT_DOMAIN, self.canonical_object)


def _decode_q5a_candidate(value: object) -> SealedQ5aCandidateProjectionV1:
    fixed_tail = (
        Q6_SIGNATURE_PREIMAGE_POLICY_ID,
        True,
        3,
        0,
        0,
        Q5A_ACTION_ID,
        Q5A_SUCCESS_STATUS,
        Q5A_CLAIM_LEVEL,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    )
    if (
        type(value) is not tuple
        or len(value) != 34
        or value[:3] != (1, Q5A_CANDIDATE_TAG, Q5A_CANDIDATE_SCHEMA_ID)
        or any(type(value[index]) is not tuple for index in (9, 11, 13, 14, 16))
        or value[18:] != fixed_tail
    ):
        _fail("REJECT_Q6_Q5A_ARTIFACT", "sealed Q5a candidate wire differs")
    for index in (3, 4, 5, 6, 7, 8, 10, 12, 15, 17):
        _bytes32(value[index], f"sealed Q5a field {index}")
    q4_source_commit = _repository_commit(value[9], "Q4 source commit")
    q5a_source_commit = _repository_commit(value[11], "Q5a source commit")
    manifests = _validate_manifests(tuple(_decode_signer_manifest(row) for row in value[13]))
    manifest_rows = tuple(item.canonical_object() for item in manifests)
    manifest_roots = tuple(item.manifest_root for item in manifests)
    genesis = value[14]
    if (
        len(genesis) != 12
        or genesis[:5]
        != (
            1,
            Q5A_TRUST_GENESIS_TAG,
            Q5A_TRUST_GENESIS_SCHEMA_ID,
            PURPOSE5_FINAL_CERTIFICATE_SIGNER,
            INITIAL_KEY_EPOCH,
        )
        or genesis[5] != manifest_rows
        or genesis[6] != manifest_roots
        or genesis[7:] != (3, 3, Q6_SIGNATURE_PREIMAGE_POLICY_ID, True, False)
    ):
        _fail("REJECT_Q6_Q5A_ARTIFACT", "sealed Q5a genesis differs")
    genesis_root = content_hash(Q5A_TRUST_GENESIS_ROOT_DOMAIN, genesis)
    status = value[16]
    if (
        len(status) != 13
        or status[:3] != (1, Q5A_KEY_STATUS_TAG, Q5A_KEY_STATUS_SCHEMA_ID)
        or status[3] != genesis
        or status[4] != genesis_root
        or not _exact_int(status[5], INITIAL_KEY_EPOCH)
        or status[6] != manifest_roots
        or status[7:] != ((), (), Q6_SIGNATURE_PREIMAGE_POLICY_ID, False, 0, 0)
    ):
        _fail("REJECT_Q6_Q5A_ARTIFACT", "sealed Q5a status differs")
    status_root = content_hash(Q5A_KEY_STATUS_ROOT_DOMAIN, status)
    if value[13] != manifest_rows or value[15] != genesis_root or value[17] != status_root:
        _fail("REJECT_Q6_Q5A_ARTIFACT", "sealed Q5a nested roots differ")
    result = SealedQ5aCandidateProjectionV1(
        value,
        value[7],
        value[8],
        manifests,
        genesis_root,
        status_root,
        q4_source_commit,
        value[10],
        q5a_source_commit,
        value[12],
    )
    if result.canonical_bytes != canonical_cbor_encode(value):
        _fail("REJECT_Q6_Q5A_ARTIFACT", "sealed Q5a replay differs")
    return result


@dataclass(frozen=True, slots=True)
class SealedPinnedTrustGenesisV1:
    canonical_object: tuple[object, ...]
    signer_manifests: tuple[SealedPurpose5SignerManifestV1, ...]

    @property
    def genesis_root(self) -> bytes:
        return content_hash(Q5B_PINNED_GENESIS_ROOT_DOMAIN, self.canonical_object)


def _decode_pinned_genesis(value: object) -> SealedPinnedTrustGenesisV1:
    if (
        type(value) is not tuple
        or len(value) != 17
        or value[:5]
        != (
            1,
            Q5B_PINNED_GENESIS_TAG,
            Q5B_PINNED_GENESIS_SCHEMA_ID,
            PURPOSE5_FINAL_CERTIFICATE_SIGNER,
            INITIAL_KEY_EPOCH,
        )
        or type(value[7]) is not tuple
        or type(value[8]) is not tuple
        or type(value[9]) is not tuple
        or value[10:]
        != (
            3,
            3,
            FUTURE_STATUS_TRANSITION_THRESHOLD,
            FUTURE_STATUS_TRANSITION_ELIGIBLE_COUNT,
            Q6_SIGNATURE_PREIMAGE_POLICY_ID,
            True,
            True,
        )
    ):
        _fail("REJECT_Q6_PINNED_GENESIS", "pinned genesis wire differs")
    _bytes32(value[5], "Q5a candidate root")
    _bytes32(value[6], "Q5a unpinned genesis root")
    manifests = _validate_manifests(tuple(_decode_signer_manifest(row) for row in value[7]))
    roots = tuple(item.manifest_root for item in manifests)
    key_ids = tuple(item.key_id for item in manifests)
    if value[8] != roots or value[9] != key_ids:
        _fail("REJECT_Q6_PINNED_GENESIS", "pinned signer roots or key ids differ")
    return SealedPinnedTrustGenesisV1(value, manifests)


@dataclass(frozen=True, slots=True)
class SealedCurrentKeyStatusV1:
    canonical_object: tuple[object, ...]
    pinned_genesis: SealedPinnedTrustGenesisV1

    @property
    def status_root(self) -> bytes:
        return content_hash(Q5B_CURRENT_STATUS_ROOT_DOMAIN, self.canonical_object)

    @property
    def signer_manifests(self) -> tuple[SealedPurpose5SignerManifestV1, ...]:
        return self.pinned_genesis.signer_manifests


def _decode_current_status(value: object) -> SealedCurrentKeyStatusV1:
    if (
        type(value) is not tuple
        or len(value) != 19
        or value[:4]
        != (
            1,
            Q5B_CURRENT_STATUS_TAG,
            Q5B_CURRENT_STATUS_SCHEMA_ID,
            PURPOSE5_FINAL_CERTIFICATE_SIGNER,
        )
        or type(value[4]) is not tuple
        or not _exact_int(value[6], INITIAL_KEY_EPOCH)
        or type(value[7]) is not tuple
        or type(value[8]) is not tuple
        or value[9:]
        != (
            SIGNER_ORDINALS,
            3,
            FUTURE_STATUS_TRANSITION_THRESHOLD,
            FUTURE_STATUS_TRANSITION_ELIGIBLE_COUNT,
            Q6_SIGNATURE_PREIMAGE_POLICY_ID,
            True,
            True,
            0,
            0,
            0,
        )
    ):
        _fail("REJECT_Q6_CURRENT_STATUS", "current status wire differs")
    pinned = _decode_pinned_genesis(value[4])
    roots = tuple(item.manifest_root for item in pinned.signer_manifests)
    key_ids = tuple(item.key_id for item in pinned.signer_manifests)
    if value[5] != pinned.genesis_root or value[7] != roots or value[8] != key_ids:
        _fail("REJECT_Q6_CURRENT_STATUS", "current status signer binding differs")
    return SealedCurrentKeyStatusV1(value, pinned)


@dataclass(frozen=True, slots=True)
class SealedQ5bTrustAmendmentArtifactV1:
    canonical_object: tuple[object, ...]
    q5a_candidate: SealedQ5aCandidateProjectionV1
    pinned_genesis: SealedPinnedTrustGenesisV1
    current_status: SealedCurrentKeyStatusV1
    q5b_source_commit_raw: bytes
    q5b_source_closure_root: bytes

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object)

    @property
    def receipt_root(self) -> bytes:
        return content_hash(Q5B_RECEIPT_ROOT_DOMAIN, self.canonical_object)

    @property
    def signer_manifests(self) -> tuple[SealedPurpose5SignerManifestV1, ...]:
        return self.current_status.signer_manifests

    @property
    def role_order_key_ids(self) -> tuple[bytes, ...]:
        return tuple(item.key_id for item in self.signer_manifests)


def decode_sealed_q5b_trust_amendment_artifact_v1(
    payload: bytes,
) -> SealedQ5bTrustAmendmentArtifactV1:
    value = _decode(payload, "sealed Q5b trust-amendment artifact")
    fixed_tail = (
        PURPOSE5_FINAL_CERTIFICATE_SIGNER,
        INITIAL_KEY_EPOCH,
        3,
        FUTURE_STATUS_TRANSITION_THRESHOLD,
        FUTURE_STATUS_TRANSITION_ELIGIBLE_COUNT,
        Q6_SIGNATURE_PREIMAGE_POLICY_ID,
        True,
        Q5B_ACTION_ID,
        Q5B_SUCCESS_STATUS,
        Q5B_CLAIM_LEVEL,
        True,
        True,
        True,
        0,
        0,
        0,
        False,
        False,
        False,
        False,
        False,
    )
    if (
        type(value) is not tuple
        or len(value) != 43
        or value[:3] != (1, Q5B_RECEIPT_TAG, Q5B_RECEIPT_SCHEMA_ID)
        or any(type(value[index]) is not tuple for index in (8, 12, 13, 14, 16, 18, 20))
        or value[22:] != fixed_tail
    ):
        _fail("REJECT_Q6_Q5B_ARTIFACT", "sealed Q5b receipt wire differs")
    for index in (3, 4, 5, 6, 7, 9, 10, 11, 15, 17, 19, 21):
        _bytes32(value[index], f"sealed Q5b field {index}")
    q5a_source_commit = _repository_commit(value[14], "Q5a source commit")
    q5b_source_commit = _repository_commit(value[16], "Q5b source commit")
    candidate = _decode_q5a_candidate(value[8])
    pinned = _decode_pinned_genesis(value[18])
    status = _decode_current_status(value[20])
    manifest_roots = tuple(item.manifest_root for item in candidate.signer_manifests)
    key_ids = tuple(item.key_id for item in candidate.signer_manifests)
    if (
        sha256(candidate.canonical_bytes).digest() != value[7]
        or candidate.candidate_root != value[9]
        or candidate.unpinned_genesis_root != value[10]
        or candidate.genesis_only_status_root != value[11]
        or value[12] != manifest_roots
        or value[13] != key_ids
        or candidate.q5a_source_commit_raw != q5a_source_commit
        or candidate.q5a_source_closure_root != value[15]
        or pinned.canonical_object[5] != candidate.candidate_root
        or pinned.canonical_object[6] != candidate.unpinned_genesis_root
        or pinned.signer_manifests != candidate.signer_manifests
        or status.pinned_genesis != pinned
        or value[19] != pinned.genesis_root
        or value[21] != status.status_root
    ):
        _fail("REJECT_Q6_Q5B_ARTIFACT", "sealed Q5b object graph differs")
    result = SealedQ5bTrustAmendmentArtifactV1(
        value,
        candidate,
        pinned,
        status,
        q5b_source_commit,
        value[17],
    )
    if result.canonical_bytes != payload:
        _fail("REJECT_Q6_Q5B_ARTIFACT", "sealed Q5b replay differs")
    return result


@dataclass(frozen=True, slots=True)
class Q6SignaturePreimageV1:
    created_at_unix_seconds: int
    q4_body: SealedQ4QuotientCertificateBodyV1
    q4_capability_receipt_root: bytes
    q5b_receipt_root: bytes
    pinned_genesis: SealedPinnedTrustGenesisV1
    current_status: SealedCurrentKeyStatusV1
    role_order_key_ids: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if type(self.created_at_unix_seconds) is not int or self.created_at_unix_seconds < 0:
            _fail("REJECT_Q6_PREIMAGE", "created_at must be a nonnegative integer")
        if type(self.q4_body) is not SealedQ4QuotientCertificateBodyV1:
            _fail("REJECT_Q6_PREIMAGE", "Q4 body type differs")
        _bytes32(self.q4_capability_receipt_root, "Q4 capability receipt root")
        _bytes32(self.q5b_receipt_root, "Q5b receipt root")
        if (
            type(self.pinned_genesis) is not SealedPinnedTrustGenesisV1
            or type(self.current_status) is not SealedCurrentKeyStatusV1
            or self.current_status.pinned_genesis != self.pinned_genesis
            or self.role_order_key_ids
            != tuple(item.key_id for item in self.current_status.signer_manifests)
        ):
            _fail("REJECT_Q6_PREIMAGE", "pinned/current signer binding differs")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q6_PREIMAGE_TAG,
            Q6_PREIMAGE_SCHEMA_ID,
            Q6_SIGNATURE_PREIMAGE_POLICY_ID,
            QUOTIENT_CERTIFICATE_KIND_ID,
            self.created_at_unix_seconds,
            INITIAL_KEY_EPOCH,
            self.q4_body.canonical_object,
            self.q4_body.body_hash,
            self.q4_capability_receipt_root,
            self.q5b_receipt_root,
            self.pinned_genesis.canonical_object,
            self.pinned_genesis.genesis_root,
            self.current_status.canonical_object,
            self.current_status.status_root,
            self.role_order_key_ids,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def preimage_root(self) -> bytes:
        return content_hash(Q6_PREIMAGE_ROOT_DOMAIN, self.canonical_object())

    @property
    def signature_message(self) -> bytes:
        return Q6_SIGNATURE_MESSAGE_PREFIX + self.preimage_root


def build_q6_signature_preimage_v1(
    q4_artifact: SealedQ4CapabilityArtifactV1,
    q5b_artifact: SealedQ5bTrustAmendmentArtifactV1,
    *,
    created_at_unix_seconds: int,
) -> Q6SignaturePreimageV1:
    if (
        type(q4_artifact) is not SealedQ4CapabilityArtifactV1
        or type(q5b_artifact) is not SealedQ5bTrustAmendmentArtifactV1
    ):
        _fail("REJECT_Q6_PREIMAGE", "sealed predecessor type differs")
    if (
        q5b_artifact.q5a_candidate.q4_body_hash != q4_artifact.body.body_hash
        or q5b_artifact.q5a_candidate.q4_capability_receipt_root
        != q4_artifact.receipt_root
        or q5b_artifact.q5a_candidate.q4_source_commit_raw
        != q4_artifact.body.q4_source_commit_raw
        or q5b_artifact.q5a_candidate.q4_source_closure_root
        != q4_artifact.body.q4_source_closure_root
    ):
        _fail("REJECT_Q6_PREIMAGE", "Q4/Q5b lineage binding differs")
    return Q6SignaturePreimageV1(
        created_at_unix_seconds,
        q4_artifact.body,
        q4_artifact.receipt_root,
        q5b_artifact.receipt_root,
        q5b_artifact.pinned_genesis,
        q5b_artifact.current_status,
        q5b_artifact.role_order_key_ids,
    )


def decode_q6_signature_preimage_v1(
    payload: bytes,
    *,
    q4_artifact: SealedQ4CapabilityArtifactV1,
    q5b_artifact: SealedQ5bTrustAmendmentArtifactV1,
) -> Q6SignaturePreimageV1:
    value = _decode(payload, "Q6 signature preimage")
    if (
        type(value) is not tuple
        or len(value) != 16
        or value[:5]
        != (
            1,
            Q6_PREIMAGE_TAG,
            Q6_PREIMAGE_SCHEMA_ID,
            Q6_SIGNATURE_PREIMAGE_POLICY_ID,
            QUOTIENT_CERTIFICATE_KIND_ID,
        )
        or type(value[5]) is not int
        or value[5] < 0
        or not _exact_int(value[6], INITIAL_KEY_EPOCH)
        or type(value[7]) is not tuple
        or type(value[11]) is not tuple
        or type(value[13]) is not tuple
        or type(value[15]) is not tuple
    ):
        _fail("REJECT_Q6_PREIMAGE", "signature preimage wire differs")
    expected = build_q6_signature_preimage_v1(
        q4_artifact,
        q5b_artifact,
        created_at_unix_seconds=value[5],
    )
    if expected.canonical_bytes != payload:
        _fail("REJECT_Q6_PREIMAGE", "signature preimage binding differs")
    return expected


# Strict, dependency-free RFC 8032 Ed25519 verification.
_ED_Q: Final = (1 << 255) - 19
_ED_L: Final = (1 << 252) + 27742317777372353535851937790883648493
_ED_D: Final = (-121665 * pow(121666, _ED_Q - 2, _ED_Q)) % _ED_Q
_ED_I: Final = pow(2, (_ED_Q - 1) // 4, _ED_Q)
_ED_IDENTITY: Final = (0, 1, 1, 0)


def _ed_recover_x(y: int, sign: int) -> int:
    if y >= _ED_Q:
        _fail("REJECT_Q6_SIGNATURE", "Ed25519 y-coordinate is noncanonical")
    yy = y * y % _ED_Q
    xx = (yy - 1) * pow((_ED_D * yy + 1) % _ED_Q, _ED_Q - 2, _ED_Q) % _ED_Q
    x = pow(xx, (_ED_Q + 3) // 8, _ED_Q)
    if (x * x - xx) % _ED_Q != 0:
        x = x * _ED_I % _ED_Q
    if (x * x - xx) % _ED_Q != 0:
        _fail("REJECT_Q6_SIGNATURE", "Ed25519 point is not on the curve")
    if (x & 1) != sign:
        x = _ED_Q - x
    if x == 0 and sign:
        _fail("REJECT_Q6_SIGNATURE", "Ed25519 x sign is noncanonical")
    return x


def _ed_decode_point(encoded: bytes, label: str) -> tuple[int, int, int, int]:
    _bytes(encoded, 32, label)
    integer = int.from_bytes(encoded, "little")
    sign = integer >> 255
    y = integer & ((1 << 255) - 1)
    x = _ed_recover_x(y, sign)
    point = (x, y, 1, x * y % _ED_Q)
    if _ed_encode_point(point) != encoded:
        _fail("REJECT_Q6_SIGNATURE", f"{label} encoding is noncanonical")
    return point


def _ed_add(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x1, y1, z1, t1 = left
    x2, y2, z2, t2 = right
    a = (y1 - x1) * (y2 - x2) % _ED_Q
    b = (y1 + x1) * (y2 + x2) % _ED_Q
    c = 2 * _ED_D * t1 * t2 % _ED_Q
    d = 2 * z1 * z2 % _ED_Q
    e = (b - a) % _ED_Q
    f = (d - c) % _ED_Q
    g = (d + c) % _ED_Q
    h = (b + a) % _ED_Q
    return (e * f % _ED_Q, g * h % _ED_Q, f * g % _ED_Q, e * h % _ED_Q)


def _ed_scalar_mult(
    point: tuple[int, int, int, int], scalar: int
) -> tuple[int, int, int, int]:
    result = _ED_IDENTITY
    addend = point
    while scalar:
        if scalar & 1:
            result = _ed_add(result, addend)
        addend = _ed_add(addend, addend)
        scalar >>= 1
    return result


def _ed_equal(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    return (
        (left[0] * right[2] - right[0] * left[2]) % _ED_Q == 0
        and (left[1] * right[2] - right[1] * left[2]) % _ED_Q == 0
    )


def _ed_encode_point(point: tuple[int, int, int, int]) -> bytes:
    inverse = pow(point[2], _ED_Q - 2, _ED_Q)
    x = point[0] * inverse % _ED_Q
    y = point[1] * inverse % _ED_Q
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


_ED_BASE: Final = _ed_decode_point(bytes.fromhex("58" + "66" * 31), "Ed25519 base point")


def verify_ed25519_v1(public_key: bytes, signature: bytes, message: bytes) -> None:
    """Strictly verify one Ed25519 signature and reject non-prime-order points."""

    _bytes(public_key, 32, "Ed25519 public key")
    _bytes(signature, 64, "Ed25519 signature")
    if type(message) is not bytes:
        raise TypeError("Ed25519 message must be exact bytes")
    public_point = _ed_decode_point(public_key, "Ed25519 public key")
    r_encoded = signature[:32]
    r_point = _ed_decode_point(r_encoded, "Ed25519 R")
    scalar = int.from_bytes(signature[32:], "little")
    if scalar >= _ED_L:
        _fail("REJECT_Q6_SIGNATURE", "Ed25519 S is noncanonical")
    if (
        _ed_equal(public_point, _ED_IDENTITY)
        or _ed_equal(r_point, _ED_IDENTITY)
        or not _ed_equal(_ed_scalar_mult(public_point, _ED_L), _ED_IDENTITY)
        or not _ed_equal(_ed_scalar_mult(r_point, _ED_L), _ED_IDENTITY)
    ):
        _fail("REJECT_Q6_SIGNATURE", "Ed25519 point is not prime-order")
    challenge = int.from_bytes(
        sha512(r_encoded + public_key + message).digest(), "little"
    ) % _ED_L
    if not _ed_equal(
        _ed_scalar_mult(_ED_BASE, scalar),
        _ed_add(r_point, _ed_scalar_mult(public_point, challenge)),
    ):
        _fail("REJECT_Q6_SIGNATURE", "Ed25519 signature rejected")


@dataclass(frozen=True, slots=True)
class SignatureRecordV1:
    key_id: bytes
    signature: bytes

    def __post_init__(self) -> None:
        _bytes(self.key_id, 16, "signature key id")
        _bytes(self.signature, 64, "Ed25519 signature")

    def canonical_object(self) -> tuple[object, ...]:
        return (self.key_id, self.signature)


@dataclass(frozen=True, slots=True)
class Q4CompatibleFinalCertificateEnvelopeV1:
    created_at_unix_seconds: int
    body: SealedQ4QuotientCertificateBodyV1
    signatures: tuple[SignatureRecordV1, ...]

    def __post_init__(self) -> None:
        if (
            type(self.created_at_unix_seconds) is not int
            or self.created_at_unix_seconds < 0
            or type(self.body) is not SealedQ4QuotientCertificateBodyV1
            or type(self.signatures) is not tuple
            or len(self.signatures) != REQUIRED_SIGNATURE_COUNT
            or any(type(item) is not SignatureRecordV1 for item in self.signatures)
            or tuple(item.key_id for item in self.signatures)
            != tuple(sorted(item.key_id for item in self.signatures))
            or len({item.key_id for item in self.signatures}) != REQUIRED_SIGNATURE_COUNT
        ):
            _fail("REJECT_Q6_ENVELOPE", "Q4-compatible envelope binding differs")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            QUOTIENT_CERTIFICATE_KIND_ID,
            self.created_at_unix_seconds,
            INITIAL_KEY_EPOCH,
            self.body.canonical_object,
            self.body.body_hash,
            tuple(item.canonical_object() for item in self.signatures),
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())


def build_q4_compatible_envelope_v1(
    preimage: Q6SignaturePreimageV1,
    q5b_artifact: SealedQ5bTrustAmendmentArtifactV1,
    *,
    role_order_signatures: tuple[bytes, bytes, bytes],
) -> Q4CompatibleFinalCertificateEnvelopeV1:
    """Build the structural envelope after per-signer verification checkpoints.

    This function intentionally performs zero cryptographic verification.  The
    controller must call :func:`verify_ed25519_v1` exactly once when each
    signer result is admitted, before that signer enters the completed prefix.
    """

    if (
        type(preimage) is not Q6SignaturePreimageV1
        or type(q5b_artifact) is not SealedQ5bTrustAmendmentArtifactV1
        or type(role_order_signatures) is not tuple
        or len(role_order_signatures) != REQUIRED_SIGNATURE_COUNT
    ):
        _fail("REJECT_Q6_ENVELOPE", "signature input shape differs")
    if (
        preimage.q5b_receipt_root != q5b_artifact.receipt_root
        or preimage.current_status != q5b_artifact.current_status
    ):
        _fail("REJECT_Q6_ENVELOPE", "preimage/current-status binding differs")
    records = [
        SignatureRecordV1(manifest.key_id, signature)
        for manifest, signature in zip(
            q5b_artifact.signer_manifests, role_order_signatures
        )
    ]
    return Q4CompatibleFinalCertificateEnvelopeV1(
        preimage.created_at_unix_seconds,
        preimage.q4_body,
        tuple(sorted(records, key=lambda item: item.key_id)),
    )


def decode_q4_compatible_envelope_v1(
    payload: bytes,
    *,
    preimage: Q6SignaturePreimageV1,
    q5b_artifact: SealedQ5bTrustAmendmentArtifactV1,
) -> Q4CompatibleFinalCertificateEnvelopeV1:
    """Replay envelope shape, body, epoch, active set, and key order only."""

    value = _decode(payload, "Q4-compatible Q6 envelope")
    if (
        type(value) is not tuple
        or len(value) != 7
        or value[:2] != (1, QUOTIENT_CERTIFICATE_KIND_ID)
        or not _exact_int(value[2], preimage.created_at_unix_seconds)
        or not _exact_int(value[3], INITIAL_KEY_EPOCH)
        or type(value[4]) is not tuple
        or type(value[6]) is not tuple
        or len(value[6]) != REQUIRED_SIGNATURE_COUNT
    ):
        _fail("REJECT_Q6_ENVELOPE", "Q4-compatible envelope wire differs")
    if value[4] != preimage.q4_body.canonical_object or value[5] != preimage.q4_body.body_hash:
        _fail("REJECT_Q6_ENVELOPE", "Q4-compatible body binding differs")
    try:
        records = tuple(SignatureRecordV1(row[0], row[1]) for row in value[6])
    except (IndexError, TypeError, Q6FormalIssuanceWireError) as error:
        _fail("REJECT_Q6_ENVELOPE", f"signature record wire differs: {error}")
    envelope = Q4CompatibleFinalCertificateEnvelopeV1(value[2], preimage.q4_body, records)
    active = {item.key_id: item.public_key for item in q5b_artifact.signer_manifests}
    if tuple(item.key_id for item in records) != tuple(sorted(active)):
        _fail("REJECT_Q6_ENVELOPE", "signature key set differs from active set")
    if envelope.canonical_bytes != payload:
        _fail("REJECT_Q6_ENVELOPE", "Q4-compatible envelope replay differs")
    return envelope

@dataclass(frozen=True, slots=True)
class Q6FormalIssuanceReceiptV1:
    q5b_capsule_digest: bytes
    q5b_authorization_digest: bytes
    q5b_consumption_record_digest: bytes
    q5b_terminal_record_digest: bytes
    q5b_artifact_raw_sha256: bytes
    q5b_artifact: SealedQ5bTrustAmendmentArtifactV1
    q4_artifact_raw_sha256: bytes
    q4_artifact: SealedQ4CapabilityArtifactV1
    signature_preimage: Q6SignaturePreimageV1
    envelope: Q4CompatibleFinalCertificateEnvelopeV1
    q6_source_commit_raw: bytes
    q6_source_closure_root: bytes
    complete_checkpoint_digest: bytes

    def __post_init__(self) -> None:
        for name in (
            "q5b_capsule_digest",
            "q5b_authorization_digest",
            "q5b_consumption_record_digest",
            "q5b_terminal_record_digest",
            "q5b_artifact_raw_sha256",
            "q4_artifact_raw_sha256",
            "q6_source_closure_root",
            "complete_checkpoint_digest",
        ):
            _bytes32(getattr(self, name), name)
        _bytes(self.q6_source_commit_raw, 20, "Q6 source commit")
        if (
            type(self.q5b_artifact) is not SealedQ5bTrustAmendmentArtifactV1
            or type(self.q4_artifact) is not SealedQ4CapabilityArtifactV1
            or type(self.signature_preimage) is not Q6SignaturePreimageV1
            or type(self.envelope) is not Q4CompatibleFinalCertificateEnvelopeV1
            or sha256(self.q5b_artifact.canonical_bytes).digest()
            != self.q5b_artifact_raw_sha256
            or sha256(self.q4_artifact.canonical_bytes).digest()
            != self.q4_artifact_raw_sha256
            or self.signature_preimage.q5b_receipt_root != self.q5b_artifact.receipt_root
            or self.signature_preimage.q4_capability_receipt_root
            != self.q4_artifact.receipt_root
            or self.envelope.body != self.q4_artifact.body
            or self.envelope.created_at_unix_seconds
            != self.signature_preimage.created_at_unix_seconds
        ):
            _fail("REJECT_Q6_RECEIPT", "issuance receipt object graph differs")

    def canonical_object(self) -> tuple[object, ...]:
        role_ids = self.q5b_artifact.role_order_key_ids
        envelope_ids = tuple(item.key_id for item in self.envelope.signatures)
        return (
            1,
            Q6_RECEIPT_TAG,
            Q6_RECEIPT_SCHEMA_ID,
            self.q5b_capsule_digest,
            self.q5b_authorization_digest,
            self.q5b_consumption_record_digest,
            self.q5b_terminal_record_digest,
            self.q5b_artifact_raw_sha256,
            self.q5b_artifact.canonical_object,
            self.q5b_artifact.receipt_root,
            self.q4_artifact_raw_sha256,
            self.q4_artifact.canonical_object,
            self.q4_artifact.receipt_root,
            self.q4_artifact.body.body_hash,
            self.signature_preimage.canonical_object(),
            self.signature_preimage.preimage_root,
            self.envelope.canonical_object(),
            (1, self.q6_source_commit_raw),
            self.q6_source_closure_root,
            self.complete_checkpoint_digest,
            role_ids,
            envelope_ids,
            REQUIRED_SIGNATURE_COUNT,
            REQUIRED_SIGNATURE_COUNT,
            Q6_ACTION_ID.encode("ascii"),
            Q6_SUCCESS_STATUS.encode("ascii"),
            Q6_CLAIM_LEVEL.encode("ascii"),
            True,
            True,
            True,
            False,
            False,
            False,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def receipt_root(self) -> bytes:
        return content_hash(Q6_RECEIPT_ROOT_DOMAIN, self.canonical_object())

    @property
    def formal_certificate_signed(self) -> bool:
        return True

    @property
    def certificate_envelope_published(self) -> bool:
        return True

    @property
    def outside_certificate_issued(self) -> bool:
        return True

    @property
    def external_authority(self) -> bool:
        return False

    @property
    def global_active_transition_performed(self) -> bool:
        return False

    @property
    def synthesis_released(self) -> bool:
        return False


def build_q6_formal_issuance_receipt_v1(
    q4_artifact_payload: bytes,
    q5b_artifact_payload: bytes,
    *,
    q5b_capsule_digest: bytes,
    q5b_authorization_digest: bytes,
    q5b_consumption_record_digest: bytes,
    q5b_terminal_record_digest: bytes,
    created_at_unix_seconds: int,
    role_order_signatures: tuple[bytes, bytes, bytes],
    q6_source_commit: str,
    q6_source_closure_root: bytes,
    complete_checkpoint_digest: bytes,
) -> Q6FormalIssuanceReceiptV1:
    """Build from three signatures already covered by durable checkpoints.

    No signature is verified here or by the round-trip decoder.  That keeps
    partial finalization structural and prevents a second cryptographic action.
    """

    if (
        type(q6_source_commit) is not str
        or len(q6_source_commit) != 40
        or any(character not in "0123456789abcdef" for character in q6_source_commit)
    ):
        _fail("REJECT_Q6_REPOSITORY_COMMIT", "Q6 source commit must be 40 lower hex")
    q4_artifact = decode_sealed_q4_capability_artifact_v1(q4_artifact_payload)
    q5b_artifact = decode_sealed_q5b_trust_amendment_artifact_v1(q5b_artifact_payload)
    preimage = build_q6_signature_preimage_v1(
        q4_artifact,
        q5b_artifact,
        created_at_unix_seconds=created_at_unix_seconds,
    )
    envelope = build_q4_compatible_envelope_v1(
        preimage,
        q5b_artifact,
        role_order_signatures=role_order_signatures,
    )
    receipt = Q6FormalIssuanceReceiptV1(
        q5b_capsule_digest,
        q5b_authorization_digest,
        q5b_consumption_record_digest,
        q5b_terminal_record_digest,
        sha256(q5b_artifact_payload).digest(),
        q5b_artifact,
        sha256(q4_artifact_payload).digest(),
        q4_artifact,
        preimage,
        envelope,
        bytes.fromhex(q6_source_commit),
        q6_source_closure_root,
        complete_checkpoint_digest,
    )
    if decode_q6_formal_issuance_receipt_v1(receipt.canonical_bytes) != receipt:
        _fail("REJECT_Q6_RECEIPT", "issuance receipt round trip differs")
    return receipt


def decode_q6_formal_issuance_receipt_v1(payload: bytes) -> Q6FormalIssuanceReceiptV1:
    """Strictly replay a sealed issuance artifact without re-verifying crypto."""

    value = _decode(payload, "Q6 formal issuance receipt")
    fixed_tail = (
        REQUIRED_SIGNATURE_COUNT,
        REQUIRED_SIGNATURE_COUNT,
        Q6_ACTION_ID.encode("ascii"),
        Q6_SUCCESS_STATUS.encode("ascii"),
        Q6_CLAIM_LEVEL.encode("ascii"),
        True,
        True,
        True,
        False,
        False,
        False,
    )
    if (
        type(value) is not tuple
        or len(value) != 33
        or value[:3] != (1, Q6_RECEIPT_TAG, Q6_RECEIPT_SCHEMA_ID)
        or any(type(value[index]) is not tuple for index in (8, 11, 14, 16, 17, 20, 21))
        or value[22:] != fixed_tail
    ):
        _fail("REJECT_Q6_RECEIPT", "issuance receipt wire differs")
    for index in (3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 18, 19):
        _bytes32(value[index], f"Q6 receipt field {index}")
    q6_commit = _repository_commit(value[17], "Q6 source commit")
    q5b_payload = canonical_cbor_encode(value[8])
    q4_payload = canonical_cbor_encode(value[11])
    q5b_artifact = decode_sealed_q5b_trust_amendment_artifact_v1(q5b_payload)
    q4_artifact = decode_sealed_q4_capability_artifact_v1(q4_payload)
    if (
        sha256(q5b_payload).digest() != value[7]
        or q5b_artifact.receipt_root != value[9]
        or sha256(q4_payload).digest() != value[10]
        or q4_artifact.receipt_root != value[12]
        or q4_artifact.body.body_hash != value[13]
    ):
        _fail("REJECT_Q6_RECEIPT", "sealed predecessor identities differ")
    preimage = decode_q6_signature_preimage_v1(
        canonical_cbor_encode(value[14]),
        q4_artifact=q4_artifact,
        q5b_artifact=q5b_artifact,
    )
    if preimage.preimage_root != value[15]:
        _fail("REJECT_Q6_RECEIPT", "signature preimage root differs")
    envelope = decode_q4_compatible_envelope_v1(
        canonical_cbor_encode(value[16]),
        preimage=preimage,
        q5b_artifact=q5b_artifact,
    )
    if (
        value[20] != q5b_artifact.role_order_key_ids
        or value[21] != tuple(item.key_id for item in envelope.signatures)
    ):
        _fail("REJECT_Q6_RECEIPT", "role/envelope key order differs")
    receipt = Q6FormalIssuanceReceiptV1(
        value[3],
        value[4],
        value[5],
        value[6],
        value[7],
        q5b_artifact,
        value[10],
        q4_artifact,
        preimage,
        envelope,
        q6_commit,
        value[18],
        value[19],
    )
    if receipt.canonical_bytes != payload:
        _fail("REJECT_Q6_RECEIPT", "issuance receipt replay differs")
    return receipt


__all__ = [
    "INITIAL_KEY_EPOCH",
    "Q4CompatibleFinalCertificateEnvelopeV1",
    "Q6FormalIssuanceReceiptV1",
    "Q6FormalIssuanceWireError",
    "Q6SignaturePreimageV1",
    "Q6_ACTION_ID",
    "Q6_CLAIM_LEVEL",
    "Q6_PREIMAGE_ROOT_DOMAIN",
    "Q6_PREIMAGE_SCHEMA_ID",
    "Q6_PREIMAGE_TAG",
    "Q6_RECEIPT_ROOT_DOMAIN",
    "Q6_RECEIPT_SCHEMA_ID",
    "Q6_RECEIPT_TAG",
    "Q6_SIGNATURE_MESSAGE_PREFIX",
    "Q6_SIGNATURE_PREIMAGE_POLICY_ID",
    "Q6_SUCCESS_STATUS",
    "QUOTIENT_CERTIFICATE_KIND_ID",
    "REQUIRED_SIGNATURE_COUNT",
    "SIGNER_ORDINALS",
    "SIGNER_ROLE_NAMES",
    "SealedCurrentKeyStatusV1",
    "SealedPinnedTrustGenesisV1",
    "SealedPurpose5SignerManifestV1",
    "SealedQ4CapabilityArtifactV1",
    "SealedQ4QuotientCertificateBodyV1",
    "SealedQ5bTrustAmendmentArtifactV1",
    "SignatureRecordV1",
    "build_q4_compatible_envelope_v1",
    "build_q6_formal_issuance_receipt_v1",
    "build_q6_signature_preimage_v1",
    "decode_q4_compatible_envelope_v1",
    "decode_q6_formal_issuance_receipt_v1",
    "decode_q6_signature_preimage_v1",
    "decode_sealed_q4_capability_artifact_v1",
    "decode_sealed_q4_body_v1",
    "decode_sealed_q5b_trust_amendment_artifact_v1",
    "verify_ed25519_v1",
]
