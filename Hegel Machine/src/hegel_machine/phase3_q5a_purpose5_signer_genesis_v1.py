"""Strict public-only wire for the Phase-3A-Q5a Purpose-5 candidate.

Q5a admits three fixed public Ed25519 identities into an *unpinned* Purpose-5
genesis candidate.  It does not generate, accept, read, or use a private key;
it does not create a signature or a certificate envelope.  In particular, the
Q4 body-only ``signature_message`` is not an issuance gate.  A later Q6 stage
must define and separately authorize the status-bound signature preimage
identified here.

All formal objects are strict numeric-array canonical CBOR.  There are no map
or text-key compatibility forms.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Final, NoReturn

from .strict_cbor_v1 import (
    canonical_cbor_decode,
    canonical_cbor_encode,
    content_hash,
    rfc6962_root,
)


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
Q4_MISSING_ISSUANCE_PREREQUISITES: Final = (
    b"PURPOSE5_TRUST_GENESIS_PIN",
    b"PURPOSE5_CURRENT_KEY_STATUS_BUNDLE",
    b"PURPOSE5_THREE_ACTIVE_SIGNERS",
    b"THREE_OF_THREE_SIGNATURES",
)
Q4_SUCCESS_STATUS: Final = (
    b"Q4_FORMAL_QUOTIENT_CERTIFICATE_CAPABILITY_ADMITTED_NOT_ISSUED"
)

ACTION_ID: Final = "Q5A_PURPOSE5_SIGNER_GENESIS_CANDIDATE_V1"
Q5A_SUCCESS_STATUS: Final = (
    "Q5A_PURPOSE5_SIGNER_GENESIS_CANDIDATE_COMPLETE_NOT_PINNED_NOT_SIGNED"
)
Q5A_CLAIM_LEVEL: Final = "LOCAL_DISCLOSED_DEVELOPMENT_CONTROL"

PURPOSE5_FINAL_CERTIFICATE_SIGNER: Final = 5
SIGNER_ROLE_CUSTODIAN: Final = 1
SIGNER_ROLE_PYTHON_REPLAY: Final = 2
SIGNER_ROLE_RUST_REPLAY: Final = 3
SIGNER_ORDINALS: Final = (
    SIGNER_ROLE_CUSTODIAN,
    SIGNER_ROLE_PYTHON_REPLAY,
    SIGNER_ROLE_RUST_REPLAY,
)
SIGNER_ROLE_NAMES: Final = (
    "FINAL_CERTIFICATE_CUSTODIAN",
    "FINAL_CERTIFICATE_PYTHON_REPLAY_ATTESTER",
    "FINAL_CERTIFICATE_RUST_REPLAY_ATTESTER",
)
INITIAL_KEY_EPOCH: Final = 1
CANDIDATE_SIGNER_COUNT: Final = 3
REQUIRED_CERTIFICATE_SIGNATURE_COUNT: Final = 3

# Q6 must bind the pinned genesis/current status in addition to the Q4 body.
# Merely signing Q4's body-only message cannot satisfy this policy.
Q6_SIGNATURE_PREIMAGE_POLICY_ID: Final = (
    b"hegel-q6-purpose5-pinned-status-bound-signature-preimage/1"
)
Q4_BODY_ONLY_SIGNATURE_MESSAGE_FORBIDDEN_AS_ISSUANCE_GATE: Final = True

GIT_SHA1_ALGORITHM_ID: Final = 1


class Q5aPurpose5WireError(ValueError):
    """One Q4/Q5a public wire object differs from the frozen contract."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q5aPurpose5WireError(code, detail)


def _bytes(value: object, length: int, label: str) -> bytes:
    if type(value) is not bytes or len(value) != length:
        _fail("REJECT_Q5A_BYTES", f"{label} must be {length} bytes")
    return value


def _bytes32(value: object, label: str) -> bytes:
    return _bytes(value, 32, label)


def _is_exact_int(value: object, expected: int) -> bool:
    return type(value) is int and value == expected


def _is_exact_bool(value: object, expected: bool) -> bool:
    return type(value) is bool and value is expected


def _decode(payload: bytes, label: str) -> object:
    if type(payload) is not bytes:
        raise TypeError(f"{label} payload must be exact bytes")
    try:
        return canonical_cbor_decode(payload)
    except ValueError as error:
        _fail("REJECT_Q5A_CBOR", f"{label}: {error}")


@dataclass(frozen=True, slots=True)
class RepositoryCommitIdV1:
    algorithm_id: int
    raw_digest: bytes

    def __post_init__(self) -> None:
        if not _is_exact_int(self.algorithm_id, GIT_SHA1_ALGORITHM_ID):
            _fail("REJECT_Q5A_REPOSITORY_COMMIT", "only Git SHA-1 is frozen")
        _bytes(self.raw_digest, 20, "Git SHA-1 commit")

    @classmethod
    def from_hex(cls, value: object) -> "RepositoryCommitIdV1":
        if (
            type(value) is not str
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            _fail(
                "REJECT_Q5A_REPOSITORY_COMMIT",
                "commit must be 40 lowercase hex characters",
            )
        return cls(GIT_SHA1_ALGORITHM_ID, bytes.fromhex(value))

    def canonical_object(self) -> tuple[object, ...]:
        return (self.algorithm_id, self.raw_digest)


def _decode_repository_commit(value: object) -> RepositoryCommitIdV1:
    if type(value) is not tuple or len(value) != 2:
        _fail("REJECT_Q5A_REPOSITORY_COMMIT", "commit wire differs")
    return RepositoryCommitIdV1(value[0], value[1])


@dataclass(frozen=True, slots=True)
class SealedQ4CapabilityReceiptV1:
    """Producer-free projection of the exact sealed Q4 public artifact."""

    canonical_object: tuple[object, ...]
    body_canonical_object: tuple[object, ...]
    body_hash: bytes
    q4_repository_commit_id: RepositoryCommitIdV1
    q4_source_closure_root: bytes

    def __post_init__(self) -> None:
        if (
            type(self.canonical_object) is not tuple
            or type(self.body_canonical_object) is not tuple
            or type(self.q4_repository_commit_id) is not RepositoryCommitIdV1
            or len(self.canonical_object) != 20
            or len(self.body_canonical_object) != 34
            or self.canonical_object[3] != self.body_canonical_object
            or self.canonical_object[4] != self.body_hash
            or self.body_canonical_object[31]
            != self.q4_repository_commit_id.canonical_object()
            or self.body_canonical_object[32] != self.q4_source_closure_root
        ):
            _fail("REJECT_Q4_CAPABILITY", "sealed Q4 projection type differs")
        _bytes32(self.body_hash, "Q4 body hash")
        _bytes32(self.q4_source_closure_root, "Q4 source closure root")
        if self.body_hash != content_hash(
            Q4_BODY_ROOT_DOMAIN, self.body_canonical_object
        ):
            _fail("REJECT_Q4_CAPABILITY", "Q4 projected body hash differs")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object)

    @property
    def receipt_root(self) -> bytes:
        return content_hash(Q4_RECEIPT_ROOT_DOMAIN, self.canonical_object)


def decode_sealed_q4_capability_receipt_v1(
    payload: bytes,
) -> SealedQ4CapabilityReceiptV1:
    """Replay Q4 without importing its producer or accepting an envelope."""

    value = _decode(payload, "Q4 capability receipt")
    q4_tail = (
        PURPOSE5_FINAL_CERTIFICATE_SIGNER,
        Q4_MISSING_ISSUANCE_PREREQUISITES,
        0,
        CANDIDATE_SIGNER_COUNT,
        0,
        REQUIRED_CERTIFICATE_SIGNATURE_COUNT,
        b"NOT_READY_FOR_ISSUANCE",
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
        or not _is_exact_int(value[0], 1)
        or not _is_exact_int(value[1], Q4_RECEIPT_TAG)
        or type(value[3]) is not tuple
        or value[6:] != q4_tail
        or any(type(value[index]) is not int for index in (6, 8, 9, 10, 11))
        or any(type(value[index]) is not bool for index in range(14, 20))
    ):
        _fail("REJECT_Q4_CAPABILITY", "Q4 capability receipt wire differs")
    body = value[3]
    if (
        len(body) != 34
        or body[:7]
        != (
            1,
            Q4_BODY_TAG,
            Q4_BODY_SCHEMA_ID,
            b"OUTSIDE_FROZEN_QUOTIENT_CLOSURE",
            b"hegel-old-dsl-v1.6.0",
            b"hegel-quotient-closure-v1.0.1",
            1,
        )
        or any(type(body[index]) is not int for index in (0, 1, 6, 20, 21))
        or body[9] != b"exact_behavior_vector_with_bottom"
        or body[20:23] != (418, 0, rfc6962_root(()))
        or type(body[23]) is not tuple
        or len(body[23]) != 7
        or body[23][0] != 2
        or body[23][4:6] != (89, 1)
        or any(type(body[23][index]) is not int for index in (0, 4, 5))
        or type(body[23][6]) is not tuple
        or body[33] != b"LOCAL_DISCLOSED_DEVELOPMENT_CONTROL"
    ):
        _fail("REJECT_Q4_CAPABILITY", "Q4 body wire differs")
    for index in (
        7,
        8,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        24,
        25,
        26,
        27,
        28,
        30,
        32,
    ):
        _bytes32(body[index], f"Q4 body field {index}")
    for index in (1, 2, 3):
        _bytes32(body[23][index], f"Q4 null row field {index}")
    if len(body[23][6]) != 1:
        _fail("REJECT_Q4_CAPABILITY", "Q4 null matching class count differs")
    _bytes32(body[23][6][0], "Q4 null matching class id")
    q4_commit = _decode_repository_commit(body[31])
    _decode_repository_commit(body[29])
    body_hash = content_hash(Q4_BODY_ROOT_DOMAIN, body)
    if value[4] != body_hash:
        _fail("REJECT_Q4_CAPABILITY", "Q4 receipt body hash differs")
    _bytes32(value[5], "Q4 candidate root")
    return SealedQ4CapabilityReceiptV1(
        canonical_object=value,
        body_canonical_object=body,
        body_hash=body_hash,
        q4_repository_commit_id=q4_commit,
        q4_source_closure_root=body[32],
    )


@dataclass(frozen=True, slots=True)
class Purpose5SignerManifestV1:
    signer_ordinal: int
    key_id: bytes
    public_key: bytes

    def __post_init__(self) -> None:
        if type(self.signer_ordinal) is not int or self.signer_ordinal not in SIGNER_ORDINALS:
            _fail("REJECT_Q5A_SIGNER", "signer ordinal differs")
        _bytes(self.key_id, 16, "Purpose-5 key id")
        _bytes32(self.public_key, "Purpose-5 public key")
        if self.key_id != sha256(self.public_key).digest()[:16]:
            _fail("REJECT_Q5A_SIGNER", "key id does not derive from public key")

    @classmethod
    def from_public_key(
        cls, signer_ordinal: int, public_key: bytes
    ) -> "Purpose5SignerManifestV1":
        _bytes32(public_key, "Purpose-5 public key")
        return cls(
            signer_ordinal=signer_ordinal,
            key_id=sha256(public_key).digest()[:16],
            public_key=public_key,
        )

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
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def manifest_root(self) -> bytes:
        return content_hash(Q5A_SIGNER_MANIFEST_ROOT_DOMAIN, self.canonical_object())


def decode_purpose5_signer_manifest_v1(payload: bytes) -> Purpose5SignerManifestV1:
    value = _decode(payload, "Purpose-5 signer manifest")
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
        or not _is_exact_int(value[0], 1)
        or not _is_exact_int(value[1], Q5A_SIGNER_MANIFEST_TAG)
        or not _is_exact_int(value[3], PURPOSE5_FINAL_CERTIFICATE_SIGNER)
        or type(value[4]) is not int
        or value[4] not in SIGNER_ORDINALS
        or value[5] != SIGNER_ROLE_NAMES[value[4] - 1].encode("ascii")
        or not _is_exact_int(value[6], INITIAL_KEY_EPOCH)
    ):
        _fail("REJECT_Q5A_SIGNER", "signer manifest wire differs")
    manifest = Purpose5SignerManifestV1(value[4], value[7], value[8])
    if manifest.canonical_bytes != payload:
        _fail("REJECT_Q5A_SIGNER", "signer manifest replay differs")
    return manifest


@dataclass(frozen=True, slots=True)
class UnpinnedTrustGenesisCandidateV1:
    signer_manifests: tuple[Purpose5SignerManifestV1, ...]

    def __post_init__(self) -> None:
        if (
            type(self.signer_manifests) is not tuple
            or len(self.signer_manifests) != CANDIDATE_SIGNER_COUNT
            or any(
                type(manifest) is not Purpose5SignerManifestV1
                for manifest in self.signer_manifests
            )
            or tuple(manifest.signer_ordinal for manifest in self.signer_manifests)
            != SIGNER_ORDINALS
            or len({manifest.key_id for manifest in self.signer_manifests})
            != CANDIDATE_SIGNER_COUNT
            or len({manifest.public_key for manifest in self.signer_manifests})
            != CANDIDATE_SIGNER_COUNT
        ):
            _fail(
                "REJECT_Q5A_GENESIS",
                "genesis requires three ordered distinct Purpose-5 signers",
            )

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q5A_TRUST_GENESIS_TAG,
            Q5A_TRUST_GENESIS_SCHEMA_ID,
            PURPOSE5_FINAL_CERTIFICATE_SIGNER,
            INITIAL_KEY_EPOCH,
            tuple(manifest.canonical_object() for manifest in self.signer_manifests),
            tuple(manifest.manifest_root for manifest in self.signer_manifests),
            CANDIDATE_SIGNER_COUNT,
            REQUIRED_CERTIFICATE_SIGNATURE_COUNT,
            Q6_SIGNATURE_PREIMAGE_POLICY_ID,
            Q4_BODY_ONLY_SIGNATURE_MESSAGE_FORBIDDEN_AS_ISSUANCE_GATE,
            False,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def genesis_root(self) -> bytes:
        return content_hash(Q5A_TRUST_GENESIS_ROOT_DOMAIN, self.canonical_object())


def decode_unpinned_trust_genesis_candidate_v1(
    payload: bytes,
) -> UnpinnedTrustGenesisCandidateV1:
    value = _decode(payload, "Purpose-5 trust genesis candidate")
    fixed_tail = (
        CANDIDATE_SIGNER_COUNT,
        REQUIRED_CERTIFICATE_SIGNATURE_COUNT,
        Q6_SIGNATURE_PREIMAGE_POLICY_ID,
        Q4_BODY_ONLY_SIGNATURE_MESSAGE_FORBIDDEN_AS_ISSUANCE_GATE,
        False,
    )
    if (
        type(value) is not tuple
        or len(value) != 12
        or value[:5]
        != (
            1,
            Q5A_TRUST_GENESIS_TAG,
            Q5A_TRUST_GENESIS_SCHEMA_ID,
            PURPOSE5_FINAL_CERTIFICATE_SIGNER,
            INITIAL_KEY_EPOCH,
        )
        or not _is_exact_int(value[0], 1)
        or not _is_exact_int(value[1], Q5A_TRUST_GENESIS_TAG)
        or not _is_exact_int(value[3], PURPOSE5_FINAL_CERTIFICATE_SIGNER)
        or not _is_exact_int(value[4], INITIAL_KEY_EPOCH)
        or type(value[5]) is not tuple
        or type(value[6]) is not tuple
        or value[7:] != fixed_tail
        or not _is_exact_int(value[7], CANDIDATE_SIGNER_COUNT)
        or not _is_exact_int(value[8], REQUIRED_CERTIFICATE_SIGNATURE_COUNT)
        or not _is_exact_bool(value[10], True)
        or not _is_exact_bool(value[11], False)
    ):
        _fail("REJECT_Q5A_GENESIS", "trust genesis candidate wire differs")
    manifests = tuple(
        decode_purpose5_signer_manifest_v1(canonical_cbor_encode(item))
        for item in value[5]
    )
    genesis = UnpinnedTrustGenesisCandidateV1(manifests)
    if value[6] != tuple(manifest.manifest_root for manifest in manifests):
        _fail("REJECT_Q5A_GENESIS", "signer manifest roots differ")
    if genesis.canonical_bytes != payload:
        _fail("REJECT_Q5A_GENESIS", "trust genesis replay differs")
    return genesis


@dataclass(frozen=True, slots=True)
class GenesisOnlyStatusCandidateV1:
    trust_genesis: UnpinnedTrustGenesisCandidateV1

    def __post_init__(self) -> None:
        if type(self.trust_genesis) is not UnpinnedTrustGenesisCandidateV1:
            _fail("REJECT_Q5A_STATUS", "trust genesis type differs")

    @property
    def signer_manifests(self) -> tuple[Purpose5SignerManifestV1, ...]:
        return self.trust_genesis.signer_manifests

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q5A_KEY_STATUS_TAG,
            Q5A_KEY_STATUS_SCHEMA_ID,
            self.trust_genesis.canonical_object(),
            self.trust_genesis.genesis_root,
            INITIAL_KEY_EPOCH,
            tuple(
                manifest.manifest_root for manifest in self.trust_genesis.signer_manifests
            ),
            (),
            (),
            Q6_SIGNATURE_PREIMAGE_POLICY_ID,
            False,
            0,
            0,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def status_root(self) -> bytes:
        return content_hash(Q5A_KEY_STATUS_ROOT_DOMAIN, self.canonical_object())


def decode_genesis_only_status_candidate_v1(
    payload: bytes,
) -> GenesisOnlyStatusCandidateV1:
    value = _decode(payload, "Purpose-5 genesis-only status candidate")
    if (
        type(value) is not tuple
        or len(value) != 13
        or value[:3] != (1, Q5A_KEY_STATUS_TAG, Q5A_KEY_STATUS_SCHEMA_ID)
        or not _is_exact_int(value[0], 1)
        or not _is_exact_int(value[1], Q5A_KEY_STATUS_TAG)
        or type(value[3]) is not tuple
        or not _is_exact_int(value[5], INITIAL_KEY_EPOCH)
        or type(value[6]) is not tuple
        or value[7:]
        != (
            (),
            (),
            Q6_SIGNATURE_PREIMAGE_POLICY_ID,
            False,
            0,
            0,
        )
        or not _is_exact_bool(value[10], False)
        or not _is_exact_int(value[11], 0)
        or not _is_exact_int(value[12], 0)
    ):
        _fail("REJECT_Q5A_STATUS", "genesis-only status candidate wire differs")
    genesis = decode_unpinned_trust_genesis_candidate_v1(
        canonical_cbor_encode(value[3])
    )
    if (
        value[4] != genesis.genesis_root
        or value[6]
        != tuple(manifest.manifest_root for manifest in genesis.signer_manifests)
    ):
        _fail("REJECT_Q5A_STATUS", "status/genesis signer roots differ")
    status = GenesisOnlyStatusCandidateV1(genesis)
    if status.canonical_bytes != payload:
        _fail("REJECT_Q5A_STATUS", "genesis-only status replay differs")
    return status


@dataclass(frozen=True, slots=True)
class Q5aPurpose5SignerGenesisCandidateV1:
    q4_capsule_digest: bytes
    q4_authorization_digest: bytes
    q4_consumption_record_digest: bytes
    q4_terminal_record_digest: bytes
    q4_capability_receipt_root: bytes
    q4_body_hash: bytes
    q4_repository_commit_id: RepositoryCommitIdV1
    q4_source_closure_root: bytes
    q5_repository_commit_id: RepositoryCommitIdV1
    q5_source_closure_root: bytes
    signer_manifests: tuple[Purpose5SignerManifestV1, ...]
    trust_genesis: UnpinnedTrustGenesisCandidateV1
    key_status: GenesisOnlyStatusCandidateV1

    def __post_init__(self) -> None:
        for name in (
            "q4_capsule_digest",
            "q4_authorization_digest",
            "q4_consumption_record_digest",
            "q4_terminal_record_digest",
            "q4_capability_receipt_root",
            "q4_body_hash",
            "q4_source_closure_root",
            "q5_source_closure_root",
        ):
            _bytes32(getattr(self, name), name)
        if (
            type(self.q4_repository_commit_id) is not RepositoryCommitIdV1
            or type(self.q5_repository_commit_id) is not RepositoryCommitIdV1
            or type(self.signer_manifests) is not tuple
            or type(self.trust_genesis) is not UnpinnedTrustGenesisCandidateV1
            or type(self.key_status) is not GenesisOnlyStatusCandidateV1
            or self.trust_genesis.signer_manifests != self.signer_manifests
            or self.key_status.trust_genesis != self.trust_genesis
        ):
            _fail("REJECT_Q5A_CANDIDATE", "candidate object graph differs")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q5A_CANDIDATE_TAG,
            Q5A_CANDIDATE_SCHEMA_ID,
            self.q4_capsule_digest,
            self.q4_authorization_digest,
            self.q4_consumption_record_digest,
            self.q4_terminal_record_digest,
            self.q4_capability_receipt_root,
            self.q4_body_hash,
            self.q4_repository_commit_id.canonical_object(),
            self.q4_source_closure_root,
            self.q5_repository_commit_id.canonical_object(),
            self.q5_source_closure_root,
            tuple(manifest.canonical_object() for manifest in self.signer_manifests),
            self.trust_genesis.canonical_object(),
            self.trust_genesis.genesis_root,
            self.key_status.canonical_object(),
            self.key_status.status_root,
            Q6_SIGNATURE_PREIMAGE_POLICY_ID,
            Q4_BODY_ONLY_SIGNATURE_MESSAGE_FORBIDDEN_AS_ISSUANCE_GATE,
            CANDIDATE_SIGNER_COUNT,
            0,
            0,
            ACTION_ID.encode("ascii"),
            Q5A_SUCCESS_STATUS.encode("ascii"),
            Q5A_CLAIM_LEVEL.encode("ascii"),
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        )

    @property
    def q4_source_commit(self) -> str:
        return self.q4_repository_commit_id.raw_digest.hex()

    @property
    def q5_source_commit(self) -> str:
        return self.q5_repository_commit_id.raw_digest.hex()

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def candidate_root(self) -> bytes:
        return content_hash(Q5A_CANDIDATE_ROOT_DOMAIN, self.canonical_object())


def build_q5a_candidate_v1(
    *,
    q4_capsule_digest: bytes,
    q4_authorization_digest: bytes,
    q4_consumption_record_digest: bytes,
    q4_terminal_record_digest: bytes,
    q4_capability_receipt_root: bytes,
    q4_body_hash: bytes,
    q4_source_commit: str,
    q4_source_closure_root: bytes,
    q5_source_commit: str,
    q5_source_closure_root: bytes,
    public_keys: tuple[bytes, bytes, bytes],
) -> Q5aPurpose5SignerGenesisCandidateV1:
    if type(public_keys) is not tuple or len(public_keys) != CANDIDATE_SIGNER_COUNT:
        _fail("REJECT_Q5A_SIGNER", "exactly three public keys are required")
    manifests = tuple(
        Purpose5SignerManifestV1.from_public_key(ordinal, public_key)
        for ordinal, public_key in zip(SIGNER_ORDINALS, public_keys, strict=True)
    )
    genesis = UnpinnedTrustGenesisCandidateV1(manifests)
    status = GenesisOnlyStatusCandidateV1(genesis)
    candidate = Q5aPurpose5SignerGenesisCandidateV1(
        q4_capsule_digest=q4_capsule_digest,
        q4_authorization_digest=q4_authorization_digest,
        q4_consumption_record_digest=q4_consumption_record_digest,
        q4_terminal_record_digest=q4_terminal_record_digest,
        q4_capability_receipt_root=q4_capability_receipt_root,
        q4_body_hash=q4_body_hash,
        q4_repository_commit_id=RepositoryCommitIdV1.from_hex(q4_source_commit),
        q4_source_closure_root=q4_source_closure_root,
        q5_repository_commit_id=RepositoryCommitIdV1.from_hex(q5_source_commit),
        q5_source_closure_root=q5_source_closure_root,
        signer_manifests=manifests,
        trust_genesis=genesis,
        key_status=status,
    )
    if decode_q5a_candidate_v1(candidate.canonical_bytes) != candidate:
        _fail("REJECT_Q5A_CANDIDATE", "candidate round trip differs")
    return candidate


def decode_q5a_candidate_v1(
    payload: bytes,
) -> Q5aPurpose5SignerGenesisCandidateV1:
    value = _decode(payload, "Q5a Purpose-5 signer genesis candidate")
    fixed_tail = (
        Q6_SIGNATURE_PREIMAGE_POLICY_ID,
        Q4_BODY_ONLY_SIGNATURE_MESSAGE_FORBIDDEN_AS_ISSUANCE_GATE,
        CANDIDATE_SIGNER_COUNT,
        0,
        0,
        ACTION_ID.encode("ascii"),
        Q5A_SUCCESS_STATUS.encode("ascii"),
        Q5A_CLAIM_LEVEL.encode("ascii"),
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
        or not _is_exact_int(value[0], 1)
        or not _is_exact_int(value[1], Q5A_CANDIDATE_TAG)
        or type(value[9]) is not tuple
        or type(value[11]) is not tuple
        or type(value[13]) is not tuple
        or type(value[14]) is not tuple
        or type(value[16]) is not tuple
        or value[18:] != fixed_tail
        or not _is_exact_bool(value[19], True)
        or not _is_exact_int(value[20], CANDIDATE_SIGNER_COUNT)
        or not _is_exact_int(value[21], 0)
        or not _is_exact_int(value[22], 0)
        or not _is_exact_bool(value[26], True)
        or any(not _is_exact_bool(value[index], False) for index in range(27, 34))
    ):
        _fail("REJECT_Q5A_CANDIDATE", "candidate wire differs")
    for index in (3, 4, 5, 6, 7, 8, 10, 12, 15, 17):
        _bytes32(value[index], f"Q5a candidate field {index}")
    manifests = tuple(
        decode_purpose5_signer_manifest_v1(canonical_cbor_encode(item))
        for item in value[13]
    )
    genesis = decode_unpinned_trust_genesis_candidate_v1(
        canonical_cbor_encode(value[14])
    )
    status = decode_genesis_only_status_candidate_v1(
        canonical_cbor_encode(value[16])
    )
    if (
        value[15] != genesis.genesis_root
        or value[17] != status.status_root
        or genesis.signer_manifests != manifests
        or status.trust_genesis != genesis
    ):
        _fail("REJECT_Q5A_CANDIDATE", "candidate nested roots differ")
    candidate = Q5aPurpose5SignerGenesisCandidateV1(
        q4_capsule_digest=value[3],
        q4_authorization_digest=value[4],
        q4_consumption_record_digest=value[5],
        q4_terminal_record_digest=value[6],
        q4_capability_receipt_root=value[7],
        q4_body_hash=value[8],
        q4_repository_commit_id=_decode_repository_commit(value[9]),
        q4_source_closure_root=value[10],
        q5_repository_commit_id=_decode_repository_commit(value[11]),
        q5_source_closure_root=value[12],
        signer_manifests=manifests,
        trust_genesis=genesis,
        key_status=status,
    )
    if candidate.canonical_bytes != payload:
        _fail("REJECT_Q5A_CANDIDATE", "candidate replay differs")
    return candidate


__all__ = [
    "ACTION_ID",
    "GenesisOnlyStatusCandidateV1",
    "INITIAL_KEY_EPOCH",
    "PURPOSE5_FINAL_CERTIFICATE_SIGNER",
    "Purpose5SignerManifestV1",
    "Q5A_SUCCESS_STATUS",
    "Q5aPurpose5SignerGenesisCandidateV1",
    "Q5aPurpose5WireError",
    "Q6_SIGNATURE_PREIMAGE_POLICY_ID",
    "RepositoryCommitIdV1",
    "SIGNER_ORDINALS",
    "SIGNER_ROLE_NAMES",
    "SIGNER_ROLE_CUSTODIAN",
    "SIGNER_ROLE_PYTHON_REPLAY",
    "SIGNER_ROLE_RUST_REPLAY",
    "SealedQ4CapabilityReceiptV1",
    "UnpinnedTrustGenesisCandidateV1",
    "build_q5a_candidate_v1",
    "decode_genesis_only_status_candidate_v1",
    "decode_purpose5_signer_manifest_v1",
    "decode_q5a_candidate_v1",
    "decode_sealed_q4_capability_receipt_v1",
    "decode_unpinned_trust_genesis_candidate_v1",
]
