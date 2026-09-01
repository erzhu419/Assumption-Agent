"""Public-only wire for the Q5b Purpose-5 trust amendment.

Q5b replays one sealed Q5a candidate and turns its three public Purpose-5
identities into a pinned trust genesis and the epoch-1 current status bundle.
It does not generate or import keys, read private material, sign a message, or
publish a certificate envelope.  Q6 remains the first stage allowed to use the
pinned-status-bound signature preimage frozen here.

The M2.5 ``ActorTrustGenesisV1`` wire is intentionally not reused: that object
binds the pre-M4 purpose set 1--4 and a replacement-policy root.  Purpose 5 has
its own three-role topology and therefore uses distinct tags, schemas, and hash
domains below.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Final, NoReturn

from .strict_cbor_v1 import (
    canonical_cbor_decode,
    canonical_cbor_encode,
    content_hash,
)


# Exact sealed-Q5a projection.  These constants are repeated deliberately so
# Q5b can replay Q5a without importing its producer module.
Q5A_SIGNER_MANIFEST_TAG: Final = 0x3F00
Q5A_SIGNER_MANIFEST_SCHEMA_ID: Final = b"hegel-q5a-purpose5-signer-manifest/1"
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
Q5A_CANDIDATE_SCHEMA_ID: Final = b"hegel-q5a-purpose5-signer-genesis-candidate/1"
Q5A_CANDIDATE_ROOT_DOMAIN: Final = (
    "HEGEL/Q5A/PURPOSE5_SIGNER_GENESIS_CANDIDATE/V1"
)
Q5A_ACTION_ID: Final = b"Q5A_PURPOSE5_SIGNER_GENESIS_CANDIDATE_V1"
Q5A_SUCCESS_STATUS: Final = (
    b"Q5A_PURPOSE5_SIGNER_GENESIS_CANDIDATE_COMPLETE_NOT_PINNED_NOT_SIGNED"
)
Q5A_CLAIM_LEVEL: Final = b"LOCAL_DISCLOSED_DEVELOPMENT_CONTROL"

PURPOSE5_FINAL_CERTIFICATE_SIGNER: Final = 5
INITIAL_KEY_EPOCH: Final = 1
SIGNER_ORDINALS: Final = (1, 2, 3)
SIGNER_ROLE_NAMES: Final = (
    "FINAL_CERTIFICATE_CUSTODIAN",
    "FINAL_CERTIFICATE_PYTHON_REPLAY_ATTESTER",
    "FINAL_CERTIFICATE_RUST_REPLAY_ATTESTER",
)
Q6_SIGNATURE_PREIMAGE_POLICY_ID: Final = (
    b"hegel-q6-purpose5-pinned-status-bound-signature-preimage/1"
)
Q4_BODY_ONLY_SIGNATURE_MESSAGE_FORBIDDEN_AS_ISSUANCE_GATE: Final = True


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

# These two M2.5 identities are recorded only to make the incompatibility
# executable in tests; they are never accepted as Q5b objects.
M25_ACTOR_TRUST_GENESIS_SCHEMA_ID: Final = b"hegel-actor-trust-genesis/1"
M25_ACTOR_TRUST_GENESIS_ROOT_DOMAIN: Final = "HEGEL/ACTOR_TRUST_GENESIS/V1"
M25_PRE_M4_PURPOSE_SET: Final = (1, 2, 3, 4)

ACTION_ID: Final = "Q5B_PURPOSE5_TRUST_GENESIS_CURRENT_STATUS_PIN_AMENDMENT_V1"
Q5B_SUCCESS_STATUS: Final = (
    "Q5B_PURPOSE5_TRUST_AMENDMENT_ADMITTED_CURRENT_NOT_SIGNED_NOT_ISSUED"
)
Q5B_CLAIM_LEVEL: Final = "LOCAL_DISCLOSED_DEVELOPMENT_CONTROL"

ACTIVE_SIGNER_COUNT: Final = 3
FUTURE_STATUS_TRANSITION_THRESHOLD: Final = 2
FUTURE_STATUS_TRANSITION_ELIGIBLE_COUNT: Final = 3
REQUIRED_CERTIFICATE_SIGNATURE_COUNT: Final = 3
GLOBAL_ACTIVE_TRANSITION_PERFORMED: Final = False


class Q5bPurpose5TrustAmendmentError(ValueError):
    """One public Q5a/Q5b object differs from the frozen wire."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Q5bPurpose5TrustAmendmentError(code, detail)


def _bytes(value: object, length: int, label: str) -> bytes:
    if type(value) is not bytes or len(value) != length:
        _fail("REJECT_Q5B_BYTES", f"{label} must be {length} bytes")
    return value


def _bytes32(value: object, label: str) -> bytes:
    return _bytes(value, 32, label)


def _is_exact_int(value: object, expected: int) -> bool:
    return type(value) is int and value == expected


def _is_exact_bool(value: object, expected: bool) -> bool:
    return type(value) is bool and value is expected


def _commit_raw(value: object, label: str) -> bytes:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail("REJECT_Q5B_REPOSITORY_COMMIT", f"{label} must be lowercase Git SHA-1")
    return bytes.fromhex(value)


def _decode(payload: bytes, label: str) -> object:
    if type(payload) is not bytes:
        raise TypeError(f"{label} payload must be exact bytes")
    try:
        return canonical_cbor_decode(payload)
    except ValueError as error:
        _fail("REJECT_Q5B_CBOR", f"{label}: {error}")


@dataclass(frozen=True, slots=True)
class SealedQ5aPurpose5SignerManifestV1:
    signer_ordinal: int
    key_id: bytes
    public_key: bytes

    def __post_init__(self) -> None:
        if type(self.signer_ordinal) is not int or self.signer_ordinal not in SIGNER_ORDINALS:
            _fail("REJECT_Q5B_Q5A_SIGNER", "sealed signer ordinal differs")
        _bytes(self.key_id, 16, "sealed Q5a key id")
        _bytes32(self.public_key, "sealed Q5a public key")
        if self.key_id != sha256(self.public_key).digest()[:16]:
            _fail("REJECT_Q5B_Q5A_SIGNER", "sealed signer key id differs")

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
        return content_hash(
            Q5A_SIGNER_MANIFEST_ROOT_DOMAIN, self.canonical_object()
        )


def _decode_sealed_q5a_signer_manifest(
    value: object,
) -> SealedQ5aPurpose5SignerManifestV1:
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
        _fail("REJECT_Q5B_Q5A_SIGNER", "sealed Q5a signer wire differs")
    manifest = SealedQ5aPurpose5SignerManifestV1(value[4], value[7], value[8])
    if manifest.canonical_object() != value:
        _fail("REJECT_Q5B_Q5A_SIGNER", "sealed Q5a signer replay differs")
    return manifest


@dataclass(frozen=True, slots=True)
class SealedQ5aPurpose5SignerGenesisCandidateV1:
    canonical_object: tuple[object, ...]
    q4_capsule_digest: bytes
    q4_authorization_digest: bytes
    q4_consumption_record_digest: bytes
    q4_terminal_record_digest: bytes
    q4_capability_receipt_root: bytes
    q4_body_hash: bytes
    q4_source_commit_raw: bytes
    q4_source_closure_root: bytes
    q5a_source_commit_raw: bytes
    q5a_source_closure_root: bytes
    signer_manifests: tuple[SealedQ5aPurpose5SignerManifestV1, ...]
    unpinned_genesis_root: bytes
    genesis_only_status_root: bytes

    def __post_init__(self) -> None:
        if type(self.canonical_object) is not tuple or len(self.canonical_object) != 34:
            _fail("REJECT_Q5B_Q5A_ARTIFACT", "sealed candidate shape differs")
        for name in (
            "q4_capsule_digest",
            "q4_authorization_digest",
            "q4_consumption_record_digest",
            "q4_terminal_record_digest",
            "q4_capability_receipt_root",
            "q4_body_hash",
            "q4_source_closure_root",
            "q5a_source_closure_root",
            "unpinned_genesis_root",
            "genesis_only_status_root",
        ):
            _bytes32(getattr(self, name), name)
        _bytes(self.q4_source_commit_raw, 20, "Q4 source commit")
        _bytes(self.q5a_source_commit_raw, 20, "Q5a source commit")
        _validate_manifests(self.signer_manifests)

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object)

    @property
    def candidate_root(self) -> bytes:
        return content_hash(Q5A_CANDIDATE_ROOT_DOMAIN, self.canonical_object)


def decode_sealed_q5a_candidate_v1(
    payload: bytes,
) -> SealedQ5aPurpose5SignerGenesisCandidateV1:
    """Decode the exact public Q5a artifact without importing its producer."""

    value = _decode(payload, "sealed Q5a candidate")
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
        or not _is_exact_int(value[0], 1)
        or not _is_exact_int(value[1], Q5A_CANDIDATE_TAG)
        or any(type(value[index]) is not tuple for index in (9, 11, 13, 14, 16))
        or value[18:] != fixed_tail
        or not _is_exact_bool(value[19], True)
        or not _is_exact_int(value[20], 3)
        or not _is_exact_int(value[21], 0)
        or not _is_exact_int(value[22], 0)
        or not _is_exact_bool(value[26], True)
        or any(not _is_exact_bool(value[index], False) for index in range(27, 34))
    ):
        _fail("REJECT_Q5B_Q5A_ARTIFACT", "sealed Q5a candidate wire differs")
    for index in (3, 4, 5, 6, 7, 8, 10, 12, 15, 17):
        _bytes32(value[index], f"sealed Q5a field {index}")
    for index in (9, 11):
        if (
            len(value[index]) != 2
            or not _is_exact_int(value[index][0], 1)
            or type(value[index][1]) is not bytes
            or len(value[index][1]) != 20
        ):
            _fail("REJECT_Q5B_Q5A_ARTIFACT", "sealed repository commit differs")
    manifests = tuple(_decode_sealed_q5a_signer_manifest(row) for row in value[13])
    _validate_manifests(manifests)
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
        or genesis[7:]
        != (3, 3, Q6_SIGNATURE_PREIMAGE_POLICY_ID, True, False)
        or not _is_exact_int(genesis[7], 3)
        or not _is_exact_int(genesis[8], 3)
        or not _is_exact_bool(genesis[10], True)
        or not _is_exact_bool(genesis[11], False)
    ):
        _fail("REJECT_Q5B_Q5A_ARTIFACT", "sealed Q5a genesis differs")
    genesis_root = content_hash(Q5A_TRUST_GENESIS_ROOT_DOMAIN, genesis)

    status = value[16]
    if (
        len(status) != 13
        or status[:3] != (1, Q5A_KEY_STATUS_TAG, Q5A_KEY_STATUS_SCHEMA_ID)
        or status[3] != genesis
        or status[4] != genesis_root
        or not _is_exact_int(status[5], INITIAL_KEY_EPOCH)
        or status[6] != manifest_roots
        or status[7:]
        != ((), (), Q6_SIGNATURE_PREIMAGE_POLICY_ID, False, 0, 0)
        or not _is_exact_bool(status[10], False)
        or not _is_exact_int(status[11], 0)
        or not _is_exact_int(status[12], 0)
    ):
        _fail("REJECT_Q5B_Q5A_ARTIFACT", "sealed Q5a status differs")
    status_root = content_hash(Q5A_KEY_STATUS_ROOT_DOMAIN, status)
    if (
        value[13] != manifest_rows
        or value[15] != genesis_root
        or value[17] != status_root
    ):
        _fail("REJECT_Q5B_Q5A_ARTIFACT", "sealed Q5a nested roots differ")
    candidate = SealedQ5aPurpose5SignerGenesisCandidateV1(
        canonical_object=value,
        q4_capsule_digest=value[3],
        q4_authorization_digest=value[4],
        q4_consumption_record_digest=value[5],
        q4_terminal_record_digest=value[6],
        q4_capability_receipt_root=value[7],
        q4_body_hash=value[8],
        q4_source_commit_raw=value[9][1],
        q4_source_closure_root=value[10],
        q5a_source_commit_raw=value[11][1],
        q5a_source_closure_root=value[12],
        signer_manifests=manifests,
        unpinned_genesis_root=genesis_root,
        genesis_only_status_root=status_root,
    )
    if candidate.canonical_bytes != payload:
        _fail("REJECT_Q5B_Q5A_ARTIFACT", "sealed Q5a replay differs")
    return candidate


def _manifest_roots(
    manifests: tuple[SealedQ5aPurpose5SignerManifestV1, ...],
) -> tuple[bytes, ...]:
    return tuple(manifest.manifest_root for manifest in manifests)


def _key_ids(
    manifests: tuple[SealedQ5aPurpose5SignerManifestV1, ...],
) -> tuple[bytes, ...]:
    return tuple(manifest.key_id for manifest in manifests)


def _validate_manifests(
    manifests: object,
) -> tuple[SealedQ5aPurpose5SignerManifestV1, ...]:
    if (
        type(manifests) is not tuple
        or len(manifests) != ACTIVE_SIGNER_COUNT
        or any(
            type(item) is not SealedQ5aPurpose5SignerManifestV1
            for item in manifests
        )
        or tuple(item.signer_ordinal for item in manifests) != SIGNER_ORDINALS
        or len(set(_manifest_roots(manifests))) != ACTIVE_SIGNER_COUNT
        or len(set(_key_ids(manifests))) != ACTIVE_SIGNER_COUNT
        or len({item.public_key for item in manifests}) != ACTIVE_SIGNER_COUNT
    ):
        _fail(
            "REJECT_Q5B_SIGNER_SET",
            "Q5b requires the three ordered distinct Q5a Purpose-5 signers",
        )
    return manifests


@dataclass(frozen=True, slots=True)
class PinnedTrustGenesisV1:
    q5a_candidate_root: bytes
    q5a_unpinned_genesis_root: bytes
    signer_manifests: tuple[SealedQ5aPurpose5SignerManifestV1, ...]

    def __post_init__(self) -> None:
        _bytes32(self.q5a_candidate_root, "Q5a candidate root")
        _bytes32(self.q5a_unpinned_genesis_root, "Q5a unpinned genesis root")
        _validate_manifests(self.signer_manifests)

    @property
    def manifest_roots(self) -> tuple[bytes, ...]:
        return _manifest_roots(self.signer_manifests)

    @property
    def key_ids(self) -> tuple[bytes, ...]:
        return _key_ids(self.signer_manifests)

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q5B_PINNED_GENESIS_TAG,
            Q5B_PINNED_GENESIS_SCHEMA_ID,
            PURPOSE5_FINAL_CERTIFICATE_SIGNER,
            INITIAL_KEY_EPOCH,
            self.q5a_candidate_root,
            self.q5a_unpinned_genesis_root,
            tuple(item.canonical_object() for item in self.signer_manifests),
            self.manifest_roots,
            self.key_ids,
            ACTIVE_SIGNER_COUNT,
            REQUIRED_CERTIFICATE_SIGNATURE_COUNT,
            FUTURE_STATUS_TRANSITION_THRESHOLD,
            FUTURE_STATUS_TRANSITION_ELIGIBLE_COUNT,
            Q6_SIGNATURE_PREIMAGE_POLICY_ID,
            Q4_BODY_ONLY_SIGNATURE_MESSAGE_FORBIDDEN_AS_ISSUANCE_GATE,
            True,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def genesis_root(self) -> bytes:
        return content_hash(
            Q5B_PINNED_GENESIS_ROOT_DOMAIN, self.canonical_object()
        )


def decode_pinned_trust_genesis_v1(payload: bytes) -> PinnedTrustGenesisV1:
    value = _decode(payload, "Q5b pinned trust genesis")
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
        or not _is_exact_int(value[0], 1)
        or not _is_exact_int(value[1], Q5B_PINNED_GENESIS_TAG)
        or not _is_exact_int(value[3], PURPOSE5_FINAL_CERTIFICATE_SIGNER)
        or not _is_exact_int(value[4], INITIAL_KEY_EPOCH)
        or type(value[7]) is not tuple
        or type(value[8]) is not tuple
        or type(value[9]) is not tuple
        or value[10:]
        != (
            ACTIVE_SIGNER_COUNT,
            REQUIRED_CERTIFICATE_SIGNATURE_COUNT,
            FUTURE_STATUS_TRANSITION_THRESHOLD,
            FUTURE_STATUS_TRANSITION_ELIGIBLE_COUNT,
            Q6_SIGNATURE_PREIMAGE_POLICY_ID,
            True,
            True,
        )
        or not _is_exact_int(value[10], ACTIVE_SIGNER_COUNT)
        or not _is_exact_int(value[11], REQUIRED_CERTIFICATE_SIGNATURE_COUNT)
        or not _is_exact_int(value[12], FUTURE_STATUS_TRANSITION_THRESHOLD)
        or not _is_exact_int(value[13], FUTURE_STATUS_TRANSITION_ELIGIBLE_COUNT)
        or not _is_exact_bool(value[15], True)
        or not _is_exact_bool(value[16], True)
    ):
        _fail("REJECT_Q5B_PINNED_GENESIS", "pinned genesis wire differs")
    _bytes32(value[5], "Q5a candidate root")
    _bytes32(value[6], "Q5a unpinned genesis root")
    manifests = tuple(
        _decode_sealed_q5a_signer_manifest(row)
        for row in value[7]
    )
    genesis = PinnedTrustGenesisV1(value[5], value[6], manifests)
    if value[8] != genesis.manifest_roots or value[9] != genesis.key_ids:
        _fail(
            "REJECT_Q5B_PINNED_GENESIS",
            "pinned genesis manifest roots or key ids differ",
        )
    if genesis.canonical_bytes != payload:
        _fail("REJECT_Q5B_PINNED_GENESIS", "pinned genesis replay differs")
    return genesis


@dataclass(frozen=True, slots=True)
class CurrentKeyStatusBundleV1:
    pinned_trust_genesis: PinnedTrustGenesisV1

    def __post_init__(self) -> None:
        if type(self.pinned_trust_genesis) is not PinnedTrustGenesisV1:
            _fail("REJECT_Q5B_CURRENT_STATUS", "pinned genesis type differs")

    @property
    def signer_manifests(
        self,
    ) -> tuple[SealedQ5aPurpose5SignerManifestV1, ...]:
        return self.pinned_trust_genesis.signer_manifests

    @property
    def purpose5_active_signer_count(self) -> int:
        return ACTIVE_SIGNER_COUNT

    @property
    def current_key_epoch(self) -> int:
        return INITIAL_KEY_EPOCH

    def canonical_object(self) -> tuple[object, ...]:
        genesis = self.pinned_trust_genesis
        return (
            1,
            Q5B_CURRENT_STATUS_TAG,
            Q5B_CURRENT_STATUS_SCHEMA_ID,
            PURPOSE5_FINAL_CERTIFICATE_SIGNER,
            genesis.canonical_object(),
            genesis.genesis_root,
            INITIAL_KEY_EPOCH,
            genesis.manifest_roots,
            genesis.key_ids,
            SIGNER_ORDINALS,
            ACTIVE_SIGNER_COUNT,
            FUTURE_STATUS_TRANSITION_THRESHOLD,
            FUTURE_STATUS_TRANSITION_ELIGIBLE_COUNT,
            Q6_SIGNATURE_PREIMAGE_POLICY_ID,
            True,
            True,
            0,
            0,
            0,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def status_root(self) -> bytes:
        return content_hash(
            Q5B_CURRENT_STATUS_ROOT_DOMAIN, self.canonical_object()
        )


def decode_current_key_status_bundle_v1(
    payload: bytes,
) -> CurrentKeyStatusBundleV1:
    value = _decode(payload, "Q5b current key-status bundle")
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
        or not _is_exact_int(value[0], 1)
        or not _is_exact_int(value[1], Q5B_CURRENT_STATUS_TAG)
        or not _is_exact_int(value[3], PURPOSE5_FINAL_CERTIFICATE_SIGNER)
        or type(value[4]) is not tuple
        or not _is_exact_int(value[6], INITIAL_KEY_EPOCH)
        or type(value[7]) is not tuple
        or type(value[8]) is not tuple
        or type(value[9]) is not tuple
        or value[9:]
        != (
            SIGNER_ORDINALS,
            ACTIVE_SIGNER_COUNT,
            FUTURE_STATUS_TRANSITION_THRESHOLD,
            FUTURE_STATUS_TRANSITION_ELIGIBLE_COUNT,
            Q6_SIGNATURE_PREIMAGE_POLICY_ID,
            True,
            True,
            0,
            0,
            0,
        )
        or not _is_exact_int(value[10], ACTIVE_SIGNER_COUNT)
        or not _is_exact_int(value[11], FUTURE_STATUS_TRANSITION_THRESHOLD)
        or not _is_exact_int(value[12], FUTURE_STATUS_TRANSITION_ELIGIBLE_COUNT)
        or not _is_exact_bool(value[14], True)
        or not _is_exact_bool(value[15], True)
        or any(not _is_exact_int(value[index], 0) for index in (16, 17, 18))
    ):
        _fail("REJECT_Q5B_CURRENT_STATUS", "current status wire differs")
    genesis = decode_pinned_trust_genesis_v1(canonical_cbor_encode(value[4]))
    if (
        value[5] != genesis.genesis_root
        or value[7] != genesis.manifest_roots
        or value[8] != genesis.key_ids
    ):
        _fail(
            "REJECT_Q5B_CURRENT_STATUS",
            "current status does not bind the pinned signer set",
        )
    status = CurrentKeyStatusBundleV1(genesis)
    if status.canonical_bytes != payload:
        _fail("REJECT_Q5B_CURRENT_STATUS", "current status replay differs")
    return status


@dataclass(frozen=True, slots=True)
class Q5bPurpose5TrustAmendmentReceiptV1:
    q5a_capsule_digest: bytes
    q5a_authorization_digest: bytes
    q5a_consumption_record_digest: bytes
    q5a_terminal_record_digest: bytes
    q5a_artifact_raw_sha256: bytes
    q5a_candidate: SealedQ5aPurpose5SignerGenesisCandidateV1
    q5a_candidate_root: bytes
    q5a_unpinned_genesis_root: bytes
    q5a_genesis_only_status_root: bytes
    q5a_manifest_roots: tuple[bytes, ...]
    q5a_key_ids: tuple[bytes, ...]
    q5a_source_commit_raw: bytes
    q5a_source_closure_root: bytes
    q5b_source_commit_raw: bytes
    q5b_source_closure_root: bytes
    pinned_trust_genesis: PinnedTrustGenesisV1
    current_key_status: CurrentKeyStatusBundleV1

    def __post_init__(self) -> None:
        for name in (
            "q5a_capsule_digest",
            "q5a_authorization_digest",
            "q5a_consumption_record_digest",
            "q5a_terminal_record_digest",
            "q5a_artifact_raw_sha256",
            "q5a_candidate_root",
            "q5a_unpinned_genesis_root",
            "q5a_genesis_only_status_root",
            "q5a_source_closure_root",
            "q5b_source_closure_root",
        ):
            _bytes32(getattr(self, name), name)
        _bytes(self.q5a_source_commit_raw, 20, "Q5a source commit")
        _bytes(self.q5b_source_commit_raw, 20, "Q5b source commit")
        if (
            type(self.q5a_candidate)
            is not SealedQ5aPurpose5SignerGenesisCandidateV1
            or sha256(self.q5a_candidate.canonical_bytes).digest()
            != self.q5a_artifact_raw_sha256
            or self.q5a_candidate.candidate_root != self.q5a_candidate_root
            or self.q5a_candidate.q5a_source_commit_raw
            != self.q5a_source_commit_raw
            or self.q5a_candidate.q5a_source_closure_root
            != self.q5a_source_closure_root
            or self.q5a_candidate.unpinned_genesis_root
            != self.q5a_unpinned_genesis_root
            or self.q5a_candidate.genesis_only_status_root
            != self.q5a_genesis_only_status_root
            or type(self.q5a_manifest_roots) is not tuple
            or len(self.q5a_manifest_roots) != ACTIVE_SIGNER_COUNT
            or any(type(root) is not bytes or len(root) != 32 for root in self.q5a_manifest_roots)
            or type(self.q5a_key_ids) is not tuple
            or len(self.q5a_key_ids) != ACTIVE_SIGNER_COUNT
            or any(type(key_id) is not bytes or len(key_id) != 16 for key_id in self.q5a_key_ids)
            or tuple(
                item.manifest_root for item in self.q5a_candidate.signer_manifests
            ) != self.q5a_manifest_roots
            or tuple(item.key_id for item in self.q5a_candidate.signer_manifests)
            != self.q5a_key_ids
            or type(self.pinned_trust_genesis) is not PinnedTrustGenesisV1
            or type(self.current_key_status) is not CurrentKeyStatusBundleV1
            or self.pinned_trust_genesis.q5a_candidate_root
            != self.q5a_candidate_root
            or self.pinned_trust_genesis.q5a_unpinned_genesis_root
            != self.q5a_unpinned_genesis_root
            or self.pinned_trust_genesis.manifest_roots
            != self.q5a_manifest_roots
            or self.pinned_trust_genesis.key_ids != self.q5a_key_ids
            or self.current_key_status.pinned_trust_genesis
            != self.pinned_trust_genesis
        ):
            _fail("REJECT_Q5B_RECEIPT", "Q5b receipt object graph differs")

    def canonical_object(self) -> tuple[object, ...]:
        return (
            1,
            Q5B_RECEIPT_TAG,
            Q5B_RECEIPT_SCHEMA_ID,
            self.q5a_capsule_digest,
            self.q5a_authorization_digest,
            self.q5a_consumption_record_digest,
            self.q5a_terminal_record_digest,
            self.q5a_artifact_raw_sha256,
            self.q5a_candidate.canonical_object,
            self.q5a_candidate_root,
            self.q5a_unpinned_genesis_root,
            self.q5a_genesis_only_status_root,
            self.q5a_manifest_roots,
            self.q5a_key_ids,
            (1, self.q5a_source_commit_raw),
            self.q5a_source_closure_root,
            (1, self.q5b_source_commit_raw),
            self.q5b_source_closure_root,
            self.pinned_trust_genesis.canonical_object(),
            self.pinned_trust_genesis.genesis_root,
            self.current_key_status.canonical_object(),
            self.current_key_status.status_root,
            PURPOSE5_FINAL_CERTIFICATE_SIGNER,
            INITIAL_KEY_EPOCH,
            ACTIVE_SIGNER_COUNT,
            FUTURE_STATUS_TRANSITION_THRESHOLD,
            FUTURE_STATUS_TRANSITION_ELIGIBLE_COUNT,
            Q6_SIGNATURE_PREIMAGE_POLICY_ID,
            Q4_BODY_ONLY_SIGNATURE_MESSAGE_FORBIDDEN_AS_ISSUANCE_GATE,
            ACTION_ID.encode("ascii"),
            Q5B_SUCCESS_STATUS.encode("ascii"),
            Q5B_CLAIM_LEVEL.encode("ascii"),
            True,
            True,
            True,
            0,
            0,
            0,
            False,
            False,
            False,
            GLOBAL_ACTIVE_TRANSITION_PERFORMED,
            False,
        )

    @property
    def purpose5_active_signer_count(self) -> int:
        return ACTIVE_SIGNER_COUNT

    @property
    def purpose5_trust_genesis_pinned(self) -> bool:
        return True

    @property
    def purpose5_current_key_status_accepted(self) -> bool:
        return True

    @property
    def global_active_transition_performed(self) -> bool:
        return GLOBAL_ACTIVE_TRANSITION_PERFORMED

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_cbor_encode(self.canonical_object())

    @property
    def receipt_root(self) -> bytes:
        return content_hash(Q5B_RECEIPT_ROOT_DOMAIN, self.canonical_object())


def build_q5b_trust_amendment_v1(
    q5a_artifact_payload: bytes,
    *,
    q5a_capsule_digest: bytes,
    q5a_authorization_digest: bytes,
    q5a_consumption_record_digest: bytes,
    q5a_terminal_record_digest: bytes,
    q5a_source_commit: str,
    q5a_source_closure_root: bytes,
    q5b_source_commit: str,
    q5b_source_closure_root: bytes,
) -> Q5bPurpose5TrustAmendmentReceiptV1:
    """Replay one Q5a artifact and derive the three public Q5b objects."""

    if type(q5a_artifact_payload) is not bytes:
        raise TypeError("Q5a artifact payload must be exact bytes")
    try:
        candidate = decode_sealed_q5a_candidate_v1(q5a_artifact_payload)
    except (TypeError, ValueError) as error:
        _fail("REJECT_Q5B_Q5A_ARTIFACT", str(error))
    pinned = PinnedTrustGenesisV1(
        candidate.candidate_root,
        candidate.unpinned_genesis_root,
        candidate.signer_manifests,
    )
    status = CurrentKeyStatusBundleV1(pinned)
    receipt = Q5bPurpose5TrustAmendmentReceiptV1(
        q5a_capsule_digest=q5a_capsule_digest,
        q5a_authorization_digest=q5a_authorization_digest,
        q5a_consumption_record_digest=q5a_consumption_record_digest,
        q5a_terminal_record_digest=q5a_terminal_record_digest,
        q5a_artifact_raw_sha256=sha256(q5a_artifact_payload).digest(),
        q5a_candidate=candidate,
        q5a_candidate_root=candidate.candidate_root,
        q5a_unpinned_genesis_root=candidate.unpinned_genesis_root,
        q5a_genesis_only_status_root=candidate.genesis_only_status_root,
        q5a_manifest_roots=tuple(
            manifest.manifest_root for manifest in candidate.signer_manifests
        ),
        q5a_key_ids=tuple(
            manifest.key_id for manifest in candidate.signer_manifests
        ),
        q5a_source_commit_raw=_commit_raw(q5a_source_commit, "Q5a source commit"),
        q5a_source_closure_root=q5a_source_closure_root,
        q5b_source_commit_raw=_commit_raw(q5b_source_commit, "Q5b source commit"),
        q5b_source_closure_root=q5b_source_closure_root,
        pinned_trust_genesis=pinned,
        current_key_status=status,
    )
    if decode_q5b_trust_amendment_receipt_v1(receipt.canonical_bytes) != receipt:
        _fail("REJECT_Q5B_RECEIPT", "Q5b receipt round trip differs")
    return receipt


def decode_q5b_trust_amendment_receipt_v1(
    payload: bytes,
) -> Q5bPurpose5TrustAmendmentReceiptV1:
    value = _decode(payload, "Q5b trust-amendment receipt")
    fixed_tail = (
        PURPOSE5_FINAL_CERTIFICATE_SIGNER,
        INITIAL_KEY_EPOCH,
        ACTIVE_SIGNER_COUNT,
        FUTURE_STATUS_TRANSITION_THRESHOLD,
        FUTURE_STATUS_TRANSITION_ELIGIBLE_COUNT,
        Q6_SIGNATURE_PREIMAGE_POLICY_ID,
        True,
        ACTION_ID.encode("ascii"),
        Q5B_SUCCESS_STATUS.encode("ascii"),
        Q5B_CLAIM_LEVEL.encode("ascii"),
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
        or not _is_exact_int(value[0], 1)
        or not _is_exact_int(value[1], Q5B_RECEIPT_TAG)
        or type(value[8]) is not tuple
        or type(value[12]) is not tuple
        or type(value[13]) is not tuple
        or type(value[14]) is not tuple
        or type(value[16]) is not tuple
        or type(value[18]) is not tuple
        or type(value[20]) is not tuple
        or value[22:] != fixed_tail
        or not _is_exact_int(value[22], PURPOSE5_FINAL_CERTIFICATE_SIGNER)
        or not _is_exact_int(value[23], INITIAL_KEY_EPOCH)
        or not _is_exact_int(value[24], ACTIVE_SIGNER_COUNT)
        or not _is_exact_int(value[25], FUTURE_STATUS_TRANSITION_THRESHOLD)
        or not _is_exact_int(value[26], FUTURE_STATUS_TRANSITION_ELIGIBLE_COUNT)
        or not _is_exact_bool(value[28], True)
        or any(not _is_exact_bool(value[index], True) for index in (32, 33, 34))
        or any(not _is_exact_int(value[index], 0) for index in (35, 36, 37))
        or any(not _is_exact_bool(value[index], False) for index in range(38, 43))
    ):
        _fail("REJECT_Q5B_RECEIPT", "Q5b receipt wire differs")
    for index, label in (
        (3, "Q5a capsule digest"),
        (4, "Q5a authorization digest"),
        (5, "Q5a consumption digest"),
        (6, "Q5a terminal digest"),
        (7, "Q5a artifact raw SHA-256"),
        (9, "Q5a candidate root"),
        (10, "Q5a unpinned genesis root"),
        (11, "Q5a genesis-only status root"),
        (15, "Q5a source closure root"),
        (17, "Q5b source closure root"),
        (19, "Q5b pinned genesis root"),
        (21, "Q5b current status root"),
    ):
        _bytes32(value[index], label)
    for index, label in ((14, "Q5a source commit"), (16, "Q5b source commit")):
        if (
            len(value[index]) != 2
            or not _is_exact_int(value[index][0], 1)
            or type(value[index][1]) is not bytes
            or len(value[index][1]) != 20
        ):
            _fail("REJECT_Q5B_REPOSITORY_COMMIT", f"{label} wire differs")
    candidate = decode_sealed_q5a_candidate_v1(canonical_cbor_encode(value[8]))
    pinned = decode_pinned_trust_genesis_v1(canonical_cbor_encode(value[18]))
    status = decode_current_key_status_bundle_v1(
        canonical_cbor_encode(value[20])
    )
    receipt = Q5bPurpose5TrustAmendmentReceiptV1(
        q5a_capsule_digest=value[3],
        q5a_authorization_digest=value[4],
        q5a_consumption_record_digest=value[5],
        q5a_terminal_record_digest=value[6],
        q5a_artifact_raw_sha256=value[7],
        q5a_candidate=candidate,
        q5a_candidate_root=value[9],
        q5a_unpinned_genesis_root=value[10],
        q5a_genesis_only_status_root=value[11],
        q5a_manifest_roots=value[12],
        q5a_key_ids=value[13],
        q5a_source_commit_raw=value[14][1],
        q5a_source_closure_root=value[15],
        q5b_source_commit_raw=value[16][1],
        q5b_source_closure_root=value[17],
        pinned_trust_genesis=pinned,
        current_key_status=status,
    )
    if (
        value[19] != pinned.genesis_root
        or value[21] != status.status_root
        or receipt.canonical_bytes != payload
    ):
        _fail("REJECT_Q5B_RECEIPT", "Q5b receipt nested roots differ")
    return receipt


def validate_q5b_receipt_against_q5a_artifact_v1(
    receipt_payload: bytes,
    q5a_artifact_payload: bytes,
) -> Q5bPurpose5TrustAmendmentReceiptV1:
    """Replay only the nested public Q5a artifact graph.

    This helper intentionally does not establish the Q5a outer authorization
    chain or either stage's source identity.  Authority-bearing callers must
    use :func:`validate_q5b_receipt_authority_v1` below.
    """

    replayed = decode_q5b_trust_amendment_receipt_v1(receipt_payload)
    expected = build_q5b_trust_amendment_v1(
        q5a_artifact_payload,
        q5a_capsule_digest=replayed.q5a_capsule_digest,
        q5a_authorization_digest=replayed.q5a_authorization_digest,
        q5a_consumption_record_digest=replayed.q5a_consumption_record_digest,
        q5a_terminal_record_digest=replayed.q5a_terminal_record_digest,
        q5a_source_commit=replayed.q5a_source_commit_raw.hex(),
        q5a_source_closure_root=replayed.q5a_source_closure_root,
        q5b_source_commit=replayed.q5b_source_commit_raw.hex(),
        q5b_source_closure_root=replayed.q5b_source_closure_root,
    )
    if replayed != expected:
        _fail("REJECT_Q5B_RECEIPT", "receipt/Q5a artifact binding differs")
    return replayed


def validate_q5b_receipt_authority_v1(
    receipt_payload: bytes,
    q5a_artifact_payload: bytes,
    *,
    expected_q5a_capsule_digest: bytes,
    expected_q5a_authorization_digest: bytes,
    expected_q5a_consumption_record_digest: bytes,
    expected_q5a_terminal_record_digest: bytes,
    expected_q5a_artifact_raw_sha256: bytes,
    expected_q5a_source_commit: str,
    expected_q5a_source_closure_root: bytes,
    expected_q5b_source_commit: str,
    expected_q5b_source_closure_root: bytes,
) -> Q5bPurpose5TrustAmendmentReceiptV1:
    """Replay Q5b against caller-supplied sealed authority expectations."""

    replayed = validate_q5b_receipt_against_q5a_artifact_v1(
        receipt_payload, q5a_artifact_payload
    )
    expected_fields = (
        (
            replayed.q5a_capsule_digest,
            _bytes32(expected_q5a_capsule_digest, "expected Q5a capsule digest"),
        ),
        (
            replayed.q5a_authorization_digest,
            _bytes32(
                expected_q5a_authorization_digest,
                "expected Q5a authorization digest",
            ),
        ),
        (
            replayed.q5a_consumption_record_digest,
            _bytes32(
                expected_q5a_consumption_record_digest,
                "expected Q5a consumption digest",
            ),
        ),
        (
            replayed.q5a_terminal_record_digest,
            _bytes32(expected_q5a_terminal_record_digest, "expected Q5a terminal digest"),
        ),
        (
            replayed.q5a_artifact_raw_sha256,
            _bytes32(
                expected_q5a_artifact_raw_sha256,
                "expected Q5a artifact raw SHA-256",
            ),
        ),
        (
            replayed.q5a_source_commit_raw,
            _commit_raw(expected_q5a_source_commit, "expected Q5a source commit"),
        ),
        (
            replayed.q5a_source_closure_root,
            _bytes32(
                expected_q5a_source_closure_root,
                "expected Q5a source closure",
            ),
        ),
        (
            replayed.q5b_source_commit_raw,
            _commit_raw(expected_q5b_source_commit, "expected Q5b source commit"),
        ),
        (
            replayed.q5b_source_closure_root,
            _bytes32(
                expected_q5b_source_closure_root,
                "expected Q5b source closure",
            ),
        ),
    )
    if any(actual != expected for actual, expected in expected_fields):
        _fail("REJECT_Q5B_AUTHORITY", "receipt authority expectation differs")
    if expected_q5a_artifact_raw_sha256 != sha256(q5a_artifact_payload).digest():
        _fail("REJECT_Q5B_AUTHORITY", "sealed Q5a artifact raw identity differs")
    return replayed


__all__ = [
    "ACTION_ID",
    "ACTIVE_SIGNER_COUNT",
    "CurrentKeyStatusBundleV1",
    "FUTURE_STATUS_TRANSITION_ELIGIBLE_COUNT",
    "FUTURE_STATUS_TRANSITION_THRESHOLD",
    "GLOBAL_ACTIVE_TRANSITION_PERFORMED",
    "PinnedTrustGenesisV1",
    "Q5B_SUCCESS_STATUS",
    "Q5bPurpose5TrustAmendmentError",
    "Q5bPurpose5TrustAmendmentReceiptV1",
    "Q6_SIGNATURE_PREIMAGE_POLICY_ID",
    "SealedQ5aPurpose5SignerGenesisCandidateV1",
    "SealedQ5aPurpose5SignerManifestV1",
    "build_q5b_trust_amendment_v1",
    "decode_current_key_status_bundle_v1",
    "decode_pinned_trust_genesis_v1",
    "decode_q5b_trust_amendment_receipt_v1",
    "decode_sealed_q5a_candidate_v1",
    "validate_q5b_receipt_against_q5a_artifact_v1",
    "validate_q5b_receipt_authority_v1",
]
