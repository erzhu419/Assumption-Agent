"""Pure split-cryptography primitives frozen for Phase-3A M2.5.

No function in this module obtains randomness, creates a key, writes custody
state, signs a root, or persists a split.  Synthetic byte strings may be used
to verify the frozen HKDF/HMAC/commitment wire, but an authoritative first seed
genesis remains an external-custody and normative-completion operation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from typing import Final, Iterable

from .phase3_m25_wire_v1 import FAIL_M25_NORMATIVE_GAP, M25WireError


HKDF_SALT: Final = b"HEGEL/SPLIT/HKDF/SALT/V1"
ROLE_INFO_PREFIX: Final = b"HEGEL/SPLIT/ROLE/V1"
RANK_PREFIX: Final = b"HEGEL/SPLIT/RANK/V1"
SEED_COMMITMENT_PREFIX: Final = b"HEGEL/SPLIT_MASTER_SEED_COMMITMENT/V1"
CUSTODIAN_SIGNATURE_PREFIX: Final = b"HEGEL/CUSTODIAN_MANIFEST_SIGNATURE/V1"
SHA256_DIGEST_SIZE: Final = 32
SPLIT_SEED_SIZE: Final = 32
ED25519_PUBLIC_KEY_SIZE: Final = 32
ED25519_KEY_ID_SIZE: Final = 16


def _fail(code: str, detail: str) -> "None":
    raise M25WireError(code, detail)


def _require_bytes(value: object, length: int, name: str) -> bytes:
    if type(value) is not bytes or len(value) != length:
        _fail("REJECT_M25_CRYPTO_INPUT", f"{name} must be exactly {length} bytes")
    return value


def uint16_be(value: int, *, name: str = "value") -> bytes:
    """Encode one unsigned 16-bit integer exactly as required by the split KDF."""

    if type(value) is not int or not 0 <= value <= 0xFFFF:
        _fail("REJECT_M25_UINT16", f"{name} must be an integer in [0, 65535]")
    return value.to_bytes(2, "big")


def hkdf_extract_sha256(ikm: bytes, salt: bytes) -> bytes:
    """RFC 5869 HKDF-Extract using SHA-256 (pure, deterministic)."""

    if type(ikm) is not bytes or type(salt) is not bytes:
        raise TypeError("HKDF input key material and salt must be bytes")
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def hkdf_expand_sha256(prk: bytes, info: bytes, length: int) -> bytes:
    """RFC 5869 HKDF-Expand using SHA-256."""

    if type(prk) is not bytes or len(prk) < SHA256_DIGEST_SIZE:
        _fail(
            "REJECT_M25_CRYPTO_INPUT",
            "HKDF SHA-256 PRK must contain at least 32 bytes",
        )
    if type(info) is not bytes:
        raise TypeError("HKDF info must be bytes")
    if type(length) is not int or not 0 <= length <= 255 * SHA256_DIGEST_SIZE:
        _fail("REJECT_M25_HKDF_LENGTH", "HKDF length is outside [0, 8160]")
    output = bytearray()
    previous = b""
    counter = 1
    while len(output) < length:
        previous = hmac.new(
            prk,
            previous + info + bytes((counter,)),
            hashlib.sha256,
        ).digest()
        output.extend(previous)
        counter += 1
    return bytes(output[:length])


def split_hkdf_prk(split_master_seed: bytes) -> bytes:
    """Extract the frozen split PRK from a caller-supplied 32-byte test input."""

    seed = _require_bytes(split_master_seed, SPLIT_SEED_SIZE, "split_master_seed")
    return hkdf_extract_sha256(seed, HKDF_SALT)


def derive_role_key(split_master_seed: bytes, role_id: int) -> bytes:
    """Derive ``K_role(role_id)`` under the frozen M2.5 profile."""

    prk = split_hkdf_prk(split_master_seed)
    info = ROLE_INFO_PREFIX + uint16_be(role_id, name="role_id")
    return hkdf_expand_sha256(prk, info, SHA256_DIGEST_SIZE)


def split_rank(
    role_key: bytes,
    role_id: int,
    stratum_id: int,
    canonical_input_hash: bytes,
) -> bytes:
    """Compute the frozen HMAC rank digest for one row."""

    key = _require_bytes(role_key, SHA256_DIGEST_SIZE, "role_key")
    input_hash = _require_bytes(
        canonical_input_hash,
        SHA256_DIGEST_SIZE,
        "canonical_input_hash",
    )
    message = (
        RANK_PREFIX
        + uint16_be(role_id, name="role_id")
        + uint16_be(stratum_id, name="stratum_id")
        + input_hash
    )
    return hmac.new(key, message, hashlib.sha256).digest()


def split_seed_commitment(split_master_seed: bytes) -> bytes:
    """Compute the public commitment digest without serializing the seed."""

    seed = _require_bytes(split_master_seed, SPLIT_SEED_SIZE, "split_master_seed")
    return hashlib.sha256(SEED_COMMITMENT_PREFIX + b"\x00" + seed).digest()


def ed25519_key_id(raw_public_key: bytes) -> bytes:
    """Return the first 16 bytes of SHA-256(raw Ed25519 public key)."""

    public_key = _require_bytes(
        raw_public_key,
        ED25519_PUBLIC_KEY_SIZE,
        "raw_ed25519_public_key",
    )
    return hashlib.sha256(public_key).digest()[:ED25519_KEY_ID_SIZE]


def custodian_signature_preimage(manifest_root: bytes) -> bytes:
    """Return the exact bytes an external custodian must sign."""

    root = _require_bytes(manifest_root, SHA256_DIGEST_SIZE, "manifest_root")
    return CUSTODIAN_SIGNATURE_PREFIX + b"\x00" + root


@dataclass(frozen=True)
class SplitRankInput:
    """Caller-owned row identity used only for deterministic collision checks."""

    canonical_input_hash: bytes
    row_identity: bytes
    stratum_id: int


@dataclass(frozen=True)
class RankedSplitRow:
    rank_digest: bytes
    canonical_input_hash: bytes
    row_identity: bytes
    stratum_id: int


def rank_split_rows(
    role_key: bytes,
    role_id: int,
    rows: Iterable[SplitRankInput],
) -> tuple[RankedSplitRow, ...]:
    """Rank and order rows by ``(rank_digest, canonical_input_hash)``.

    A different row sharing both tie-break values fails with the amendment's
    exact identity-collision code.  This helper does not assign partitions or
    quotas because the sink split contract is still normatively incomplete.
    """

    key = _require_bytes(role_key, SHA256_DIGEST_SIZE, "role_key")
    ranked: list[RankedSplitRow] = []
    for row in rows:
        if not isinstance(row, SplitRankInput):
            raise TypeError("rows must contain SplitRankInput values")
        input_hash = _require_bytes(
            row.canonical_input_hash,
            SHA256_DIGEST_SIZE,
            "canonical_input_hash",
        )
        if type(row.row_identity) is not bytes:
            _fail("REJECT_M25_CRYPTO_INPUT", "row_identity must be bytes")
        rank = split_rank(key, role_id, row.stratum_id, input_hash)
        ranked.append(
            RankedSplitRow(
                rank_digest=rank,
                canonical_input_hash=input_hash,
                row_identity=row.row_identity,
                stratum_id=row.stratum_id,
            )
        )
    ranked.sort(key=lambda item: (item.rank_digest, item.canonical_input_hash))
    for left, right in zip(ranked, ranked[1:]):
        if (
            left.rank_digest == right.rank_digest
            and left.canonical_input_hash == right.canonical_input_hash
            and left.row_identity != right.row_identity
        ):
            _fail(
                "FAIL_SPLIT_RANK_IDENTITY_COLLISION",
                "different rows share both rank digest and canonical input hash",
            )
    return tuple(ranked)


def assert_authoritative_seed_genesis_available() -> "None":
    """Fail closed: this pure foundation is not an independent custodian."""

    _fail(
        FAIL_M25_NORMATIVE_GAP,
        "CustodianGenesis: actor eligibility, persistent second-invocation guard, "
        "secret transport/storage, and public artifact boundary are not frozen",
    )


__all__ = [
    "CUSTODIAN_SIGNATURE_PREFIX",
    "ED25519_KEY_ID_SIZE",
    "ED25519_PUBLIC_KEY_SIZE",
    "HKDF_SALT",
    "RANK_PREFIX",
    "ROLE_INFO_PREFIX",
    "RankedSplitRow",
    "SEED_COMMITMENT_PREFIX",
    "SHA256_DIGEST_SIZE",
    "SPLIT_SEED_SIZE",
    "SplitRankInput",
    "assert_authoritative_seed_genesis_available",
    "custodian_signature_preimage",
    "derive_role_key",
    "ed25519_key_id",
    "hkdf_expand_sha256",
    "hkdf_extract_sha256",
    "rank_split_rows",
    "split_hkdf_prk",
    "split_rank",
    "split_seed_commitment",
    "uint16_be",
]
