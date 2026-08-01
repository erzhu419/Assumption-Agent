"""Pure split cryptography and deterministic allocation for Phase-3A M2.5.

No function in this module obtains randomness, creates a key, writes custody
state, signs a root, or persists a split. Synthetic byte strings may be used
to verify the frozen HKDF/HMAC/commitment and exhaustive quota allocation wire,
but an authoritative first seed genesis remains an external-custody operation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from types import MappingProxyType
from typing import Final, Iterable, Mapping

from .phase3_m25_rows_v1 import (
    ODD_INPUT_SIGNATURE_ID,
    SINK_INPUT_SIGNATURE_ID,
    TypedRoleRows,
)
from .phase3_m25_wire_v1 import M25WireError
from .strict_cbor_v1 import canonical_cbor_encode, rfc6962_root


HKDF_SALT: Final = b"HEGEL/SPLIT/HKDF/SALT/V1"
ROLE_INFO_PREFIX: Final = b"HEGEL/SPLIT/ROLE/V1"
RANK_PREFIX: Final = b"HEGEL/SPLIT/RANK/V1"
SEED_COMMITMENT_PREFIX: Final = b"HEGEL/SPLIT_MASTER_SEED_COMMITMENT/V1"
CUSTODIAN_SIGNATURE_PREFIX: Final = b"HEGEL/CUSTODIAN_MANIFEST_SIGNATURE/V1"
SHA256_DIGEST_SIZE: Final = 32
SPLIT_SEED_SIZE: Final = 32
ED25519_PUBLIC_KEY_SIZE: Final = 32
ED25519_KEY_ID_SIZE: Final = 16

OUTSIDE_ROLE_ID: Final = 1
NULL_CONTROL_ROLE_ID: Final = 2
DISCOVERY_PARTITION_ID: Final = 1
VALIDATION_PARTITION_ID: Final = 2
SEALED_PREDICTION_PARTITION_ID: Final = 3
SPLIT_ASSIGNMENT_ROW_TAG: Final = 0x3203
SPLIT_ASSIGNMENT_ROW_SCHEMA_ID: Final = b"hegel-split-assignment-row/1"


@dataclass(frozen=True, slots=True)
class StratumQuota:
    """One exhaustive stratum quota frozen by v1.1.2."""

    universe: int
    discovery: int
    validation: int
    sealed: int

    def __post_init__(self) -> None:
        values = (self.universe, self.discovery, self.validation, self.sealed)
        if any(type(value) is not int or value < 0 for value in values):
            raise TypeError("split quotas must be nonnegative exact integers")
        if self.discovery + self.validation + self.sealed != self.universe:
            raise ValueError("split quotas must exhaust their stratum")


ODD_STRATUM_QUOTAS: Final = MappingProxyType(
    {
        1: StratumQuota(16, 6, 3, 7),
        2: StratumQuota(16, 6, 3, 7),
        3: StratumQuota(32, 13, 6, 13),
        4: StratumQuota(32, 13, 6, 13),
        5: StratumQuota(64, 26, 13, 25),
        6: StratumQuota(64, 26, 13, 25),
        7: StratumQuota(128, 51, 26, 51),
        8: StratumQuota(128, 51, 26, 51),
    }
)
SINK_STRATUM_QUOTAS: Final = MappingProxyType(
    {
        9: StratumQuota(15, 7, 4, 4),
        10: StratumQuota(18, 8, 4, 6),
        11: StratumQuota(19, 9, 4, 6),
        12: StratumQuota(18, 8, 4, 6),
        13: StratumQuota(15, 7, 4, 4),
    }
)


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
    universe_index: int | None = None


@dataclass(frozen=True)
class RankedSplitRow:
    rank_digest: bytes
    canonical_input_hash: bytes
    row_identity: bytes
    stratum_id: int
    universe_index: int | None


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """One public formal assignment identity; membership remains custodian-sealed."""

    role_id: int
    universe_index: int
    canonical_input_hash: bytes
    stratum_id: int
    partition_id: int
    rank_digest: bytes

    def formal_row(self) -> tuple[object, ...]:
        return (
            1,
            SPLIT_ASSIGNMENT_ROW_TAG,
            SPLIT_ASSIGNMENT_ROW_SCHEMA_ID,
            self.role_id,
            self.universe_index,
            self.canonical_input_hash,
            self.stratum_id,
            self.partition_id,
            self.rank_digest,
        )


@dataclass(frozen=True, slots=True)
class SplitPartitionCommitments:
    """Counts and RFC6962 roots for one role's exhaustive three-way split."""

    role_id: int
    discovery_count: int
    validation_count: int
    sealed_count: int
    discovery_root: bytes
    validation_root: bytes
    sealed_root: bytes


def rank_split_rows(
    role_key: bytes,
    role_id: int,
    rows: Iterable[SplitRankInput],
) -> tuple[RankedSplitRow, ...]:
    """Rank and order rows by ``(rank_digest, canonical_input_hash)``.

    A different row sharing both tie-break values fails with the amendment's
    exact identity-collision code. Partition cuts are applied separately by
    :func:`allocate_split_rows` after per-stratum ranking.
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
                universe_index=row.universe_index,
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


def _validate_quota_table(quotas: Mapping[int, StratumQuota]) -> None:
    if not isinstance(quotas, Mapping) or not quotas:
        _fail("FAIL_SPLIT_QUOTA_MISMATCH", "quota table must be a nonempty mapping")
    keys = tuple(quotas)
    if keys != tuple(sorted(keys)):
        _fail("FAIL_SPLIT_QUOTA_MISMATCH", "quota strata must be ascending")
    if any(type(key) is not int or key <= 0 for key in keys):
        _fail("FAIL_SPLIT_QUOTA_MISMATCH", "stratum IDs must be positive integers")
    if any(not isinstance(quota, StratumQuota) for quota in quotas.values()):
        _fail("FAIL_SPLIT_QUOTA_MISMATCH", "quota values must be StratumQuota")


def allocate_split_rows(
    role_key: bytes,
    role_id: int,
    rows: Iterable[SplitRankInput],
    quotas: Mapping[int, StratumQuota],
) -> tuple[SplitAssignment, ...]:
    """Allocate caller-supplied rows under the frozen quota registry.

    This is a low-level synthetic/untrusted primitive: it validates structural
    exhaustion, but it cannot establish that a caller-supplied stratum matches
    the target semantics.  Typed candidate replay must enter through
    :func:`allocate_typed_role_rows`, which derives index, input hash, and
    stratum from validated ``TypedRoleRows``.  Neither path constitutes Gate 22
    or authoritative custodian evidence.
    """

    key = _require_bytes(role_key, SHA256_DIGEST_SIZE, "role_key")
    if type(role_id) is not int or role_id not in {OUTSIDE_ROLE_ID, NULL_CONTROL_ROLE_ID}:
        _fail("REJECT_UNKNOWN_ENUM_VALUE", "split role ID must be 1 or 2")
    _validate_quota_table(quotas)
    expected_quotas = (
        ODD_STRATUM_QUOTAS
        if role_id == OUTSIDE_ROLE_ID
        else SINK_STRATUM_QUOTAS
    )
    if dict(quotas) != dict(expected_quotas):
        _fail(
            "FAIL_SPLIT_QUOTA_MISMATCH",
            "role quota table differs from the frozen v1.1.2 registry",
        )
    materialized = tuple(rows)
    if any(not isinstance(row, SplitRankInput) for row in materialized):
        raise TypeError("rows must contain SplitRankInput values")
    indices = [row.universe_index for row in materialized]
    if any(type(index) is not int or index < 0 for index in indices):
        _fail(
            "FAIL_SPLIT_UNIVERSE_INDEX_GAP",
            "allocation rows require nonnegative universe_index values",
        )
    exact_indices = [int(index) for index in indices]
    if len(set(exact_indices)) != len(exact_indices):
        _fail("FAIL_SPLIT_UNIVERSE_INDEX_DUPLICATE", "universe indices repeat")
    if sorted(exact_indices) != list(range(len(exact_indices))):
        _fail("FAIL_SPLIT_UNIVERSE_INDEX_GAP", "universe indices must be contiguous")

    assignments: list[SplitAssignment] = []
    for stratum_id, quota in quotas.items():
        stratum_rows = tuple(row for row in materialized if row.stratum_id == stratum_id)
        if len(stratum_rows) != quota.universe:
            _fail(
                "FAIL_SPLIT_QUOTA_MISMATCH",
                f"stratum {stratum_id} has {len(stratum_rows)} rows, expected {quota.universe}",
            )
        ranked = rank_split_rows(key, role_id, stratum_rows)
        boundaries = (quota.discovery, quota.discovery + quota.validation)
        for rank_index, item in enumerate(ranked):
            if rank_index < boundaries[0]:
                partition_id = DISCOVERY_PARTITION_ID
            elif rank_index < boundaries[1]:
                partition_id = VALIDATION_PARTITION_ID
            else:
                partition_id = SEALED_PREDICTION_PARTITION_ID
            assert item.universe_index is not None
            assignments.append(
                SplitAssignment(
                    role_id=role_id,
                    universe_index=item.universe_index,
                    canonical_input_hash=item.canonical_input_hash,
                    stratum_id=stratum_id,
                    partition_id=partition_id,
                    rank_digest=item.rank_digest,
                )
            )
    unexpected = sorted({row.stratum_id for row in materialized} - set(quotas))
    if unexpected:
        _fail(
            "FAIL_SPLIT_QUOTA_MISMATCH",
            f"rows use unregistered strata {unexpected}",
        )
    if len(assignments) != len(materialized):
        _fail("FAIL_SPLIT_ASSIGNMENT_NOT_EXHAUSTIVE", "not every row was assigned")
    assignments.sort(key=lambda row: row.universe_index)
    return tuple(assignments)


def typed_role_split_rank_inputs(
    rows: TypedRoleRows,
) -> tuple[int, tuple[SplitRankInput, ...], Mapping[int, StratumQuota]]:
    """Derive split inputs from validated typed rows without caller strata.

    Odd strata are bound to ``(set_size, target label)`` exactly as frozen in
    amendment section 5.1.  Sink strata are bound to the observed ``d`` value
    exactly as frozen in section 5.2.  The universe index and canonical input
    hash are taken from the mutually validated universe/truth row pair.
    """

    if not isinstance(rows, TypedRoleRows):
        raise TypeError("rows must be a TypedRoleRows value")
    rows.validate()
    if rows.input_signature_id == ODD_INPUT_SIGNATURE_ID:
        if rows.role_name != "odd":
            _fail(
                "FAIL_SPLIT_CUSTODIAN_BINDING_MISMATCH",
                "odd InputSignatureId requires role_name 'odd'",
            )
        role_id = OUTSIDE_ROLE_ID
        quotas = ODD_STRATUM_QUOTAS
    elif rows.input_signature_id == SINK_INPUT_SIGNATURE_ID:
        if rows.role_name != "sink":
            _fail(
                "FAIL_SPLIT_CUSTODIAN_BINDING_MISMATCH",
                "sink InputSignatureId requires role_name 'sink'",
            )
        role_id = NULL_CONTROL_ROLE_ID
        quotas = SINK_STRATUM_QUOTAS
    else:
        _fail(
            "FAIL_SPLIT_CUSTODIAN_BINDING_MISMATCH",
            "typed role has no frozen split-role binding",
        )

    rank_inputs: list[SplitRankInput] = []
    for universe_row, truth_row in zip(
        rows.universe_rows,
        rows.truth_rows,
        strict=True,
    ):
        universe_index = universe_row[3]
        input_object = universe_row[5]
        canonical_input_hash = truth_row[4]
        target_output = truth_row[5]
        if role_id == OUTSIDE_ROLE_ID:
            set_size = input_object[3]
            stratum_id = 1 + 2 * (set_size - 5) + target_output
        else:
            d_value = input_object[6]
            stratum_id = 9 + d_value
        if stratum_id not in quotas:
            _fail(
                "FAIL_SPLIT_QUOTA_MISMATCH",
                "typed row derives an unregistered stratum",
            )
        rank_inputs.append(
            SplitRankInput(
                canonical_input_hash=canonical_input_hash,
                row_identity=canonical_cbor_encode(universe_row),
                stratum_id=stratum_id,
                universe_index=universe_index,
            )
        )
    return role_id, tuple(rank_inputs), quotas


def allocate_typed_role_rows(
    role_key: bytes,
    rows: TypedRoleRows,
) -> tuple[SplitAssignment, ...]:
    """Allocate validated typed candidate rows with semantic strata binding."""

    role_id, rank_inputs, quotas = typed_role_split_rank_inputs(rows)
    return allocate_split_rows(role_key, role_id, rank_inputs, quotas)


def split_partition_commitments(
    role_id: int,
    assignments: Iterable[SplitAssignment],
) -> SplitPartitionCommitments:
    """Commit each role+partition directly over universe-index-ordered rows."""

    materialized = tuple(assignments)
    if type(role_id) is not int or role_id not in {OUTSIDE_ROLE_ID, NULL_CONTROL_ROLE_ID}:
        _fail("REJECT_UNKNOWN_ENUM_VALUE", "split role ID must be 1 or 2")
    if any(not isinstance(row, SplitAssignment) for row in materialized):
        raise TypeError("assignments must contain SplitAssignment values")
    if any(type(row.role_id) is not int or row.role_id != role_id for row in materialized):
        _fail("FAIL_SPLIT_CUSTODIAN_BINDING_MISMATCH", "assignment role mismatch")
    expected_quotas = (
        ODD_STRATUM_QUOTAS
        if role_id == OUTSIDE_ROLE_ID
        else SINK_STRATUM_QUOTAS
    )
    indices = [row.universe_index for row in materialized]
    if any(type(index) is not int or index < 0 for index in indices):
        _fail("FAIL_SPLIT_UNIVERSE_INDEX_GAP", "assignment index is invalid")
    if len(indices) != len(set(indices)):
        _fail("FAIL_SPLIT_UNIVERSE_INDEX_DUPLICATE", "assignment indices repeat")
    if sorted(indices) != list(range(len(indices))):
        _fail("FAIL_SPLIT_UNIVERSE_INDEX_GAP", "assignment indices must be contiguous")
    if any(
        type(row.canonical_input_hash) is not bytes
        or len(row.canonical_input_hash) != SHA256_DIGEST_SIZE
        or type(row.rank_digest) is not bytes
        or len(row.rank_digest) != SHA256_DIGEST_SIZE
        for row in materialized
    ):
        _fail("REJECT_M25_CRYPTO_INPUT", "assignment hashes must be 32 bytes")
    if any(
        type(row.partition_id) is not int
        or type(row.stratum_id) is not int
        or row.partition_id
        not in {
            DISCOVERY_PARTITION_ID,
            VALIDATION_PARTITION_ID,
            SEALED_PREDICTION_PARTITION_ID,
        }
        or row.stratum_id not in expected_quotas
        for row in materialized
    ):
        _fail("FAIL_SPLIT_QUOTA_MISMATCH", "assignment partition/stratum is invalid")
    for stratum_id, quota in expected_quotas.items():
        actual = {
            partition_id: sum(
                row.stratum_id == stratum_id and row.partition_id == partition_id
                for row in materialized
            )
            for partition_id in (
                DISCOVERY_PARTITION_ID,
                VALIDATION_PARTITION_ID,
                SEALED_PREDICTION_PARTITION_ID,
            )
        }
        if actual != {
            DISCOVERY_PARTITION_ID: quota.discovery,
            VALIDATION_PARTITION_ID: quota.validation,
            SEALED_PREDICTION_PARTITION_ID: quota.sealed,
        }:
            _fail(
                "FAIL_SPLIT_QUOTA_MISMATCH",
                f"assignment stratum {stratum_id} does not match frozen quotas",
            )
    roots: dict[int, bytes] = {}
    counts: dict[int, int] = {}
    for partition_id in (
        DISCOVERY_PARTITION_ID,
        VALIDATION_PARTITION_ID,
        SEALED_PREDICTION_PARTITION_ID,
    ):
        partition = sorted(
            (row for row in materialized if row.partition_id == partition_id),
            key=lambda row: row.universe_index,
        )
        counts[partition_id] = len(partition)
        roots[partition_id] = rfc6962_root(
            [row.formal_row() for row in partition]
        )
    return SplitPartitionCommitments(
        role_id=role_id,
        discovery_count=counts[DISCOVERY_PARTITION_ID],
        validation_count=counts[VALIDATION_PARTITION_ID],
        sealed_count=counts[SEALED_PREDICTION_PARTITION_ID],
        discovery_root=roots[DISCOVERY_PARTITION_ID],
        validation_root=roots[VALIDATION_PARTITION_ID],
        sealed_root=roots[SEALED_PREDICTION_PARTITION_ID],
    )


def assert_authoritative_seed_genesis_available() -> "None":
    """Fail closed: this pure foundation is not an independent custodian."""

    _fail(
        "FAIL_CUSTODIAN_KEY_MISSING",
        "this process is not an independent custodian and has no external key binding",
    )


__all__ = [
    "CUSTODIAN_SIGNATURE_PREFIX",
    "DISCOVERY_PARTITION_ID",
    "ED25519_KEY_ID_SIZE",
    "ED25519_PUBLIC_KEY_SIZE",
    "HKDF_SALT",
    "NULL_CONTROL_ROLE_ID",
    "ODD_STRATUM_QUOTAS",
    "OUTSIDE_ROLE_ID",
    "RANK_PREFIX",
    "ROLE_INFO_PREFIX",
    "RankedSplitRow",
    "SEALED_PREDICTION_PARTITION_ID",
    "SEED_COMMITMENT_PREFIX",
    "SHA256_DIGEST_SIZE",
    "SPLIT_SEED_SIZE",
    "SplitRankInput",
    "SINK_STRATUM_QUOTAS",
    "SPLIT_ASSIGNMENT_ROW_SCHEMA_ID",
    "SPLIT_ASSIGNMENT_ROW_TAG",
    "SplitAssignment",
    "SplitPartitionCommitments",
    "StratumQuota",
    "VALIDATION_PARTITION_ID",
    "allocate_split_rows",
    "allocate_typed_role_rows",
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
    "split_partition_commitments",
    "typed_role_split_rank_inputs",
    "uint16_be",
]
