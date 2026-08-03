#!/usr/bin/env python3
"""Independent one-shot v1.1.2 split-partition calculator.

This executable deliberately imports no Hegel Machine package.  Its only
secret input is exactly 32 bytes followed by EOF from an inherited FIFO at raw
file descriptor 3.  Its only success output is one public frame on an
inherited FIFO at raw file descriptor 5:

    uint64_be(payload_length) || strict_canonical_cbor(payload)

The implementation-evidence payload is the ceremony's unique V2 array (not a
formal object and not a new formal tag or ContentHash domain):

    [
      1,
      b"hegel-phase3-split-calculator-fd3-response/2",
      seed_commitment_32,
      [
        [role_id, partition_id, public_row_count, rfc6962_root],
        ...
      ]
    ]

Only public commitment, quota/count, and root evidence leaves the process.
Derived keys, rank digests, and row membership never do.  Failures are silent
and fail closed with a nonzero exit status.
"""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import os
import stat
import sys


SEED_FD = 3
PUBLIC_RESPONSE_FD = 5
SEED_SIZE = 32
FAILURE_EXIT_STATUS = 71

RESPONSE_SCHEMA_ID = b"hegel-phase3-split-calculator-fd3-response/2"

SEED_COMMITMENT_PREFIX = b"HEGEL/SPLIT_MASTER_SEED_COMMITMENT/V1"
HKDF_SALT = b"HEGEL/SPLIT/HKDF/SALT/V1"
ROLE_INFO_PREFIX = b"HEGEL/SPLIT/ROLE/V1"
RANK_PREFIX = b"HEGEL/SPLIT/RANK/V1"
CANONICAL_INPUT_DOMAIN = b"HEGEL/CANONICAL_INPUT/V1"

OUTSIDE_ROLE_ID = 1
NULL_CONTROL_ROLE_ID = 2
DISCOVERY_PARTITION_ID = 1
VALIDATION_PARTITION_ID = 2
SEALED_PARTITION_ID = 3

ODD_INPUT_TAG = 0x3401
SINK_INPUT_TAG = 0x3402
SPLIT_ASSIGNMENT_ROW_TAG = 0x3203
ODD_INPUT_SCHEMA_ID = b"hegel-odd-input/1"
SINK_INPUT_SCHEMA_ID = b"hegel-sink-input/1"
SPLIT_ASSIGNMENT_ROW_SCHEMA_ID = b"hegel-split-assignment-row/1"

# (stratum_id, universe, discovery, validation, sealed)
ODD_QUOTAS = (
    (1, 16, 6, 3, 7),
    (2, 16, 6, 3, 7),
    (3, 32, 13, 6, 13),
    (4, 32, 13, 6, 13),
    (5, 64, 26, 13, 25),
    (6, 64, 26, 13, 25),
    (7, 128, 51, 26, 51),
    (8, 128, 51, 26, 51),
)
SINK_QUOTAS = (
    (9, 15, 7, 4, 4),
    (10, 18, 8, 4, 6),
    (11, 19, 9, 4, 6),
    (12, 18, 8, 4, 6),
    (13, 15, 7, 4, 4),
)


class CalculatorFailure(Exception):
    """Internal marker whose detail is never serialized or logged."""


def _zeroize(secret: bytearray) -> None:
    for index in range(len(secret)):
        secret[index] = 0


def _try_mlock(secret: bytearray) -> bool:
    if not secret:
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        operation = libc.mlock
        operation.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        operation.restype = ctypes.c_int
        address = ctypes.addressof(ctypes.c_ubyte.from_buffer(secret))
        return operation(address, len(secret)) == 0
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def _try_munlock(secret: bytearray, locked: bool) -> None:
    if not locked or not secret:
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        operation = libc.munlock
        operation.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        operation.restype = ctypes.c_int
        address = ctypes.addressof(ctypes.c_ubyte.from_buffer(secret))
        operation(address, len(secret))
    except (AttributeError, OSError, TypeError, ValueError):
        pass


def _require_fifo(fd: int) -> None:
    try:
        mode = os.fstat(fd).st_mode
    except OSError as error:
        raise CalculatorFailure from error
    if not stat.S_ISFIFO(mode):
        raise CalculatorFailure


def _read_seed() -> tuple[bytearray, bool]:
    _require_fifo(SEED_FD)
    seed = bytearray(SEED_SIZE)
    locked = _try_mlock(seed)
    offset = 0
    try:
        while offset < SEED_SIZE:
            count = os.readv(SEED_FD, (memoryview(seed)[offset:],))
            if count == 0:
                raise CalculatorFailure
            offset += count
        extra = bytearray(1)
        try:
            extra_count = os.readv(SEED_FD, (extra,))
        finally:
            _zeroize(extra)
        if extra_count != 0:
            raise CalculatorFailure
        return seed, locked
    except BaseException:
        _zeroize(seed)
        _try_munlock(seed, locked)
        raise
    finally:
        try:
            os.close(SEED_FD)
        except OSError:
            pass


def _cbor_head(major: int, value: int) -> bytes:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CalculatorFailure
    prefix = major << 5
    if value <= 23:
        return bytes((prefix | value,))
    if value <= 0xFF:
        return bytes((prefix | 24, value))
    if value <= 0xFFFF:
        return bytes((prefix | 25,)) + value.to_bytes(2, "big")
    if value <= 0xFFFFFFFF:
        return bytes((prefix | 26,)) + value.to_bytes(4, "big")
    if value <= 0xFFFFFFFFFFFFFFFF:
        return bytes((prefix | 27,)) + value.to_bytes(8, "big")
    raise CalculatorFailure


def _cbor(value: object) -> bytes:
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            return _cbor_head(1, -1 - value)
        return _cbor_head(0, value)
    if isinstance(value, bytes):
        return _cbor_head(2, len(value)) + value
    if isinstance(value, (tuple, list)):
        return _cbor_head(4, len(value)) + b"".join(_cbor(item) for item in value)
    raise CalculatorFailure


def _content_hash(domain: bytes, value: object) -> bytes:
    return hashlib.sha256(domain + b"\x00" + _cbor(value)).digest()


def _rfc6962_root(records: list[object]) -> bytes:
    leaves = [hashlib.sha256(b"\x00" + _cbor(record)).digest() for record in records]
    return _rfc6962_hashes(leaves)


def _rfc6962_hashes(hashes: list[bytes]) -> bytes:
    count = len(hashes)
    if count == 0:
        return hashlib.sha256(b"").digest()
    if count == 1:
        return hashes[0]
    split = 1 << ((count - 1).bit_length() - 1)
    left = _rfc6962_hashes(hashes[:split])
    right = _rfc6962_hashes(hashes[split:])
    return hashlib.sha256(b"\x01" + left + right).digest()


def _derive_role_key(seed: bytearray, role_id: int) -> bytearray:
    prk = bytearray(hmac.new(HKDF_SALT, seed, hashlib.sha256).digest())
    try:
        info = ROLE_INFO_PREFIX + role_id.to_bytes(2, "big")
        # Exactly one RFC 5869 expand block is needed for a 32-byte SHA-256 key.
        return bytearray(hmac.new(prk, info + b"\x01", hashlib.sha256).digest())
    finally:
        _zeroize(prk)


def _rank(
    role_key: bytearray,
    role_id: int,
    stratum_id: int,
    input_hash: bytes,
) -> bytes:
    message = (
        RANK_PREFIX
        + role_id.to_bytes(2, "big")
        + stratum_id.to_bytes(2, "big")
        + input_hash
    )
    return hmac.new(role_key, message, hashlib.sha256).digest()


def _odd_rows() -> list[tuple[int, bytes, int]]:
    rows: list[tuple[int, bytes, int]] = []
    for set_size in range(5, 9):
        for numeric_value in range(1 << set_size):
            bits = tuple(
                (numeric_value >> (set_size - 1 - offset)) & 1
                for offset in range(set_size)
            )
            target = sum(bits) % 2
            typed_input = (1, ODD_INPUT_TAG, ODD_INPUT_SCHEMA_ID, set_size, bits)
            input_hash = _content_hash(CANONICAL_INPUT_DOMAIN, typed_input)
            stratum_id = 1 + 2 * (set_size - 5) + target
            rows.append((len(rows), input_hash, stratum_id))
    if len(rows) != 480:
        raise CalculatorFailure
    return rows


def _sink_rows() -> list[tuple[int, bytes, int]]:
    rows: list[tuple[int, bytes, int]] = []
    for a in range(5):
        for b in range(5):
            for c in range(5):
                for d in range(5):
                    if d != a + b - c:
                        continue
                    typed_input = (
                        1,
                        SINK_INPUT_TAG,
                        SINK_INPUT_SCHEMA_ID,
                        a,
                        b,
                        c,
                        d,
                    )
                    input_hash = _content_hash(CANONICAL_INPUT_DOMAIN, typed_input)
                    rows.append((len(rows), input_hash, 9 + d))
    if len(rows) != 85:
        raise CalculatorFailure
    return rows


def _role_evidence(
    seed: bytearray,
    role_id: int,
    rows: list[tuple[int, bytes, int]],
    quotas: tuple[tuple[int, int, int, int, int], ...],
) -> tuple[object, ...]:
    role_key = _derive_role_key(seed, role_id)
    try:
        by_stratum: dict[int, list[tuple[bytes, bytes, int]]] = {
            quota[0]: [] for quota in quotas
        }
        for universe_index, input_hash, stratum_id in rows:
            rank_digest = _rank(role_key, role_id, stratum_id, input_hash)
            try:
                by_stratum[stratum_id].append((rank_digest, input_hash, universe_index))
            except KeyError as error:
                raise CalculatorFailure from error

        assignments: list[tuple[int, int, bytes, int, int, bytes]] = []
        for stratum_id, universe, discovery, validation, sealed in quotas:
            ranked = sorted(by_stratum[stratum_id], key=lambda row: (row[0], row[1]))
            if len(ranked) != universe or discovery + validation + sealed != universe:
                raise CalculatorFailure
            if any(
                left[:2] == right[:2] and left[2] != right[2]
                for left, right in zip(ranked, ranked[1:])
            ):
                raise CalculatorFailure
            for position, (rank_digest, input_hash, universe_index) in enumerate(ranked):
                if position < discovery:
                    partition_id = DISCOVERY_PARTITION_ID
                elif position < discovery + validation:
                    partition_id = VALIDATION_PARTITION_ID
                else:
                    partition_id = SEALED_PARTITION_ID
                assignments.append(
                    (
                        role_id,
                        universe_index,
                        input_hash,
                        stratum_id,
                        partition_id,
                        rank_digest,
                    )
                )

        assignments.sort(key=lambda row: row[1])
        if len(assignments) != len(rows) or [row[1] for row in assignments] != list(
            range(len(rows))
        ):
            raise CalculatorFailure

        counts: list[int] = []
        roots: list[bytes] = []
        for partition_id in (
            DISCOVERY_PARTITION_ID,
            VALIDATION_PARTITION_ID,
            SEALED_PARTITION_ID,
        ):
            formal_rows = [
                (
                    1,
                    SPLIT_ASSIGNMENT_ROW_TAG,
                    SPLIT_ASSIGNMENT_ROW_SCHEMA_ID,
                    role,
                    universe_index,
                    input_hash,
                    stratum_id,
                    assigned_partition,
                    rank_digest,
                )
                for (
                    role,
                    universe_index,
                    input_hash,
                    stratum_id,
                    assigned_partition,
                    rank_digest,
                ) in assignments
                if assigned_partition == partition_id
            ]
            counts.append(len(formal_rows))
            roots.append(_rfc6962_root(formal_rows))

        expected_counts = tuple(
            sum(quota[index] for quota in quotas) for index in (2, 3, 4)
        )
        if tuple(counts) != expected_counts or sum(counts) != len(rows):
            raise CalculatorFailure
        return (
            role_id,
            len(rows),
            quotas,
            counts[0],
            counts[1],
            counts[2],
            roots[0],
            roots[1],
            roots[2],
        )
    finally:
        _zeroize(role_key)


def _payload(seed: bytearray) -> bytes:
    commitment = hashlib.sha256(SEED_COMMITMENT_PREFIX + b"\x00" + seed).digest()
    odd = _role_evidence(seed, OUTSIDE_ROLE_ID, _odd_rows(), ODD_QUOTAS)
    sink = _role_evidence(seed, NULL_CONTROL_ROLE_ID, _sink_rows(), SINK_QUOTAS)
    partitions = (
        (OUTSIDE_ROLE_ID, DISCOVERY_PARTITION_ID, odd[3], odd[6]),
        (OUTSIDE_ROLE_ID, VALIDATION_PARTITION_ID, odd[4], odd[7]),
        (OUTSIDE_ROLE_ID, SEALED_PARTITION_ID, odd[5], odd[8]),
        (NULL_CONTROL_ROLE_ID, DISCOVERY_PARTITION_ID, sink[3], sink[6]),
        (NULL_CONTROL_ROLE_ID, VALIDATION_PARTITION_ID, sink[4], sink[7]),
        (NULL_CONTROL_ROLE_ID, SEALED_PARTITION_ID, sink[5], sink[8]),
    )
    return _cbor((1, RESPONSE_SCHEMA_ID, commitment, partitions))


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        try:
            written = os.write(fd, view[offset:])
        except OSError as error:
            raise CalculatorFailure from error
        if written <= 0:
            raise CalculatorFailure
        offset += written


def _write_response(payload: bytes) -> None:
    _require_fifo(PUBLIC_RESPONSE_FD)
    frame = len(payload).to_bytes(8, "big") + payload
    try:
        _write_all(PUBLIC_RESPONSE_FD, frame)
    finally:
        try:
            os.close(PUBLIC_RESPONSE_FD)
        except OSError:
            pass


def _run() -> None:
    if len(sys.argv) != 1:
        raise CalculatorFailure
    seed, locked = _read_seed()
    try:
        payload = _payload(seed)
    finally:
        _zeroize(seed)
        _try_munlock(seed, locked)
    _write_response(payload)


def main() -> int:
    try:
        _run()
    except BaseException:
        try:
            os.close(PUBLIC_RESPONSE_FD)
        except OSError:
            pass
        return FAILURE_EXIT_STATUS
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
