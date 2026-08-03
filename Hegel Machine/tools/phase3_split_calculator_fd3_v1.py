#!/usr/bin/env python3
"""One-shot, dependency-free Phase-3 split-seed commitment calculator.

Secret input is accepted only from an inherited FIFO at raw file descriptor 3.
The calculator reads exactly 32 bytes followed by EOF.  It emits one public,
strict-canonical-CBOR response frame to an inherited FIFO at file descriptor 5:

    uint64_be(payload_length) || canonical_cbor([
        1,
        b"hegel-phase3-split-calculator-fd3-response/1",
        commitment_sha256_32_bytes,
    ])

No command-line, environment, stdin, or filesystem secret fallback exists.
Failures are deliberately silent and return a nonzero status.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import stat
import sys


SEED_FD = 3
PUBLIC_RESPONSE_FD = 5
SEED_SIZE = 32
SEED_COMMITMENT_DOMAIN = b"HEGEL/SPLIT_MASTER_SEED_COMMITMENT/V1"
RESPONSE_SCHEMA_ID = b"hegel-phase3-split-calculator-fd3-response/1"
FAILURE_EXIT_STATUS = 70


class CalculatorFailure(Exception):
    """Internal silent failure marker."""


def _try_mlock(secret: bytearray) -> bool:
    """Best-effort POSIX page lock; unsupported/denied attempts stay silent."""

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
    """Best-effort release after zeroization; never turn cleanup into output."""

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
    seed_locked = _try_mlock(seed)
    offset = 0
    try:
        while offset < SEED_SIZE:
            count = os.readv(SEED_FD, (memoryview(seed)[offset:],))
            if count == 0:
                raise CalculatorFailure
            offset += count
        extra = bytearray(1)
        extra_count = os.readv(SEED_FD, (extra,))
        _zeroize(extra)
        if extra_count != 0:
            raise CalculatorFailure
        return seed, seed_locked
    except CalculatorFailure:
        _zeroize(seed)
        _try_munlock(seed, seed_locked)
        raise
    except OSError as error:
        _zeroize(seed)
        _try_munlock(seed, seed_locked)
        raise CalculatorFailure from error
    finally:
        try:
            os.close(SEED_FD)
        except OSError:
            pass


def _cbor_byte_string(value: bytes) -> bytes:
    length = len(value)
    if length < 24:
        return bytes((0x40 | length,)) + value
    if length <= 0xFF:
        return bytes((0x58, length)) + value
    raise CalculatorFailure


def _public_payload(commitment: bytes) -> bytes:
    if len(commitment) != 32:
        raise CalculatorFailure
    # Definite three-element array, shortest-form uint 1, and two byte strings.
    return (
        b"\x83\x01"
        + _cbor_byte_string(RESPONSE_SCHEMA_ID)
        + _cbor_byte_string(commitment)
    )


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


def _write_response(commitment: bytes) -> None:
    _require_fifo(PUBLIC_RESPONSE_FD)
    payload = _public_payload(commitment)
    frame = len(payload).to_bytes(8, "big") + payload
    try:
        _write_all(PUBLIC_RESPONSE_FD, frame)
    finally:
        try:
            os.close(PUBLIC_RESPONSE_FD)
        except OSError:
            pass


def _zeroize(seed: bytearray) -> None:
    for index in range(len(seed)):
        seed[index] = 0


def _run() -> None:
    # Arguments are inspected only to reject every nonempty user argument.
    if len(sys.argv) != 1:
        raise CalculatorFailure
    seed, seed_locked = _read_seed()
    try:
        hasher = hashlib.sha256()
        hasher.update(SEED_COMMITMENT_DOMAIN)
        hasher.update(b"\x00")
        hasher.update(seed)
        commitment = hasher.digest()
    finally:
        _zeroize(seed)
        _try_munlock(seed, seed_locked)
    _write_response(commitment)


def main() -> int:
    try:
        _run()
    except BaseException:
        # The public channel is all-or-nothing.  Do not serialize exception
        # text because it could accidentally include secret-bearing state.
        try:
            os.close(PUBLIC_RESPONSE_FD)
        except OSError:
            pass
        return FAILURE_EXIT_STATUS
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
