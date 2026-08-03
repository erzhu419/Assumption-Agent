#!/usr/bin/env python3
"""Keyless, read-only verifier for the retained M2.5 split seed custody.

The verifier is executed in a one-shot offline container with no actor state
volume.  It reads the raw seed only inside that container, recomputes the
frozen commitment, checks the durable intent/completion receipt, zeroes its
buffer, and emits only one bounded public commitment receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys


CUSTODY = Path("/custody")
STATE_MOUNT = Path("/state")
SEED_INTENT = CUSTODY / "split_seed_generation.intent"
SEED_FILE = CUSTODY / "split_master_seed.bin"
SEED_COMPLETE = CUSTODY / "split_seed_generation.complete"
SEED_INTENT_PAYLOAD = (
    b'{"schema":"hegel-phase3-m25-seed-generation-intent/1",'
    b'"state":"CSPRNG_CALL_COMMITTED_NO_REDRAW"}\n'
)
SEED_COMMITMENT_PREFIX = b"HEGEL/SPLIT_MASTER_SEED_COMMITMENT/V1\x00"
SCHEMA = "hegel-phase3-m25-keyless-seed-custody-inner-verification/1"


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def fail() -> "None":
    try:
        sys.stderr.write("FAIL_M25_KEYLESS_SEED_CUSTODY_VERIFICATION\n")
    finally:
        raise SystemExit(70)


def exact_regular(path: Path, mode: int, uid: int, gid: int) -> bytes:
    if path.is_symlink():
        fail()
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != mode
        or metadata.st_uid != uid
        or metadata.st_gid != gid
    ):
        fail()
    return path.read_bytes()


def main() -> int:
    expected = os.environ.get("HEGEL_EXPECTED_SEED_COMMITMENT_HEX")
    expected_uid = os.environ.get("HEGEL_VERIFIER_NUMERIC_UID")
    expected_gid = os.environ.get("HEGEL_VERIFIER_NUMERIC_GID")
    if (
        expected is None
        or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        or expected_uid is None
        or re.fullmatch(r"0|[1-9][0-9]*", expected_uid) is None
        or expected_gid is None
        or re.fullmatch(r"0|[1-9][0-9]*", expected_gid) is None
    ):
        fail()
    uid = int(expected_uid)
    gid = int(expected_gid)
    custody_metadata = CUSTODY.lstat()
    if (
        STATE_MOUNT.exists()
        or STATE_MOUNT.is_symlink()
        or os.geteuid() != uid
        or os.getegid() != gid
        or not stat.S_ISDIR(custody_metadata.st_mode)
        or stat.S_IMODE(custody_metadata.st_mode) != 0o700
        or custody_metadata.st_uid != uid
        or custody_metadata.st_gid != gid
    ):
        fail()
    intent = exact_regular(SEED_INTENT, 0o600, uid, gid)
    completion = exact_regular(SEED_COMPLETE, 0o600, uid, gid)
    seed_bytes = exact_regular(SEED_FILE, 0o600, uid, gid)
    if intent != SEED_INTENT_PAYLOAD or len(seed_bytes) != 32:
        fail()
    seed = bytearray(seed_bytes)
    del seed_bytes
    try:
        commitment = hashlib.sha256(SEED_COMMITMENT_PREFIX + bytes(seed)).hexdigest()
        try:
            completion_fields = json.loads(completion)
        except (UnicodeDecodeError, json.JSONDecodeError):
            fail()
        expected_completion = {
            "attempt": 1,
            "intent_sha256": hashlib.sha256(intent).hexdigest(),
            "schema": "hegel-phase3-m25-seed-generation-complete/1",
            "seed_commitment_hex": commitment,
            "seed_length_bytes": 32,
        }
        if (
            type(completion_fields) is not dict
            or completion != canonical_json(completion_fields)
            or completion_fields != expected_completion
            or commitment != expected
        ):
            fail()
        receipt: dict[str, object] = {
            "schema": SCHEMA,
            "verified": True,
            "seed_commitment_hex": commitment,
            "seed_length_bytes": 32,
            "seed_intent_sha256": hashlib.sha256(intent).hexdigest(),
            "completion_receipt_sha256": hashlib.sha256(completion).hexdigest(),
            "raw_seed_read_inside_keyless_verifier": True,
            "raw_seed_exported": False,
            "private_key_mount_present": False,
            "state_mount_present": False,
            "verifier_numeric_uid": uid,
            "verifier_numeric_gid": gid,
            "custody_artifacts_owned_by_verifier_identity": True,
        }
        receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
        sys.stdout.buffer.write(canonical_json(receipt))
        return 0
    finally:
        for index in range(len(seed)):
            seed[index] = 0


if __name__ == "__main__":
    raise SystemExit(main())
