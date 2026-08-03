#!/usr/bin/env python3
"""Purpose-1 custodian worker for the owner-authorized M2.5 ceremony.

The worker is intentionally not a general signing service.  Every signing
operation reconstructs the one formal object that the frozen purpose is
allowed to attest, derives the exact domain-separated preimage locally, and
only then asks OpenSSL to use the private key held in the purpose-private
``/state`` volume.  The persisted ``/custody`` bind is purpose-1-only and may
contain only the transaction metadata, split marker, generation intent,
completion receipt and raw 32-byte split seed.

Success output is public.  No seed, private key, role key, rank, or partition
membership is written to stdout/stderr or ``/output``.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types
from typing import Mapping

import phase3_m25_actor_operation_probe_v1 as operation_probe


STATE = Path("/state")
INPUT = Path("/input")
OUTPUT = Path("/output")
CUSTODY = Path("/custody")
PRIVATE_KEY = STATE / "ed25519-private.pem"
PUBLIC_DER = OUTPUT / "ed25519-public.der"
SIGNATURE = OUTPUT / "ed25519-signature.bin"
BRIDGE_DAG_PACKAGE = INPUT / "bridge-dag-package.cbor"
BRIDGE_REPLAY_RECEIPT = OUTPUT / "bridge-dag-replay-receipt.json"
SEED_FILE = CUSTODY / "split_master_seed.bin"
SEED_INTENT = CUSTODY / "split_seed_generation.intent"
SEED_COMPLETE = CUSTODY / "split_seed_generation.complete"
MARKER = CUSTODY / "split_seed_instantiation.marker"
MAX_PUBLIC_REQUEST_BYTES = 32 * 1024 * 1024
SYNTHETIC_SEED = hashlib.sha256(
    b"HEGEL/M25/NON_AUTHORITATIVE/SYNTHETIC/CEREMONY/SEED/V1"
).digest()
SEED_INTENT_PAYLOAD = (
    b'{"schema":"hegel-phase3-m25-seed-generation-intent/1",'
    b'"state":"CSPRNG_CALL_COMMITTED_NO_REDRAW"}\n'
)
SEED_COMMITMENT_PREFIX = b"HEGEL/SPLIT_MASTER_SEED_COMMITMENT/V1"


class WorkerFailure(RuntimeError):
    pass


def fail(code: str) -> "None":
    # Stable code only: exception details and subprocess diagnostics may carry
    # host paths and therefore never cross the actor boundary.
    try:
        sys.stderr.write(code + "\n")
    finally:
        raise SystemExit(70)


def require_profile(purpose: int) -> None:
    if os.environ.get("HEGEL_ACTOR_PROFILE_ID") != (
        "hegel-owner-accepted-container-technical-actors-v1"
    ):
        fail("FAIL_M25_ACTOR_WORKER_PROFILE")
    if purpose != 1 or os.environ.get("HEGEL_PURPOSE_ID") != "1":
        fail("FAIL_M25_ACTOR_WORKER_PURPOSE")
    if os.getuid() != 65534 or os.getgid() != 65534:
        fail("FAIL_M25_ACTOR_WORKER_IDENTITY")


def exclusive_bytes(path: Path, payload: bytes, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, mode)
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            count = os.write(descriptor, view[offset:])
            if count <= 0:
                raise WorkerFailure
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_public_json(path: Path) -> Mapping[str, object]:
    raw = path.read_bytes()
    if len(raw) > MAX_PUBLIC_REQUEST_BYTES:
        raise WorkerFailure
    value = json.loads(raw)
    if type(value) is not dict:
        raise WorkerFailure
    return value


def run_quiet(command: list[str], *, stdin: bytes | None = None) -> bytes:
    completed = subprocess.run(
        command,
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=120,
        env={
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONPATH": "/input/src",
        },
    )
    if completed.returncode != 0:
        raise WorkerFailure
    return completed.stdout


def keygen(purpose: int, *, recovery: bool = False) -> None:
    require_profile(purpose)
    if PUBLIC_DER.exists():
        raise WorkerFailure
    os.chmod(STATE, 0o700)
    if PRIVATE_KEY.exists():
        metadata = PRIVATE_KEY.stat()
        if (
            PRIVATE_KEY.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise WorkerFailure
    else:
        if recovery:
            fail("FAIL_M25_ACTOR_RECOVERY_KEY_ABSENT")
        run_quiet(
            [
                "/usr/bin/openssl",
                "genpkey",
                "-algorithm",
                "ED25519",
                "-out",
                str(PRIVATE_KEY),
            ]
        )
        os.chmod(PRIVATE_KEY, 0o600)
    der = run_quiet(
        [
            "/usr/bin/openssl",
            "pkey",
            "-in",
            str(PRIVATE_KEY),
            "-pubout",
            "-outform",
            "DER",
        ]
    )
    if len(der) != 44:
        raise WorkerFailure
    exclusive_bytes(PUBLIC_DER, der, 0o644)


def openssl_sign(preimage: bytes) -> None:
    if not PRIVATE_KEY.is_file() or SIGNATURE.exists() or len(preimage) > 4096:
        raise WorkerFailure
    request = INPUT / "signing-preimage.bin"
    if request.read_bytes() != preimage:
        raise WorkerFailure
    signature = run_quiet(
        [
            "/usr/bin/openssl",
            "pkeyutl",
            "-sign",
            "-rawin",
            "-inkey",
            str(PRIVATE_KEY),
            "-in",
            str(request),
        ]
    )
    if len(signature) != 64:
        raise WorkerFailure
    exclusive_bytes(SIGNATURE, signature, 0o644)


def load_wire():
    # Construct a deliberately empty package shell so importing the two-file
    # wire/CBOR closure does not execute the repository's broad ``__init__``.
    # The actor snapshot intentionally does not contain those unrelated files.
    package = sys.modules.get("hegel_machine")
    if package is None:
        package = types.ModuleType("hegel_machine")
        package.__path__ = ["/input/src/hegel_machine"]
        package.__package__ = "hegel_machine"
        sys.modules["hegel_machine"] = package
    return importlib.import_module("hegel_machine.phase3_m25_wire_v1")


def load_bridge_replay():
    load_wire()
    return importlib.import_module(
        "hegel_machine.phase3_m25_bridge_full_dag_replay_v1"
    )


def _bridge_verifier_directory(purpose: int) -> Path:
    nonce = os.environ.get("HEGEL_OPERATION_NONCE", "")
    if re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
        raise WorkerFailure
    return Path("/tmp") / f"hegel-m25-bridge-verifier-p{purpose}-{nonce}"


def _openssl_sign_internal(preimage: bytes, private_directory: Path) -> bytes:
    if (
        not PRIVATE_KEY.is_file()
        or PRIVATE_KEY.is_symlink()
        or SIGNATURE.exists()
        or len(preimage) > 4096
    ):
        raise WorkerFailure
    signing_input = private_directory / "signing-preimage.bin"
    exclusive_bytes(signing_input, preimage, 0o600)
    try:
        signature = run_quiet(
            [
                "/usr/bin/openssl",
                "pkeyutl",
                "-sign",
                "-rawin",
                "-inkey",
                str(PRIVATE_KEY),
                "-in",
                str(signing_input),
            ]
        )
    finally:
        try:
            signing_input.unlink()
        except FileNotFoundError:
            pass
    if len(signature) != 64:
        raise WorkerFailure
    exclusive_bytes(SIGNATURE, signature, 0o644)
    return signature


def authorized_object_sign(purpose: int) -> None:
    """Replay one exact purpose-1 object before signing it."""

    require_profile(purpose)
    if purpose != 1:
        raise WorkerFailure
    wire = load_wire()
    request = read_public_json(INPUT / "authorized-object.json")
    name = request.get("schema_name")
    allowed = {
        "SplitSeedCommitmentManifestV1",
        "CustodianBindingManifestV1",
        "SeedContinuityManifestV1",
        "HiddenAccessLedgerRecordV1",
    }
    if name not in allowed or type(request.get("formal_cbor_hex")) is not str:
        raise WorkerFailure
    payload = bytes.fromhex(request["formal_cbor_hex"])
    decoded = wire.decode_formal_object(payload, expected_name=name)
    if wire.encode_formal_object(name, decoded.fields) != payload:
        raise WorkerFailure
    root = wire.candidate_content_root(name, decoded.fields)
    expected_root = bytes.fromhex(str(request.get("expected_root_hex")))
    if root != expected_root:
        raise WorkerFailure
    tag = wire.OBJECT_TAGS[name]
    preimage = wire.external_signature_preimage_v1(tag, root, 1, 0)
    openssl_sign(preimage)


def python_bridge_replay_sign(purpose: int) -> None:
    require_profile(purpose)
    if purpose != 1:
        raise WorkerFailure
    wire = load_wire()
    replay = load_bridge_replay()
    payload = BRIDGE_DAG_PACKAGE.read_bytes()
    if not payload or len(payload) > MAX_PUBLIC_REQUEST_BYTES:
        raise WorkerFailure
    verifier_directory = _bridge_verifier_directory(purpose)
    os.mkdir(verifier_directory, 0o700)
    try:
        verifier = replay.make_openssl_ed25519_verifier_v1(verifier_directory)
        result = replay.replay_bridge_dag_package_v1(
            payload,
            allow_authoritative=True,
            signature_verifier=verifier,
        )
        if (
            result.purpose_id != purpose
            or not result.authoritative
            or result.purpose1_signature_verified
            or not result.eligible_to_sign_bridge_statement
            or result.split_membership_recomputed
        ):
            raise WorkerFailure
        preimage = wire.bridge_attestation_signature_preimage_v1(
            result.bridge_statement_root, purpose, 0
        )
        signature = _openssl_sign_internal(preimage, verifier_directory)
        exclusive_bytes(
            BRIDGE_REPLAY_RECEIPT,
            replay.build_bridge_actor_replay_receipt_v1(
                result, implementation="python-full-dag-replay-v1"
            ),
            0o644,
        )
    finally:
        try:
            verifier_directory.rmdir()
        except FileNotFoundError:
            pass


def run_fd_calculator(executable: list[str], seed: bytes) -> bytes:
    seed_read, seed_write = os.pipe()
    response_read, response_write = os.pipe()
    devnull = os.open("/dev/null", os.O_RDWR)
    child_sources: list[int] = []
    try:
        # Never use a pipe endpoint itself as a dup2 source.  With the usual
        # descriptor allocation response_read is FD 5; the old action order
        # first replaced FD 5 with response_write and then closed FD 5 under
        # its former name.  Move every child-side source above the wire FDs so
        # the dup/close plan is collision-free for every parent FD layout.
        for descriptor in (devnull, seed_read, response_write):
            child_sources.append(
                fcntl.fcntl(descriptor, fcntl.F_DUPFD_CLOEXEC, 10)
            )
        devnull_source, seed_source, response_source = child_sources

        for descriptor_name, descriptor in (
            ("devnull", devnull),
            ("seed_read", seed_read),
            ("response_write", response_write),
        ):
            os.close(descriptor)
            if descriptor_name == "devnull":
                devnull = -1
            elif descriptor_name == "seed_read":
                seed_read = -1
            else:
                response_write = -1

        actions = [
            (os.POSIX_SPAWN_DUP2, devnull_source, 0),
            (os.POSIX_SPAWN_DUP2, devnull_source, 1),
            (os.POSIX_SPAWN_DUP2, devnull_source, 2),
            (os.POSIX_SPAWN_DUP2, seed_source, 3),
            (os.POSIX_SPAWN_DUP2, response_source, 5),
        ]
        # A parent endpoint can already have number 3 or 5.  In that case the
        # preceding DUP2 has overwritten it in the child, so closing that
        # number would close the required wire endpoint.
        for descriptor in (seed_write, response_read):
            if descriptor not in {0, 1, 2, 3, 5}:
                actions.append((os.POSIX_SPAWN_CLOSE, descriptor))
        for descriptor in child_sources:
            actions.append((os.POSIX_SPAWN_CLOSE, descriptor))

        pid = os.posix_spawn(
            executable[0],
            executable,
            {
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/local/bin:/usr/bin:/bin",
            },
            file_actions=actions,
        )
        for descriptor in child_sources:
            os.close(descriptor)
        child_sources.clear()
        written = 0
        while written < len(seed):
            count = os.write(seed_write, seed[written:])
            if count <= 0:
                raise WorkerFailure
            written += count
        os.close(seed_write)
        seed_write = -1
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(response_read, 4096)
            if not chunk:
                break
            total += len(chunk)
            if total > 2048:
                raise WorkerFailure
            chunks.append(chunk)
        _pid, status = os.waitpid(pid, 0)
        if status != 0:
            raise WorkerFailure
        return b"".join(chunks)
    finally:
        for descriptor in (
            seed_read,
            seed_write,
            response_read,
            response_write,
            devnull,
            *child_sources,
        ):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _seed_completion_payload(seed: bytes) -> bytes:
    value = {
        "attempt": 1,
        "intent_sha256": hashlib.sha256(SEED_INTENT_PAYLOAD).hexdigest(),
        "schema": "hegel-phase3-m25-seed-generation-complete/1",
        "seed_commitment_hex": hashlib.sha256(
            SEED_COMMITMENT_PREFIX + b"\x00" + seed
        ).hexdigest(),
        "seed_length_bytes": 32,
    }
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _load_completed_seed_for_resume() -> bytearray:
    if (
        SEED_INTENT.is_symlink()
        or not SEED_INTENT.is_file()
        or stat.S_IMODE(SEED_INTENT.stat().st_mode) != 0o600
        or SEED_INTENT.read_bytes() != SEED_INTENT_PAYLOAD
        or SEED_FILE.is_symlink()
        or not SEED_FILE.is_file()
        or stat.S_IMODE(SEED_FILE.stat().st_mode) != 0o600
        or SEED_FILE.stat().st_size != 32
        or SEED_COMPLETE.is_symlink()
        or not SEED_COMPLETE.is_file()
        or stat.S_IMODE(SEED_COMPLETE.stat().st_mode) != 0o600
    ):
        fail("FAIL_M25_SPLIT_SEED_UNRECOVERABLE_NO_REDRAW")
    seed = bytearray(SEED_FILE.read_bytes())
    if SEED_COMPLETE.read_bytes() != _seed_completion_payload(bytes(seed)):
        for index in range(len(seed)):
            seed[index] = 0
        fail("FAIL_M25_SPLIT_SEED_UNRECOVERABLE_NO_REDRAW")
    return seed


def _generate_seed_after_durable_intent() -> bytearray:
    """Perform the one permitted CSPRNG call after durable no-redraw intent.

    Callers must first prove that all three seed-state paths are absent.  This
    helper is shared by the uninterrupted first invocation and the explicit
    recovery edge in which the PENDING marker was durable but the worker had
    not yet written its intent.  Once the intent exists, every incomplete or
    malformed state is terminal and this helper is unreachable.
    """

    if SEED_INTENT.exists() or SEED_INTENT.is_symlink():
        fail("FAIL_M25_SPLIT_SEED_UNRECOVERABLE_NO_REDRAW")
    if SEED_FILE.exists() or SEED_FILE.is_symlink():
        fail("FAIL_M25_SPLIT_SEED_UNRECOVERABLE_NO_REDRAW")
    if SEED_COMPLETE.exists() or SEED_COMPLETE.is_symlink():
        fail("FAIL_M25_SPLIT_SEED_UNRECOVERABLE_NO_REDRAW")
    exclusive_bytes(SEED_INTENT, SEED_INTENT_PAYLOAD, 0o600)
    fsync_dir(CUSTODY)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(SEED_FILE, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        fsync_dir(CUSTODY)
        seed = bytearray(os.getrandom(32))
        if len(seed) != 32:
            raise WorkerFailure
        offset = 0
        while offset < len(seed):
            written = os.write(descriptor, seed[offset:])
            if written <= 0:
                raise WorkerFailure
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_dir(CUSTODY)
    # The completion receipt is written only after the seed inode and its
    # directory entry have both crossed an fsync boundary.
    try:
        exclusive_bytes(SEED_COMPLETE, _seed_completion_payload(bytes(seed)), 0o600)
        fsync_dir(CUSTODY)
    except Exception:
        for index in range(len(seed)):
            seed[index] = 0
        raise
    return seed


def seed_and_split(*, synthetic: bool, recovery: bool = False) -> None:
    require_profile(1)
    marker = read_public_json(MARKER)
    if marker.get("state") != "PENDING":
        raise WorkerFailure
    if synthetic:
        if SEED_INTENT.exists() or SEED_FILE.exists() or SEED_COMPLETE.exists() or recovery:
            raise WorkerFailure
        seed = bytearray(SYNTHETIC_SEED)
        split_mode = "SYNTHETIC_NON_AUTHORITATIVE"
    elif recovery:
        # A durable PENDING marker with no intent proves that the worker never
        # reached its CSPRNG call.  Explicit recovery may therefore perform
        # first genesis once.  Any intent-bearing state must instead satisfy
        # the exact completion receipt and is never redrawn.
        if not any(
            path.exists() or path.is_symlink()
            for path in (SEED_INTENT, SEED_FILE, SEED_COMPLETE)
        ):
            seed = _generate_seed_after_durable_intent()
            split_mode = "REAL_FIRST_GENESIS_AFTER_PENDING_NO_INTENT"
        else:
            seed = _load_completed_seed_for_resume()
            split_mode = "REAL_PENDING_RESUME"
    elif not SEED_INTENT.exists() and not SEED_FILE.exists() and not SEED_COMPLETE.exists():
        seed = _generate_seed_after_durable_intent()
        split_mode = "REAL_FIRST_GENESIS"
    else:
        completed = _load_completed_seed_for_resume()
        for index in range(len(completed)):
            completed[index] = 0
        fail("FAIL_M25_SPLIT_SEED_EXPLICIT_RECOVERY_REQUIRED")
    if len(seed) != 32:
        raise WorkerFailure
    try:
        python_frame = run_fd_calculator(
            ["/usr/local/bin/python3", "/input/tools/phase3_split_partition_calculator_fd3_v1.py"],
            bytes(seed),
        )
        rust_frame = run_fd_calculator(
            ["/input/rust-split-calculator"], bytes(seed)
        )
    finally:
        for index in range(len(seed)):
            seed[index] = 0
    if python_frame != rust_frame or not python_frame:
        raise WorkerFailure
    exclusive_bytes(OUTPUT / "python-split-frame.bin", python_frame, 0o644)
    exclusive_bytes(OUTPUT / "rust-split-frame.bin", rust_frame, 0o644)
    exclusive_bytes(
        OUTPUT / "split-mode.txt",
        (split_mode + "\n").encode("ascii"),
        0o644,
    )


def complete_marker() -> None:
    require_profile(1)
    marker = read_public_json(MARKER)
    if marker.get("state") != "PENDING":
        raise WorkerFailure
    root = (INPUT / "seed-manifest-root.bin").read_bytes()
    if len(root) != 32:
        raise WorkerFailure
    complete = dict(marker)
    complete["state"] = "COMPLETE"
    complete["seed_commitment_manifest_root_hex_or_null"] = root.hex()
    payload = (
        json.dumps(complete, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    temporary = CUSTODY / "split_seed_instantiation.marker.complete.tmp"
    exclusive_bytes(temporary, payload, 0o600)
    os.replace(temporary, MARKER)
    fsync_dir(CUSTODY)
    exclusive_bytes(OUTPUT / "complete-marker.json", payload, 0o644)


def main() -> int:
    if len(sys.argv) != 2:
        fail("FAIL_M25_ACTOR_WORKER_ARGUMENTS")
    operation = sys.argv[1]
    try:
        operation_probe.qualify_operation_v1(operation)
        if operation == "qualify-only":
            pass
        elif operation == "keygen":
            keygen(1, recovery=False)
        elif operation == "keygen-resume":
            keygen(1, recovery=True)
        elif operation == "purpose1-authorized-sign":
            authorized_object_sign(1)
        elif operation == "bridge-replay-sign-python":
            python_bridge_replay_sign(1)
        elif operation == "seed-split-real":
            seed_and_split(synthetic=False, recovery=False)
        elif operation == "seed-split-resume":
            seed_and_split(synthetic=False, recovery=True)
        elif operation == "seed-split-synthetic":
            seed_and_split(synthetic=True, recovery=False)
        elif operation == "complete-marker":
            complete_marker()
        else:
            raise WorkerFailure
        return 0
    except (Exception, SystemExit):
        fail("FAIL_M25_ACTOR_WORKER_OPERATION")
    return 70


if __name__ == "__main__":
    raise SystemExit(main())
