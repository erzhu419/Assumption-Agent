#!/usr/bin/env python3
"""One-shot Purpose-5 Ed25519 signing worker.

``--startup-smoke`` validates only the frozen invocation schema, process
identity, the public signer parameters, and the exact local OpenSSL executable
metadata.  It performs no filesystem input read and starts no subprocess.

``sign`` is intended to run only after a future Q6 controller has durably
published that signer's one-shot intent.  It hands the fixed Q5a custody PEM to
one ``openssl pkeyutl -sign -rawin`` invocation, then publishes exactly one
64-byte signature and one public canonical-JSON receipt.  The future controller
performs the independent producer-free Ed25519 verification.  This worker never
generates, imports, exports, or copies a private key and has no target or network
action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from typing import Final, NoReturn, Sequence


PURPOSE_ID: Final = 5
KEY_EPOCH: Final = 1
OPENSSL: Final = Path("/usr/bin/openssl")
PRIVATE_KEY_NAME: Final = "ed25519-private.pem"
PUBLIC_RECEIPT_SCHEMA: Final = (
    "hegel-phase3-q6-purpose5-signing-public-receipt/1"
)
STARTUP_SMOKE_SCHEMA: Final = (
    "hegel-phase3-q6-purpose5-signing-startup-smoke-receipt/1"
)
SIGNER_ROLE_BY_ORDINAL: Final = {
    1: "FINAL_CERTIFICATE_CUSTODIAN",
    2: "FINAL_CERTIFICATE_PYTHON_REPLAY_ATTESTER",
    3: "FINAL_CERTIFICATE_RUST_REPLAY_ATTESTER",
}
MAX_PRIVATE_KEY_BYTES: Final = 16 * 1024
MAX_MESSAGE_BYTES: Final = 1024 * 1024
SIGNATURE_BYTES: Final = 64
OPENSSL_TIMEOUT_SECONDS: Final = 120

FAIL_ARGUMENT_SCHEMA: Final = "FAIL_Q6_SIGNING_ARGUMENT_SCHEMA"
FAIL_IDENTITY: Final = "FAIL_Q6_SIGNING_IDENTITY"
FAIL_OPENSSL: Final = "FAIL_Q6_SIGNING_OPENSSL"
FAIL_PATH: Final = "FAIL_Q6_SIGNING_PATH"
FAIL_OUTPUT_PARENT: Final = "FAIL_Q6_SIGNING_OUTPUT_PARENT"
FAIL_OUTPUT_NOT_FRESH: Final = "FAIL_Q6_SIGNING_OUTPUT_NOT_FRESH"
FAIL_CUSTODY_ROOT: Final = "FAIL_Q6_SIGNING_CUSTODY_ROOT"
FAIL_PRIVATE_KEY: Final = "FAIL_Q6_SIGNING_PRIVATE_KEY"
FAIL_PUBLIC_BINDING: Final = "FAIL_Q6_SIGNING_PUBLIC_BINDING"
FAIL_MESSAGE: Final = "FAIL_Q6_SIGNING_MESSAGE"
FAIL_OPENSSL_SIGN: Final = "FAIL_Q6_SIGNING_OPENSSL_SIGN"
FAIL_SIGNATURE_SHAPE: Final = "FAIL_Q6_SIGNING_SIGNATURE_SHAPE"
FAIL_TRANSIENT_PUBLICATION: Final = "FAIL_Q6_SIGNING_TRANSIENT_PUBLICATION"
FAIL_SIGNATURE_PUBLICATION: Final = "FAIL_Q6_SIGNING_SIGNATURE_PUBLICATION"
FAIL_RECEIPT_PUBLICATION: Final = "FAIL_Q6_SIGNING_RECEIPT_PUBLICATION"
FAIL_UNEXPECTED: Final = "FAIL_Q6_SIGNING_UNEXPECTED"


class WorkerFailure(RuntimeError):
    """A stable worker failure that is safe to expose to the controller."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _StrictArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:  # pragma: no cover - argparse varies
        del message
        raise WorkerFailure(FAIL_ARGUMENT_SCHEMA)


@dataclass(frozen=True, slots=True)
class WorkerArgumentsV1:
    operation: str
    signer_ordinal: int
    signer_role: str
    expected_uid: int
    expected_gid: int
    expected_public_key: bytes
    expected_key_id: bytes
    expected_message_raw_sha256: str
    expected_message_byte_count: int
    custody_root: Path | None
    message_path: Path | None
    signature_path: Path | None
    receipt_path: Path | None


def _strict_uint(text: str) -> int:
    if re.fullmatch(r"0|[1-9][0-9]*", text) is None:
        raise argparse.ArgumentTypeError("not a canonical uint")
    value = int(text)
    if value > (1 << 32) - 1:
        raise argparse.ArgumentTypeError("uint exceeds uint32")
    return value


def _strict_hex(text: str, byte_count: int) -> bytes:
    if re.fullmatch(rf"[0-9a-f]{{{byte_count * 2}}}", text) is None:
        raise argparse.ArgumentTypeError("not canonical lowercase hex")
    return bytes.fromhex(text)


def _strict_public_key_hex(text: str) -> bytes:
    return _strict_hex(text, 32)


def _strict_key_id_hex(text: str) -> bytes:
    return _strict_hex(text, 16)


def _strict_tagged_sha256(text: str) -> str:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
        raise argparse.ArgumentTypeError("not one tagged lowercase SHA-256")
    return text


def _canonical_absolute_path(raw: str) -> Path:
    if (
        type(raw) is not str
        or not raw
        or not raw.startswith("/")
        or raw != os.path.normpath(raw)
        or raw == "/"
    ):
        raise WorkerFailure(FAIL_PATH)
    path = Path(raw)
    try:
        if path.resolve(strict=False) != path:
            raise WorkerFailure(FAIL_PATH)
    except OSError as error:
        raise WorkerFailure(FAIL_PATH) from error
    return path


def parse_arguments_v1(argv: Sequence[str]) -> WorkerArgumentsV1:
    parser = _StrictArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("operation", nargs="?")
    parser.add_argument("--startup-smoke", action="store_true")
    parser.add_argument("--signer-ordinal", type=_strict_uint, required=True)
    parser.add_argument("--signer-role", required=True)
    parser.add_argument("--expected-uid", type=_strict_uint, required=True)
    parser.add_argument("--expected-gid", type=_strict_uint, required=True)
    parser.add_argument(
        "--expected-public-key-hex", type=_strict_public_key_hex, required=True
    )
    parser.add_argument(
        "--expected-key-id-hex", type=_strict_key_id_hex, required=True
    )
    parser.add_argument(
        "--expected-message-raw-sha256", type=_strict_tagged_sha256, required=True
    )
    parser.add_argument(
        "--expected-message-byte-count", type=_strict_uint, required=True
    )
    parser.add_argument("--custody-root")
    parser.add_argument("--message-path")
    parser.add_argument("--signature-path")
    parser.add_argument("--receipt-path")
    namespace = parser.parse_args(list(argv))

    if namespace.signer_ordinal not in SIGNER_ROLE_BY_ORDINAL:
        raise WorkerFailure(FAIL_ARGUMENT_SCHEMA)
    if SIGNER_ROLE_BY_ORDINAL[namespace.signer_ordinal] != namespace.signer_role:
        raise WorkerFailure(FAIL_ARGUMENT_SCHEMA)
    if not 1 <= namespace.expected_message_byte_count <= MAX_MESSAGE_BYTES:
        raise WorkerFailure(FAIL_ARGUMENT_SCHEMA)

    raw_paths = (
        namespace.custody_root,
        namespace.message_path,
        namespace.signature_path,
        namespace.receipt_path,
    )
    if namespace.startup_smoke:
        if namespace.operation is not None or any(item is not None for item in raw_paths):
            raise WorkerFailure(FAIL_ARGUMENT_SCHEMA)
        operation = "startup-smoke"
        paths: tuple[Path | None, Path | None, Path | None, Path | None] = (
            None,
            None,
            None,
            None,
        )
    else:
        if namespace.operation != "sign" or any(item is None for item in raw_paths):
            raise WorkerFailure(FAIL_ARGUMENT_SCHEMA)
        operation = "sign"
        paths = tuple(  # type: ignore[assignment]
            _canonical_absolute_path(item) for item in raw_paths  # type: ignore[arg-type]
        )

    return WorkerArgumentsV1(
        operation=operation,
        signer_ordinal=namespace.signer_ordinal,
        signer_role=namespace.signer_role,
        expected_uid=namespace.expected_uid,
        expected_gid=namespace.expected_gid,
        expected_public_key=namespace.expected_public_key_hex,
        expected_key_id=namespace.expected_key_id_hex,
        expected_message_raw_sha256=namespace.expected_message_raw_sha256,
        expected_message_byte_count=namespace.expected_message_byte_count,
        custody_root=paths[0],
        message_path=paths[1],
        signature_path=paths[2],
        receipt_path=paths[3],
    )


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def _raw_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_runtime_identity(expected_uid: int, expected_gid: int) -> None:
    if os.getuid() != expected_uid or os.getgid() != expected_gid:
        raise WorkerFailure(FAIL_IDENTITY)


def _require_openssl_v1(path: Path = OPENSSL) -> Path:
    try:
        if not path.is_absolute() or path != OPENSSL:
            raise WorkerFailure(FAIL_OPENSSL)
        lexical = path.lstat()
        resolved = path.resolve(strict=True)
        resolved_metadata = resolved.lstat()
        if (
            stat.S_ISLNK(lexical.st_mode)
            or not stat.S_ISREG(lexical.st_mode)
            or resolved != path
            or not stat.S_ISREG(resolved_metadata.st_mode)
            or not os.access(resolved, os.X_OK)
        ):
            raise WorkerFailure(FAIL_OPENSSL)
        return resolved
    except WorkerFailure:
        raise
    except OSError as error:
        raise WorkerFailure(FAIL_OPENSSL) from error


def _validate_public_binding_v1(public_key: bytes, key_id: bytes) -> bytes:
    if (
        type(public_key) is not bytes
        or len(public_key) != 32
        or type(key_id) is not bytes
        or len(key_id) != 16
        or hashlib.sha256(public_key).digest()[:16] != key_id
    ):
        raise WorkerFailure(FAIL_PUBLIC_BINDING)
    return public_key


def _read_regular_v1(
    path: Path,
    *,
    expected_uid: int,
    expected_gid: int,
    maximum_bytes: int,
    failure_code: str,
) -> bytes:
    descriptor = -1
    try:
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != expected_uid
            or metadata.st_gid != expected_gid
            or metadata.st_nlink != 1
            or metadata.st_size < 1
            or metadata.st_size > maximum_bytes
        ):
            raise OSError("input metadata differs")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        payload = bytearray()
        while len(payload) <= maximum_bytes:
            chunk = os.read(descriptor, min(65536, maximum_bytes + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) != metadata.st_size or len(payload) > maximum_bytes:
            raise OSError("input byte count differs")
        return bytes(payload)
    except OSError as error:
        raise WorkerFailure(failure_code) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _require_private_key_metadata_v1(
    custody_root: Path, *, expected_uid: int, expected_gid: int
) -> Path:
    private_key = custody_root / PRIVATE_KEY_NAME
    try:
        root = custody_root.lstat()
        key = private_key.lstat()
        if (
            stat.S_ISLNK(root.st_mode)
            or not stat.S_ISDIR(root.st_mode)
            or stat.S_IMODE(root.st_mode) != 0o700
            or root.st_uid != expected_uid
            or root.st_gid != expected_gid
            or custody_root.resolve(strict=True) != custody_root
        ):
            raise WorkerFailure(FAIL_CUSTODY_ROOT)
        if (
            stat.S_ISLNK(key.st_mode)
            or not stat.S_ISREG(key.st_mode)
            or stat.S_IMODE(key.st_mode) != 0o600
            or key.st_uid != expected_uid
            or key.st_gid != expected_gid
            or key.st_nlink != 1
            or key.st_size < 1
            or key.st_size > MAX_PRIVATE_KEY_BYTES
        ):
            raise WorkerFailure(FAIL_PRIVATE_KEY)
        return private_key
    except WorkerFailure:
        raise
    except OSError as error:
        raise WorkerFailure(FAIL_PRIVATE_KEY) from error


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _require_output_paths_v1(
    signature_path: Path,
    receipt_path: Path,
    custody_root: Path,
    *,
    expected_uid: int,
    expected_gid: int,
) -> Path:
    signature_tmp = signature_path.with_name(signature_path.name + ".q6-tmp")
    candidates = (signature_path, receipt_path, signature_tmp)
    if len(set(candidates)) != len(candidates):
        raise WorkerFailure(FAIL_PATH)
    if any(_is_within(path, custody_root) for path in candidates):
        raise WorkerFailure(FAIL_PATH)
    for path in candidates:
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        except OSError as error:
            raise WorkerFailure(FAIL_OUTPUT_NOT_FRESH) from error
        else:
            raise WorkerFailure(FAIL_OUTPUT_NOT_FRESH)
    for parent in {signature_path.parent, receipt_path.parent}:
        try:
            metadata = parent.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != expected_uid
                or metadata.st_gid != expected_gid
                or parent.resolve(strict=True) != parent
                or not os.access(parent, os.W_OK | os.X_OK)
            ):
                raise OSError("output parent metadata differs")
        except OSError as error:
            raise WorkerFailure(FAIL_OUTPUT_PARENT) from error
    return signature_tmp


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_bytes(path: Path, payload: bytes, mode: int, failure_code: str) -> None:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short write")
            offset += written
        os.fsync(descriptor)
    except OSError as error:
        raise WorkerFailure(failure_code) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _run_openssl_v1(
    arguments: list[str], *, working_directory: Path, failure_code: str
) -> bytes:
    try:
        completed = subprocess.run(
            [str(OPENSSL), *arguments],
            cwd=working_directory,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=OPENSSL_TIMEOUT_SECONDS,
            umask=0o077,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise WorkerFailure(failure_code) from error
    if completed.returncode != 0:
        raise WorkerFailure(failure_code)
    return completed.stdout


def _publish_signature_v1(signature_tmp: Path, signature_path: Path) -> None:
    descriptor = -1
    try:
        descriptor = os.open(
            signature_tmp, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != SIGNATURE_BYTES
            or metadata.st_nlink != 1
        ):
            raise OSError("signature temporary metadata differs")
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
    except OSError as error:
        raise WorkerFailure(FAIL_SIGNATURE_PUBLICATION) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        os.replace(signature_tmp, signature_path)
        _fsync_directory(signature_path.parent)
        metadata = signature_path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o644
            or metadata.st_size != SIGNATURE_BYTES
        ):
            raise OSError("published signature metadata differs")
    except OSError as error:
        raise WorkerFailure(FAIL_SIGNATURE_PUBLICATION) from error


def build_public_receipt_v1(
    *,
    arguments: WorkerArgumentsV1,
    message: bytes,
    signature: bytes,
) -> dict[str, object]:
    checked_public_key = _validate_public_binding_v1(
        arguments.expected_public_key, arguments.expected_key_id
    )
    if (
        arguments.operation != "sign"
        or checked_public_key != arguments.expected_public_key
        or type(message) is not bytes
        or len(message) != arguments.expected_message_byte_count
        or _raw_sha256(message) != arguments.expected_message_raw_sha256
        or type(signature) is not bytes
        or len(signature) != SIGNATURE_BYTES
    ):
        raise WorkerFailure(FAIL_ARGUMENT_SCHEMA)
    return {
        "key_epoch": KEY_EPOCH,
        "key_generation_count": 0,
        "key_id_hex": arguments.expected_key_id.hex(),
        "key_import_count": 0,
        "message_byte_count": len(message),
        "message_raw_sha256": _raw_sha256(message),
        "openssl_path": str(OPENSSL),
        "openssl_popen_count": 1,
        "private_key_copy_count": 0,
        "private_key_exported": False,
        "private_key_openssl_use_count": 1,
        "public_key_hex": arguments.expected_public_key.hex(),
        "purpose_id": PURPOSE_ID,
        "schema": PUBLIC_RECEIPT_SCHEMA,
        "signature_byte_count": len(signature),
        "signature_count": 1,
        "signature_raw_sha256": _raw_sha256(signature),
        "signature_verification_count": 0,
        "signer_ordinal": arguments.signer_ordinal,
        "signer_role": arguments.signer_role,
        "status": "PURPOSE5_SIGNATURE_COMPLETE_PENDING_PRODUCER_FREE_VERIFICATION",
    }


def _startup_smoke_receipt(arguments: WorkerArgumentsV1) -> dict[str, object]:
    _validate_public_binding_v1(
        arguments.expected_public_key, arguments.expected_key_id
    )
    return {
        "expected_gid": arguments.expected_gid,
        "expected_key_id_hex": arguments.expected_key_id.hex(),
        "expected_message_byte_count": arguments.expected_message_byte_count,
        "expected_message_raw_sha256": arguments.expected_message_raw_sha256,
        "expected_public_key_hex": arguments.expected_public_key.hex(),
        "expected_uid": arguments.expected_uid,
        "key_generation_count": 0,
        "key_import_count": 0,
        "openssl_path": str(OPENSSL),
        "openssl_popen_count": 0,
        "private_key_access_count": 0,
        "purpose_id": PURPOSE_ID,
        "schema": STARTUP_SMOKE_SCHEMA,
        "signature_count": 0,
        "signer_ordinal": arguments.signer_ordinal,
        "signer_role": arguments.signer_role,
        "status": "Q6_PURPOSE5_SIGNING_STARTUP_SMOKE_PASS",
    }


def sign_v1(arguments: WorkerArgumentsV1) -> dict[str, object]:
    if (
        arguments.operation != "sign"
        or arguments.signer_ordinal not in SIGNER_ROLE_BY_ORDINAL
        or SIGNER_ROLE_BY_ORDINAL[arguments.signer_ordinal]
        != arguments.signer_role
        or arguments.custody_root is None
        or arguments.message_path is None
        or arguments.signature_path is None
        or arguments.receipt_path is None
    ):
        raise WorkerFailure(FAIL_ARGUMENT_SCHEMA)

    _validate_public_binding_v1(
        arguments.expected_public_key, arguments.expected_key_id
    )
    signature_tmp = _require_output_paths_v1(
        arguments.signature_path,
        arguments.receipt_path,
        arguments.custody_root,
        expected_uid=arguments.expected_uid,
        expected_gid=arguments.expected_gid,
    )
    message = _read_regular_v1(
        arguments.message_path,
        expected_uid=arguments.expected_uid,
        expected_gid=arguments.expected_gid,
        maximum_bytes=MAX_MESSAGE_BYTES,
        failure_code=FAIL_MESSAGE,
    )
    if (
        len(message) != arguments.expected_message_byte_count
        or _raw_sha256(message) != arguments.expected_message_raw_sha256
    ):
        raise WorkerFailure(FAIL_MESSAGE)
    private_key = _require_private_key_metadata_v1(
        arguments.custody_root,
        expected_uid=arguments.expected_uid,
        expected_gid=arguments.expected_gid,
    )

    signature = _run_openssl_v1(
        [
            "pkeyutl",
            "-sign",
            "-rawin",
            "-inkey",
            str(private_key),
            "-in",
            str(arguments.message_path),
        ],
        working_directory=arguments.signature_path.parent,
        failure_code=FAIL_OPENSSL_SIGN,
    )
    if len(signature) != SIGNATURE_BYTES:
        raise WorkerFailure(FAIL_SIGNATURE_SHAPE)
    _exclusive_bytes(
        signature_tmp, signature, 0o600, FAIL_TRANSIENT_PUBLICATION
    )
    _fsync_directory(signature_tmp.parent)
    _publish_signature_v1(signature_tmp, arguments.signature_path)
    receipt = build_public_receipt_v1(
        arguments=arguments,
        message=message,
        signature=signature,
    )
    _exclusive_bytes(
        arguments.receipt_path,
        _canonical_json(receipt),
        0o644,
        FAIL_RECEIPT_PUBLICATION,
    )
    _fsync_directory(arguments.receipt_path.parent)
    return receipt


def _emit_public_json(value: object) -> None:
    sys.stdout.write(_canonical_json(value).decode("ascii"))
    sys.stdout.flush()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = parse_arguments_v1(sys.argv[1:] if argv is None else argv)
        _require_runtime_identity(arguments.expected_uid, arguments.expected_gid)
        _require_openssl_v1()
        if arguments.operation == "startup-smoke":
            receipt = _startup_smoke_receipt(arguments)
        else:
            receipt = sign_v1(arguments)
        _emit_public_json(receipt)
        return 0
    except WorkerFailure as error:
        sys.stderr.write(error.code + "\n")
        return 70
    except Exception:
        sys.stderr.write(FAIL_UNEXPECTED + "\n")
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
