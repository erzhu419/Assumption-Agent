#!/usr/bin/env python3
"""One-shot Purpose-5 Ed25519 key-generation worker.

``--startup-smoke`` validates only the frozen invocation schema, process
identity, and the exact local OpenSSL executable.  It neither starts OpenSSL
nor creates a filesystem object.

``generate`` creates one fresh signer key.  The private PEM remains only in
the fresh mode-0700 state root; the worker publishes only the raw public-key
DER and one canonical public JSON receipt.  This worker has no signing or
network operation.
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
ED25519_SPKI_PREFIX: Final = bytes.fromhex("302a300506032b6570032100")
PRIVATE_KEY_NAME: Final = "ed25519-private.pem"
PUBLIC_DER_NAME: Final = "ed25519-public.der"
PUBLIC_RECEIPT_NAME: Final = "purpose5-keygen-public-receipt.json"
PUBLIC_RECEIPT_SCHEMA: Final = (
    "hegel-phase3-q5a-purpose5-keygen-public-receipt/1"
)
STARTUP_SMOKE_SCHEMA: Final = (
    "hegel-phase3-q5a-purpose5-keygen-startup-smoke-receipt/1"
)
SIGNER_ROLE_BY_ORDINAL: Final = {
    1: "FINAL_CERTIFICATE_CUSTODIAN",
    2: "FINAL_CERTIFICATE_PYTHON_REPLAY_ATTESTER",
    3: "FINAL_CERTIFICATE_RUST_REPLAY_ATTESTER",
}
MAX_PRIVATE_KEY_BYTES: Final = 16 * 1024
OPENSSL_TIMEOUT_SECONDS: Final = 120

FAIL_ARGUMENT_SCHEMA: Final = "FAIL_Q5A_KEYGEN_ARGUMENT_SCHEMA"
FAIL_IDENTITY: Final = "FAIL_Q5A_KEYGEN_IDENTITY"
FAIL_OPENSSL: Final = "FAIL_Q5A_KEYGEN_OPENSSL"
FAIL_ROOT_PATH: Final = "FAIL_Q5A_KEYGEN_ROOT_PATH"
FAIL_ROOT_OVERLAP: Final = "FAIL_Q5A_KEYGEN_ROOT_OVERLAP"
FAIL_ROOT_NOT_FRESH: Final = "FAIL_Q5A_KEYGEN_ROOT_NOT_FRESH"
FAIL_ROOT_CREATE: Final = "FAIL_Q5A_KEYGEN_ROOT_CREATE"
FAIL_OPENSSL_GENPKEY: Final = "FAIL_Q5A_KEYGEN_OPENSSL_GENPKEY"
FAIL_OPENSSL_PUBOUT: Final = "FAIL_Q5A_KEYGEN_OPENSSL_PUBOUT"
FAIL_PRIVATE_KEY: Final = "FAIL_Q5A_KEYGEN_PRIVATE_KEY"
FAIL_PUBLIC_DER: Final = "FAIL_Q5A_KEYGEN_PUBLIC_DER"
FAIL_PUBLICATION: Final = "FAIL_Q5A_KEYGEN_PUBLICATION"
FAIL_UNEXPECTED: Final = "FAIL_Q5A_KEYGEN_UNEXPECTED"


class WorkerFailure(RuntimeError):
    """A stable worker failure that is safe to expose to the controller."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _StrictArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:  # pragma: no cover - argparse text varies
        del message
        raise WorkerFailure(FAIL_ARGUMENT_SCHEMA)


@dataclass(frozen=True, slots=True)
class WorkerArgumentsV1:
    operation: str
    signer_ordinal: int
    signer_role: str
    expected_uid: int
    expected_gid: int
    state_root: Path | None
    receipt_root: Path | None
    output_root: Path | None


def _strict_uint(text: str) -> int:
    if re.fullmatch(r"0|[1-9][0-9]*", text) is None:
        raise argparse.ArgumentTypeError("not a canonical uint")
    value = int(text)
    if value > (1 << 32) - 1:
        raise argparse.ArgumentTypeError("uint exceeds uint32")
    return value


def _canonical_absolute_path(raw: str) -> Path:
    if (
        type(raw) is not str
        or not raw
        or not raw.startswith("/")
        or raw != os.path.normpath(raw)
        or raw == "/"
    ):
        raise WorkerFailure(FAIL_ROOT_PATH)
    path = Path(raw)
    try:
        if path.resolve(strict=False) != path:
            raise WorkerFailure(FAIL_ROOT_PATH)
    except OSError as error:
        raise WorkerFailure(FAIL_ROOT_PATH) from error
    return path


def parse_arguments_v1(argv: Sequence[str]) -> WorkerArgumentsV1:
    parser = _StrictArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("operation", nargs="?")
    parser.add_argument("--startup-smoke", action="store_true")
    parser.add_argument("--signer-ordinal", type=_strict_uint, required=True)
    parser.add_argument("--signer-role", required=True)
    parser.add_argument("--expected-uid", type=_strict_uint, required=True)
    parser.add_argument("--expected-gid", type=_strict_uint, required=True)
    parser.add_argument("--state-root")
    parser.add_argument("--receipt-root")
    parser.add_argument("--output-root")
    namespace = parser.parse_args(list(argv))

    if namespace.signer_ordinal not in SIGNER_ROLE_BY_ORDINAL:
        raise WorkerFailure(FAIL_ARGUMENT_SCHEMA)
    if SIGNER_ROLE_BY_ORDINAL[namespace.signer_ordinal] != namespace.signer_role:
        raise WorkerFailure(FAIL_ARGUMENT_SCHEMA)

    roots = (namespace.state_root, namespace.receipt_root, namespace.output_root)
    if namespace.startup_smoke:
        if namespace.operation is not None or any(value is not None for value in roots):
            raise WorkerFailure(FAIL_ARGUMENT_SCHEMA)
        operation = "startup-smoke"
        parsed_roots: tuple[Path | None, Path | None, Path | None] = (
            None,
            None,
            None,
        )
    else:
        if namespace.operation != "generate" or any(value is None for value in roots):
            raise WorkerFailure(FAIL_ARGUMENT_SCHEMA)
        operation = "generate"
        parsed_roots = tuple(  # type: ignore[assignment]
            _canonical_absolute_path(value) for value in roots  # type: ignore[arg-type]
        )

    return WorkerArgumentsV1(
        operation=operation,
        signer_ordinal=namespace.signer_ordinal,
        signer_role=namespace.signer_role,
        expected_uid=namespace.expected_uid,
        expected_gid=namespace.expected_gid,
        state_root=parsed_roots[0],
        receipt_root=parsed_roots[1],
        output_root=parsed_roots[2],
    )


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


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


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def validate_generate_roots_v1(
    state_root: Path,
    receipt_root: Path,
    output_root: Path,
) -> tuple[Path, Path, Path]:
    roots = (state_root, receipt_root, output_root)
    if any(not isinstance(path, Path) or not path.is_absolute() for path in roots):
        raise WorkerFailure(FAIL_ROOT_PATH)
    if any(
        _paths_overlap(left, right)
        for index, left in enumerate(roots)
        for right in roots[index + 1 :]
    ):
        raise WorkerFailure(FAIL_ROOT_OVERLAP)
    for path in roots:
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        except OSError as error:
            raise WorkerFailure(FAIL_ROOT_NOT_FRESH) from error
        else:
            raise WorkerFailure(FAIL_ROOT_NOT_FRESH)
        try:
            parent = path.parent
            metadata = parent.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISDIR(metadata.st_mode)
                or parent.resolve(strict=True) != parent
                or not os.access(parent, os.W_OK | os.X_OK)
            ):
                raise WorkerFailure(FAIL_ROOT_PATH)
        except WorkerFailure:
            raise
        except OSError as error:
            raise WorkerFailure(FAIL_ROOT_PATH) from error
    return roots


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_fresh_roots_v1(roots: tuple[Path, Path, Path]) -> None:
    created: list[Path] = []
    try:
        for path in roots:
            os.mkdir(path, 0o700)
            created.append(path)
            os.chmod(path, 0o700, follow_symlinks=False)
            metadata = path.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISDIR(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or metadata.st_uid != os.getuid()
                or metadata.st_gid != os.getgid()
            ):
                raise OSError("created root differs")
            _fsync_directory(path.parent)
    except OSError as error:
        for path in reversed(created):
            try:
                os.rmdir(path)
                _fsync_directory(path.parent)
            except OSError:
                pass
        raise WorkerFailure(FAIL_ROOT_CREATE) from error


def _exclusive_bytes(path: Path, payload: bytes, mode: int) -> None:
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
        raise WorkerFailure(FAIL_PUBLICATION) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def parse_ed25519_spki_der_v1(payload: bytes) -> tuple[bytes, bytes]:
    if (
        type(payload) is not bytes
        or len(payload) != len(ED25519_SPKI_PREFIX) + 32
        or not payload.startswith(ED25519_SPKI_PREFIX)
    ):
        raise WorkerFailure(FAIL_PUBLIC_DER)
    public_key = payload[len(ED25519_SPKI_PREFIX) :]
    return public_key, hashlib.sha256(public_key).digest()[:16]


def build_public_receipt_v1(
    *, signer_ordinal: int, signer_role: str, public_key: bytes
) -> dict[str, object]:
    if (
        signer_ordinal not in SIGNER_ROLE_BY_ORDINAL
        or SIGNER_ROLE_BY_ORDINAL[signer_ordinal] != signer_role
        or type(public_key) is not bytes
        or len(public_key) != 32
    ):
        raise WorkerFailure(FAIL_PUBLIC_DER)
    key_id = hashlib.sha256(public_key).digest()[:16]
    return {
        "key_epoch": KEY_EPOCH,
        "key_generation_count": 1,
        "key_id_hex": key_id.hex(),
        "openssl_popen_count": 2,
        "private_key_exported": False,
        "public_key_derivation_count": 1,
        "public_key_hex": public_key.hex(),
        "purpose_id": PURPOSE_ID,
        "schema": PUBLIC_RECEIPT_SCHEMA,
        "signature_count": 0,
        "signer_ordinal": signer_ordinal,
        "signer_role": signer_role,
        "status": "PURPOSE5_KEYGEN_COMPLETE_PRIVATE_KEY_RETAINED",
    }


def _require_private_key_v1(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise WorkerFailure(FAIL_PRIVATE_KEY) from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.getuid()
        or metadata.st_gid != os.getgid()
        or metadata.st_nlink != 1
        or metadata.st_size <= 0
        or metadata.st_size > MAX_PRIVATE_KEY_BYTES
    ):
        raise WorkerFailure(FAIL_PRIVATE_KEY)
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        os.fsync(descriptor)
    except OSError as error:
        raise WorkerFailure(FAIL_PRIVATE_KEY) from error
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


def _startup_smoke_receipt(arguments: WorkerArgumentsV1) -> dict[str, object]:
    return {
        "expected_gid": arguments.expected_gid,
        "expected_uid": arguments.expected_uid,
        "key_generation_count": 0,
        "openssl_path": str(OPENSSL),
        "openssl_popen_count": 0,
        "public_key_derivation_count": 0,
        "purpose_id": PURPOSE_ID,
        "schema": STARTUP_SMOKE_SCHEMA,
        "signature_count": 0,
        "signer_ordinal": arguments.signer_ordinal,
        "signer_role": arguments.signer_role,
        "status": "Q5A_PURPOSE5_KEYGEN_STARTUP_SMOKE_PASS",
    }


def generate_v1(arguments: WorkerArgumentsV1) -> dict[str, object]:
    if (
        arguments.operation != "generate"
        or arguments.signer_ordinal not in SIGNER_ROLE_BY_ORDINAL
        or SIGNER_ROLE_BY_ORDINAL[arguments.signer_ordinal]
        != arguments.signer_role
        or arguments.state_root is None
        or arguments.receipt_root is None
        or arguments.output_root is None
    ):
        raise WorkerFailure(FAIL_ARGUMENT_SCHEMA)
    roots = validate_generate_roots_v1(
        arguments.state_root, arguments.receipt_root, arguments.output_root
    )
    _create_fresh_roots_v1(roots)
    state_root, receipt_root, output_root = roots
    private_key = state_root / PRIVATE_KEY_NAME
    public_der_path = output_root / PUBLIC_DER_NAME
    receipt_path = receipt_root / PUBLIC_RECEIPT_NAME

    if private_key.exists() or public_der_path.exists() or receipt_path.exists():
        raise WorkerFailure(FAIL_ROOT_NOT_FRESH)
    _run_openssl_v1(
        ["genpkey", "-algorithm", "ED25519", "-out", str(private_key)],
        working_directory=state_root,
        failure_code=FAIL_OPENSSL_GENPKEY,
    )
    _require_private_key_v1(private_key)
    _fsync_directory(state_root)
    public_der = _run_openssl_v1(
        ["pkey", "-in", str(private_key), "-pubout", "-outform", "DER"],
        working_directory=state_root,
        failure_code=FAIL_OPENSSL_PUBOUT,
    )
    public_key, _key_id = parse_ed25519_spki_der_v1(public_der)
    receipt = build_public_receipt_v1(
        signer_ordinal=arguments.signer_ordinal,
        signer_role=arguments.signer_role,
        public_key=public_key,
    )
    _exclusive_bytes(public_der_path, public_der, 0o644)
    _fsync_directory(output_root)
    _exclusive_bytes(receipt_path, _canonical_json(receipt), 0o644)
    _fsync_directory(receipt_root)
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
            receipt = generate_v1(arguments)
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
