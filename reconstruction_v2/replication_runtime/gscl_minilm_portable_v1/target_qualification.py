"""One-shot, source-free qualification of the target-local MiniLM manifest.

This command is infrastructure qualification only.  It accepts no benchmark
source, labels, predictions, or scorer.  It creates the target manifest with
the frozen MiniLM binding and then publishes a commitment-only safe receipt.
Both outputs are create-once mode-0600 files below one mode-0700 directory.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Any, Mapping, Sequence

from .binding import (
    GSCL_MINILM_TARGET_SCHEMA,
    GSCLMiniLMPortableError,
    MAXIMUM_TARGET_MANIFEST_BYTES,
    write_target_manifest_qualification_only,
)


VERSION = "gscl_minilm_target_qualification_v1"
SAFE_RECEIPT_SCHEMA = f"{VERSION}.safe_receipt.v1"
_RECONSTRUCTION_ROOT = Path(__file__).resolve().parents[2]
_CLOSURE_PATHS = {
    "replication_runtime_init": (
        _RECONSTRUCTION_ROOT / "replication_runtime/__init__.py"
    ).resolve(),
    "qualification_cli": Path(__file__).resolve(),
    "gscl_minilm_init": Path(__file__).with_name("__init__.py").resolve(),
    "gscl_minilm_binding": Path(__file__).with_name("binding.py").resolve(),
    "portable_minilm_binding": (
        _RECONSTRUCTION_ROOT
        / "replication_runtime/qasper_minilm_portable_v2/binding.py"
    ).resolve(),
    "portable_minilm_init": (
        _RECONSTRUCTION_ROOT
        / "replication_runtime/qasper_minilm_portable_v2/__init__.py"
    ).resolve(),
    "base_minilm_binding": (
        _RECONSTRUCTION_ROOT
        / "replication_runtime/qasper_minilm_v1/binding.py"
    ).resolve(),
    "base_minilm_init": (
        _RECONSTRUCTION_ROOT
        / "replication_runtime/qasper_minilm_v1/__init__.py"
    ).resolve(),
}
_BUNDLE_TARGET_NAME = "target_manifest.json"
_BUNDLE_RECEIPT_NAME = "qualification.safe.json"
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class TargetQualificationError(RuntimeError):
    """The source-free target qualification failed closed."""

    def __init__(self, issue_id: str) -> None:
        super().__init__(issue_id)
        self.issue_id = issue_id


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise TargetQualificationError(
            "qualification_receipt_not_canonical"
        ) from exc


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_regular(path: Path, *, maximum: int) -> bytes:
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TargetQualificationError(
            "qualification_input_unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 1
            or before.st_size > maximum
        ):
            raise TargetQualificationError(
                "qualification_input_invalid"
            )
        raw = b""
        while len(raw) <= maximum:
            block = os.read(
                descriptor,
                min(1024 * 1024, maximum + 1 - len(raw)),
            )
            if not block:
                break
            raw += block
        after = os.fstat(descriptor)
        if (
            len(raw) > maximum
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
        ):
            raise TargetQualificationError(
                "qualification_input_changed"
            )
        return raw
    finally:
        os.close(descriptor)


def _safe_absolute(path: Path, *, field: str) -> Path:
    if not path.is_absolute():
        raise TargetQualificationError(f"{field}_not_absolute")
    absolute = Path(os.path.abspath(os.fspath(path)))
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor = cursor / component
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise TargetQualificationError(
                f"{field}_symlink_component"
            )
    return absolute


def _require_private_parent(path: Path) -> None:
    try:
        metadata = path.parent.stat(follow_symlinks=False)
    except OSError as exc:
        raise TargetQualificationError(
            "qualification_output_parent_invalid"
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
    ):
        raise TargetQualificationError(
            "qualification_output_parent_invalid"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_safe_once(path: Path, raw: bytes) -> str:
    """Validate a pending inode, then hardlink-publish without replacement."""

    if path.exists() or path.is_symlink():
        raise TargetQualificationError(
            "qualification_receipt_already_exists"
        )
    pending: Path | None = None
    descriptor: int | None = None
    published_identity: tuple[int, int] | None = None
    succeeded = False
    try:
        for _ in range(32):
            candidate = path.with_name(
                f".{path.name}.pending-{secrets.token_hex(16)}"
            )
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                )
            except FileExistsError:
                continue
            pending = candidate
            break
        if pending is None or descriptor is None:
            raise TargetQualificationError(
                "qualification_receipt_pending_unavailable"
            )
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            count = os.write(descriptor, view[offset:])
            if count <= 0:
                raise TargetQualificationError(
                    "qualification_receipt_write_failed"
                )
            offset += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        observed = _read_regular(pending, maximum=1024 * 1024)
        if observed != raw:
            raise TargetQualificationError(
                "qualification_receipt_write_mismatch"
            )
        pending_metadata = pending.stat(follow_symlinks=False)
        try:
            os.link(pending, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise TargetQualificationError(
                "qualification_receipt_already_exists"
            ) from exc
        published_identity = (
            pending_metadata.st_dev,
            pending_metadata.st_ino,
        )
        final_metadata = path.stat(follow_symlinks=False)
        if (
            final_metadata.st_dev,
            final_metadata.st_ino,
        ) != published_identity:
            raise TargetQualificationError(
                "qualification_receipt_publish_mismatch"
            )
        pending.unlink()
        pending = None
        _fsync_directory(path.parent)
        if _read_regular(path, maximum=1024 * 1024) != raw:
            raise TargetQualificationError(
                "qualification_receipt_write_mismatch"
            )
        succeeded = True
        return _sha256_bytes(raw)
    except TargetQualificationError:
        raise
    except OSError as exc:
        raise TargetQualificationError(
            "qualification_receipt_publish_failed"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if not succeeded and published_identity is not None:
            try:
                metadata = path.stat(follow_symlinks=False)
                if (
                    metadata.st_dev,
                    metadata.st_ino,
                ) == published_identity:
                    path.unlink()
            except FileNotFoundError:
                pass
        if pending is not None:
            try:
                pending.unlink()
            except FileNotFoundError:
                pass
        if not succeeded:
            try:
                _fsync_directory(path.parent)
            except OSError:
                pass


def _code_closure() -> dict[str, str]:
    return {
        name: _sha256_bytes(
            _read_regular(path, maximum=8 * 1024 * 1024)
        )
        for name, path in sorted(_CLOSURE_PATHS.items())
    }


def _private_regular_bytes(path: Path, *, maximum: int) -> bytes:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise TargetQualificationError(
            "qualification_bundle_file_invalid"
        ) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
    ):
        raise TargetQualificationError(
            "qualification_bundle_file_invalid"
        )
    return _read_regular(path, maximum=maximum)


def _decode_canonical(raw: bytes, *, issue: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TargetQualificationError(issue) from exc
    if (
        not isinstance(value, dict)
        or raw != _canonical_bytes(value) + b"\n"
    ):
        raise TargetQualificationError(issue)
    return value


def _load_complete_bundle(
    bundle: Path,
    *,
    asset_manifest: Path,
    expected_code_closure: Mapping[str, str],
) -> dict[str, Any]:
    """Validate one atomically visible completed qualification bundle."""

    try:
        metadata = bundle.stat(follow_symlinks=False)
        names = set(os.listdir(bundle))
    except OSError as exc:
        raise TargetQualificationError(
            "qualification_bundle_invalid"
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
        or names != {_BUNDLE_TARGET_NAME, _BUNDLE_RECEIPT_NAME}
    ):
        raise TargetQualificationError(
            "qualification_bundle_invalid"
        )
    target_raw = _private_regular_bytes(
        bundle / _BUNDLE_TARGET_NAME,
        maximum=MAXIMUM_TARGET_MANIFEST_BYTES,
    )
    receipt_raw = _private_regular_bytes(
        bundle / _BUNDLE_RECEIPT_NAME,
        maximum=1024 * 1024,
    )
    target = _decode_canonical(
        target_raw, issue="qualification_target_invalid"
    )
    receipt = _decode_canonical(
        receipt_raw, issue="qualification_receipt_invalid"
    )
    receipt_body = dict(receipt)
    claimed = receipt_body.pop("self_sha256", None)
    if (
        target.get("schema") != GSCL_MINILM_TARGET_SCHEMA
        or receipt.get("schema") != SAFE_RECEIPT_SCHEMA
        or receipt.get("status")
        != "PASS_MINILM_TARGET_SOURCE_FREE_QUALIFICATION"
        or not isinstance(claimed, str)
        or _sha256_bytes(_canonical_bytes(receipt_body)) != claimed
        or receipt.get("target_manifest_file_sha256")
        != _sha256_bytes(target_raw)
        or receipt.get("target_manifest_self_sha256")
        != target.get("self_sha256")
        or receipt.get("runtime_code_closure_sha256s")
        != dict(expected_code_closure)
        or receipt.get("asset_manifest_file_sha256")
        != _sha256_bytes(
            _read_regular(
                asset_manifest, maximum=MAXIMUM_TARGET_MANIFEST_BYTES
            )
        )
        or receipt.get("official_source_open_count") != 0
        or receipt.get("label_open_count") != 0
        or receipt.get("network_call_count") != 0
        or receipt.get("formal_measurement") is not False
        or receipt.get("effect_gate_added") is not False
    ):
        raise TargetQualificationError(
            "qualification_bundle_commitment_invalid"
        )
    return receipt


def _rename_bundle_no_replace(source: Path, destination: Path) -> None:
    """Atomically expose the complete bundle and never replace a prior one."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise TargetQualificationError(
            "qualification_renameat2_unavailable"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        observed_errno = ctypes.get_errno()
        if observed_errno == errno.EEXIST:
            raise TargetQualificationError(
                "qualification_bundle_already_exists"
            )
        raise TargetQualificationError(
            "qualification_bundle_publish_failed"
        )


def _cleanup_pending_bundle(path: Path) -> None:
    """Best-effort cleanup of only this invocation's hidden bundle."""

    try:
        names = set(os.listdir(path))
    except FileNotFoundError:
        return
    if names - {_BUNDLE_TARGET_NAME, _BUNDLE_RECEIPT_NAME}:
        return
    for name in (_BUNDLE_TARGET_NAME, _BUNDLE_RECEIPT_NAME):
        candidate = path / name
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass
    try:
        path.rmdir()
    except FileNotFoundError:
        pass


def run_source_free_target_qualification(
    *,
    asset_manifest_path: Path,
    model_root: Path,
    output_bundle_path: Path,
) -> Mapping[str, object]:
    """Atomically create or recover the complete non-scoring bundle."""

    asset_manifest = _safe_absolute(
        asset_manifest_path, field="asset_manifest"
    )
    model = _safe_absolute(model_root, field="model_root")
    bundle = _safe_absolute(
        output_bundle_path, field="output_bundle"
    )
    _require_private_parent(bundle)
    initial_code_closure = _code_closure()
    if bundle.exists() and not bundle.is_symlink():
        return _load_complete_bundle(
            bundle,
            asset_manifest=asset_manifest,
            expected_code_closure=initial_code_closure,
        )
    if bundle.is_symlink():
        raise TargetQualificationError(
            "qualification_bundle_invalid"
        )

    pending: Path | None = None
    for _ in range(32):
        candidate = bundle.with_name(
            f".{bundle.name}.pending-{secrets.token_hex(16)}"
        )
        try:
            os.mkdir(candidate, mode=0o700)
        except FileExistsError:
            continue
        candidate.chmod(0o700)
        pending = candidate
        break
    if pending is None:
        raise TargetQualificationError(
            "qualification_bundle_pending_unavailable"
        )
    target = pending / _BUNDLE_TARGET_NAME
    receipt_path = pending / _BUNDLE_RECEIPT_NAME
    try:
        target_receipt = write_target_manifest_qualification_only(
            target_manifest_path=target,
            asset_manifest_path=asset_manifest,
            model_root=model,
        )
        target_metadata = target.stat(follow_symlinks=False)
        target_file_sha256 = str(
            target_receipt["target_manifest_file_sha256"]
        )
        target_raw = _read_regular(
            target, maximum=MAXIMUM_TARGET_MANIFEST_BYTES
        )
        if (
            not stat.S_ISREG(target_metadata.st_mode)
            or stat.S_IMODE(target_metadata.st_mode) != 0o600
            or target_metadata.st_uid != os.geteuid()
            or target_metadata.st_nlink != 1
            or _sha256_bytes(target_raw) != target_file_sha256
        ):
            raise TargetQualificationError(
                "qualification_target_custody_invalid"
            )
        target_value = _decode_canonical(
            target_raw, issue="qualification_target_invalid"
        )
        if (
            target_value.get("schema")
            != GSCL_MINILM_TARGET_SCHEMA
            or target_value.get("self_sha256")
            != target_receipt["target_manifest_self_sha256"]
        ):
            raise TargetQualificationError(
                "qualification_target_invalid"
            )
        observed_code_closure = _code_closure()
        if observed_code_closure != initial_code_closure:
            raise TargetQualificationError(
                "qualification_code_closure_changed"
            )
        asset_file_sha256 = _sha256_bytes(
            _read_regular(
                asset_manifest, maximum=MAXIMUM_TARGET_MANIFEST_BYTES
            )
        )
        body: dict[str, Any] = {
            "api_evaluation_count": 0,
            "claim_scope": (
                "source_free_target_runtime_qualification_only"
            ),
            "effect_gate_added": False,
            "formal_measurement": False,
            "label_open_count": 0,
            "minilm_model_construction_count": 1,
            "network_call_count": 0,
            "official_source_open_count": 0,
            "qualification_only": True,
            "runtime_code_closure_sha256s": initial_code_closure,
            "schema": SAFE_RECEIPT_SCHEMA,
            "source_content_emitted": False,
            "status": "PASS_MINILM_TARGET_SOURCE_FREE_QUALIFICATION",
            "target_manifest_file_sha256": target_file_sha256,
            "target_manifest_self_sha256": target_value[
                "self_sha256"
            ],
            "target_observed_float32_sha256": target_value[
                "public_synthetic_canary"
            ]["target_observed_float32_sha256"],
            "target_observed_quantized_sha256": target_value[
                "public_synthetic_canary"
            ]["target_observed_quantized_sha256"],
            "target_repeat_count": 2,
            "target_repeat_exact": True,
            "asset_manifest_file_sha256": asset_file_sha256,
            "model_tree_sha256": target_value["base_asset"][
                "model_tree_sha256"
            ],
            "version": VERSION,
        }
        receipt = {
            **body,
            "self_sha256": _sha256_bytes(_canonical_bytes(body)),
        }
        _publish_safe_once(
            receipt_path, _canonical_bytes(receipt) + b"\n"
        )
        observed = _load_complete_bundle(
            pending,
            asset_manifest=asset_manifest,
            expected_code_closure=initial_code_closure,
        )
        if observed != receipt:
            raise TargetQualificationError(
                "qualification_bundle_commitment_invalid"
            )
        _fsync_directory(pending)
        _rename_bundle_no_replace(pending, bundle)
        pending = None
        _fsync_directory(bundle.parent)
        return receipt
    except BaseException:
        if pending is not None:
            _cleanup_pending_bundle(pending)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-manifest", required=True, type=Path
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--output-bundle", required=True, type=Path
    )
    arguments = parser.parse_args(argv)
    receipt = run_source_free_target_qualification(
        asset_manifest_path=arguments.asset_manifest,
        model_root=arguments.model,
        output_bundle_path=arguments.output_bundle,
    )
    print(
        json.dumps(
            {
                "self_sha256": receipt["self_sha256"],
                "status": receipt["status"],
                "target_manifest_file_sha256": receipt[
                    "target_manifest_file_sha256"
                ],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        GSCLMiniLMPortableError,
        TargetQualificationError,
    ) as exc:
        issue_id = getattr(exc, "issue_id", str(exc))
        print(
            f"{VERSION} failed closed: {issue_id}",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


__all__ = [
    "SAFE_RECEIPT_SCHEMA",
    "TargetQualificationError",
    "VERSION",
    "run_source_free_target_qualification",
]
