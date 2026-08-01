"""Target-local GSCL binding for the exact public MiniLM asset.

The immutable model and package-version contract remains owned by
``qasper_minilm_v1``.  Hardware-portable execution remains owned by
``qasper_minilm_portable_v2``.  This module adds only one missing custody
layer: a canonical, mode-0600 manifest generated during source-free runtime
qualification on the eventual target machine.

The target manifest binds the complete installed-file closure of every
critical MiniLM distribution, the interpreter/platform/CPU/torch build, the
small deterministic environment allow-list, and the full portable-v2 runtime
and public-synthetic canary receipt preimages.  The machine-local observed
float and quantized hashes are acceptance evidence only for that exact target
closure.  The older qasper-v1 output hashes are retained as historical
references and are explicitly *not* cross-hardware byte-identity claims.

Formal construction accepts no injected encoder and no caller receipt.  It
requires a securely owned canonical target manifest, reconstructs the exact
portable-v2 encoder (whose canary cannot be skipped), recomputes the complete
manifest, and requires byte equality before exposing the familiar
``encode``, ``runtime_receipt``, and ``canary_receipt`` interface.
"""

from __future__ import annotations

import hashlib
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import platform
import secrets
import stat
import sys
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from replication_runtime.qasper_minilm_portable_v2.binding import (
    PortableMiniLMError,
    PortableOfflineMiniLMEncoder,
)
from replication_runtime.qasper_minilm_v1 import binding as frozen_v1


GSCL_MINILM_TARGET_SCHEMA = "gscl_minilm_portable_target_manifest_v1"
GSCL_MINILM_RUNTIME_SCHEMA = "gscl_minilm_portable_runtime_receipt_v1"
GSCL_MINILM_CANARY_SCHEMA = "gscl_minilm_portable_canary_receipt_v1"
MAXIMUM_TARGET_MANIFEST_BYTES = 2 * 1024 * 1024
MAXIMUM_DISTRIBUTION_FILE_BYTES = 8 * 1024 * 1024 * 1024
MAXIMUM_DISTRIBUTION_TOTAL_BYTES = 32 * 1024 * 1024 * 1024
MAXIMUM_DISTRIBUTION_FILE_COUNT = 200_000

_CRITICAL_DISTRIBUTIONS = (
    ("huggingface-hub", "huggingface_hub"),
    ("numpy", "numpy"),
    ("safetensors", "safetensors"),
    ("sentence-transformers", "sentence_transformers"),
    ("tokenizers", "tokenizers"),
    ("torch", "torch"),
    ("transformers", "transformers"),
)
_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HUB_OFFLINE",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
    "TOKENIZERS_PARALLELISM",
    "TRANSFORMERS_OFFLINE",
)
_GSCL_ENCODER_AUTHORITY_MARKER = object()


class GSCLMiniLMPortableError(RuntimeError):
    """The GSCL target-local MiniLM contract failed closed."""


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
        raise GSCLMiniLMPortableError("target_manifest_not_canonical") from exc


def _canonical_json_text(value: object) -> str:
    return _canonical_bytes(value).decode("ascii")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path, *, maximum: int) -> str:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            total += len(block)
            if total > maximum:
                raise GSCLMiniLMPortableError("runtime_file_oversized")
            digest.update(block)
    return digest.hexdigest()


def _safe_absolute(path: str | Path, *, field: str) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise GSCLMiniLMPortableError(f"{field}_not_absolute")
    absolute = Path(os.path.abspath(os.fspath(raw)))
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor = cursor / component
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise GSCLMiniLMPortableError(f"{field}_symlink_component")
    return absolute


def _secure_manifest_bytes(path: str | Path) -> tuple[Path, bytes]:
    absolute = _safe_absolute(path, field="target_manifest")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise GSCLMiniLMPortableError("target_manifest_unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or not 1 <= before.st_size <= MAXIMUM_TARGET_MANIFEST_BYTES
        ):
            raise GSCLMiniLMPortableError("target_manifest_custody_invalid")
        raw = b""
        while len(raw) <= MAXIMUM_TARGET_MANIFEST_BYTES:
            block = os.read(descriptor, min(1024 * 1024, MAXIMUM_TARGET_MANIFEST_BYTES + 1 - len(raw)))
            if not block:
                break
            raw += block
        after = os.fstat(descriptor)
        if (
            len(raw) > MAXIMUM_TARGET_MANIFEST_BYTES
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        ):
            raise GSCLMiniLMPortableError("target_manifest_changed_during_read")
        return absolute, raw
    finally:
        os.close(descriptor)


def _decode_target_manifest(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GSCLMiniLMPortableError("target_manifest_invalid") from exc
    if not isinstance(value, dict):
        raise GSCLMiniLMPortableError("target_manifest_invalid")
    if raw != _canonical_bytes(value) + b"\n":
        raise GSCLMiniLMPortableError("target_manifest_not_canonical")
    body = dict(value)
    claimed = body.pop("self_sha256", None)
    if (
        value.get("schema") != GSCL_MINILM_TARGET_SCHEMA
        or not isinstance(claimed, str)
        or len(claimed) != 64
        or _sha256_bytes(_canonical_bytes(body)) != claimed
    ):
        raise GSCLMiniLMPortableError("target_manifest_self_hash_invalid")
    return value


def _interpreter_receipt() -> dict[str, object]:
    executable = Path(sys.executable).resolve(strict=True)
    return {
        "cache_tag": sys.implementation.cache_tag,
        "implementation": platform.python_implementation(),
        "invoked_path": sys.executable,
        "resolved_path": str(executable),
        "sha256": _sha256_file(executable, maximum=512 * 1024 * 1024),
        "version": platform.python_version(),
    }


def _cpu_receipt() -> dict[str, object]:
    identity: dict[str, str] = {}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file() and not cpuinfo.is_symlink():
        try:
            text = cpuinfo.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise GSCLMiniLMPortableError("cpu_identity_unavailable") from exc
        first = text.split("\n\n", 1)[0]
        allowed = {
            "cpu family",
            "flags",
            "microcode",
            "model",
            "model name",
            "stepping",
            "vendor_id",
        }
        for line in first.splitlines():
            key, separator, value = line.partition(":")
            normalized = key.strip()
            if separator and normalized in allowed:
                identity[normalized] = " ".join(value.split())
    return {
        "architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "processor": platform.processor(),
        "stable_identity": dict(sorted(identity.items())),
    }


def _torch_build_receipt() -> dict[str, object]:
    try:
        import torch
    except ImportError as exc:
        raise GSCLMiniLMPortableError("torch_runtime_unavailable") from exc
    config_text = str(torch.__config__.show())
    return {
        "config_sha256": _sha256_bytes(config_text.encode("utf-8")),
        "config_text": config_text,
        "cuda_build": getattr(torch.version, "cuda", None),
        "cudnn_version": (
            torch.backends.cudnn.version()
            if hasattr(torch.backends, "cudnn")
            else None
        ),
        "debug_build": bool(getattr(torch.version, "debug", False)),
        "git_version": getattr(torch.version, "git_version", None),
        "num_interop_threads": int(torch.get_num_interop_threads()),
        "num_threads": int(torch.get_num_threads()),
        "version": str(torch.__version__),
    }


def _distribution_content_closure_with_origin(
    distribution_name: str,
    *,
    required_module_origin: str | Path | None,
) -> tuple[str, str | None]:
    """Hash every declared file and bind the module actually imported."""

    try:
        distribution = metadata.distribution(distribution_name)
        declared_files = distribution.files
    except metadata.PackageNotFoundError as exc:
        raise GSCLMiniLMPortableError(
            "critical_distribution_unavailable"
        ) from exc
    if (
        not declared_files
        or len(declared_files) > MAXIMUM_DISTRIBUTION_FILE_COUNT
    ):
        raise GSCLMiniLMPortableError(
            "critical_distribution_file_set_invalid"
        )
    rows: list[dict[str, object]] = []
    declared_absolute_paths: dict[str, str] = {}
    declared_observations: dict[
        str,
        tuple[
            Path,
            tuple[int, int, int, int, int],
            dict[str, object],
        ],
    ] = {}
    physical_path_owners: dict[tuple[int, int], str] = {}
    total_size = 0
    for declared in sorted(declared_files, key=str):
        declared_text = str(declared)
        if (
            not declared_text
            or "\x00" in declared_text
            or Path(declared_text).is_absolute()
        ):
            raise GSCLMiniLMPortableError(
                "critical_distribution_path_invalid"
            )
        path = Path(
            os.path.abspath(
                os.fspath(distribution.locate_file(declared))
            )
        )
        cached = declared_observations.get(declared_text)
        if cached is not None:
            cached_path, _, cached_row = cached
            if path != cached_path:
                raise GSCLMiniLMPortableError(
                    "critical_distribution_duplicate_path_drifted"
                )
            rows.append(dict(cached_row))
            total_size += int(cached_row["size"])
            if total_size > MAXIMUM_DISTRIBUTION_TOTAL_BYTES:
                raise GSCLMiniLMPortableError(
                    "critical_distribution_too_large"
                )
            continue
        signature, row = _stable_distribution_file_observation(
            path, declared_text=declared_text
        )
        owner = declared_absolute_paths.get(str(path))
        if owner is not None and owner != declared_text:
            raise GSCLMiniLMPortableError(
                "critical_distribution_path_alias"
            )
        physical_identity = (signature[0], signature[1])
        physical_owner = physical_path_owners.get(physical_identity)
        if (
            physical_owner is not None
            and physical_owner != declared_text
        ):
            raise GSCLMiniLMPortableError(
                "critical_distribution_path_alias"
            )
        declared_absolute_paths[str(path)] = declared_text
        physical_path_owners[physical_identity] = declared_text
        declared_observations[declared_text] = (
            path,
            signature,
            row,
        )
        rows.append(dict(row))
        total_size += int(row["size"])
        if total_size > MAXIMUM_DISTRIBUTION_TOTAL_BYTES:
            raise GSCLMiniLMPortableError(
                "critical_distribution_too_large"
            )
    for declared_text in sorted(declared_observations):
        path, expected_signature, expected_row = (
            declared_observations[declared_text]
        )
        observed_signature, observed_row = (
            _stable_distribution_file_observation(
                path, declared_text=declared_text
            )
        )
        if (
            observed_signature != expected_signature
            or observed_row != expected_row
        ):
            raise GSCLMiniLMPortableError(
                "critical_distribution_file_changed_across_closure"
            )
    bound_origin: str | None = None
    if required_module_origin is not None:
        origin = Path(
            os.path.abspath(os.fspath(required_module_origin))
        )
        bound_origin = declared_absolute_paths.get(str(origin))
        if bound_origin is None:
            raise GSCLMiniLMPortableError(
                "critical_distribution_module_origin_unbound"
            )
    return _sha256_bytes(_canonical_bytes(rows)), bound_origin


def _stable_distribution_file_observation(
    path: Path,
    *,
    declared_text: str,
) -> tuple[
    tuple[int, int, int, int, int],
    dict[str, object],
]:
    """Read one declared regular file and prove that read was stable."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise GSCLMiniLMPortableError(
            "critical_distribution_file_unavailable"
        ) from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_size < 0
        or before.st_size > MAXIMUM_DISTRIBUTION_FILE_BYTES
    ):
        raise GSCLMiniLMPortableError(
            "critical_distribution_file_invalid"
        )
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise GSCLMiniLMPortableError(
            "critical_distribution_file_unavailable"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (before.st_dev, before.st_ino, before.st_size)
        ):
            raise GSCLMiniLMPortableError(
                "critical_distribution_file_changed"
            )
        digest = hashlib.sha256()
        observed_size = 0
        for block in iter(
            lambda: os.read(descriptor, 1024 * 1024), b""
        ):
            observed_size += len(block)
            if observed_size > MAXIMUM_DISTRIBUTION_FILE_BYTES:
                raise GSCLMiniLMPortableError(
                    "critical_distribution_file_oversized"
                )
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            observed_size != opened.st_size
            or (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
        ):
            raise GSCLMiniLMPortableError(
                "critical_distribution_file_changed"
            )
    finally:
        os.close(descriptor)
    signature = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    return (
        signature,
        {
            "declared_path": declared_text,
            "sha256": digest.hexdigest(),
            "size": observed_size,
        },
    )


def _distribution_content_closure(
    distribution_name: str,
) -> str:
    """Hash every file declared by one installed distribution."""

    closure, _ = _distribution_content_closure_with_origin(
        distribution_name, required_module_origin=None
    )
    return closure


def _distribution_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for distribution_name, runtime_key in _CRITICAL_DISTRIBUTIONS:
        try:
            distribution_version = metadata.version(distribution_name)
            runtime_module = import_module(runtime_key)
            module_version = str(
                getattr(runtime_module, "__version__")
            )
            module_origin = getattr(runtime_module, "__file__", None)
            if not isinstance(module_origin, str) or not module_origin:
                raise GSCLMiniLMPortableError(
                    "critical_distribution_module_origin_unavailable"
                )
            closure, origin_declared_path = (
                _distribution_content_closure_with_origin(
                    distribution_name,
                    required_module_origin=module_origin,
                )
            )
        except (
            AttributeError,
            ImportError,
            metadata.PackageNotFoundError,
            RuntimeError,
            OSError,
        ) as exc:
            raise GSCLMiniLMPortableError(
                "critical_distribution_closure_failed"
            ) from exc
        expected_version = frozen_v1.EXPECTED_RUNTIME_VERSIONS[
            runtime_key
        ]
        if (
            module_version != expected_version
            or (
                runtime_key != "torch"
                and distribution_version != module_version
            )
        ):
            raise GSCLMiniLMPortableError(
                "critical_distribution_version_drifted"
            )
        rows.append(
            {
                "closure_algorithm": (
                    "sha256(canonical_JSON([{path,sha256,size} for "
                    "every installed file declared by distribution.files]))"
                ),
                "content_closure_sha256": closure,
                "distribution": distribution_name,
                "distribution_version": distribution_version,
                "runtime_module_origin_declared_path": (
                    origin_declared_path
                ),
                "runtime_module_version": module_version,
            }
        )
    return rows


def _runtime_closure() -> dict[str, object]:
    distribution_rows = _distribution_rows()
    environment = {
        key: os.environ.get(key) for key in _ENVIRONMENT_KEYS
    }
    expected_environment = frozen_v1.EXPECTED_EXECUTION["environment"]
    if any(
        environment.get(str(key)) != str(value)
        for key, value in expected_environment.items()
    ):
        raise GSCLMiniLMPortableError("offline_environment_drifted")
    body: dict[str, object] = {
        "critical_distributions": distribution_rows,
        "critical_distribution_content_closure_sha256": _sha256_bytes(
            _canonical_bytes(distribution_rows)
        ),
        "environment_allowlist": environment,
        "interpreter": _interpreter_receipt(),
        "platform": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "system": platform.system(),
            "uname": list(platform.uname()),
        },
        "cpu": _cpu_receipt(),
        "torch_build": _torch_build_receipt(),
    }
    return body


def _require_mapping(value: object, *, issue: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GSCLMiniLMPortableError(issue)
    return value


def _manifest_body(
    encoder: PortableOfflineMiniLMEncoder,
) -> dict[str, object]:
    runtime = _require_mapping(
        getattr(encoder, "runtime_receipt", None),
        issue="portable_runtime_receipt_invalid",
    )
    canary = _require_mapping(
        getattr(encoder, "canary_receipt", None),
        issue="portable_canary_receipt_invalid",
    )
    hashes = _require_mapping(
        canary.get("observed_output_hashes"),
        issue="portable_canary_hashes_invalid",
    )
    observed_float = hashes.get(
        "float32_little_endian_c_order_sha256"
    )
    observed_quantized = hashes.get(
        "quantized_embedding_matrix_sha256"
    )
    if (
        canary.get("repeat_count") != 2
        or canary.get("repeat_byte_exact") is not True
        or canary.get("repeat_elementwise_exact") is not True
        or canary.get("public_text_vector_sha256")
        != frozen_v1.CANARY_TEXT_VECTOR_SHA256
        or hashes.get("normative_acceptance") is not False
        or hashes.get("compared_to_expected_or_allowlist") is not False
        or not isinstance(observed_float, str)
        or len(observed_float) != 64
        or not isinstance(observed_quantized, str)
        or len(observed_quantized) != 64
    ):
        raise GSCLMiniLMPortableError("portable_canary_receipt_invalid")
    runtime_json = _canonical_json_text(dict(runtime))
    canary_json = _canonical_json_text(dict(canary))
    return {
        "schema": GSCL_MINILM_TARGET_SCHEMA,
        "claim_scope": (
            "target_local_repeat_exact_public_synthetic_minilm_binding"
        ),
        "base_asset": {
            "asset_file_sha256": frozen_v1.ASSET_FILE_SHA256,
            "asset_self_sha256": frozen_v1.ASSET_SELF_SHA256,
            "model_revision": frozen_v1.MODEL_REVISION,
            "model_tree_sha256": frozen_v1.MODEL_TREE_SHA256,
            "weights_sha256": frozen_v1.WEIGHTS_SHA256,
        },
        "portable_runtime_receipt_json": runtime_json,
        "portable_runtime_receipt_sha256": _sha256_bytes(
            runtime_json.encode("ascii")
        ),
        "portable_canary_receipt_json": canary_json,
        "portable_canary_receipt_sha256": _sha256_bytes(
            canary_json.encode("ascii")
        ),
        "public_synthetic_canary": {
            "input_text_vector_sha256": (
                frozen_v1.CANARY_TEXT_VECTOR_SHA256
            ),
            "sentence_count": frozen_v1.CANARY_SENTENCE_COUNT,
            "target_observed_float32_sha256": observed_float,
            "target_observed_quantized_sha256": observed_quantized,
            "target_repeat_count": 2,
            "target_repeat_byte_exact": True,
            "target_repeat_elementwise_exact": True,
            "legacy_v1_float32_sha256_reference_only": (
                frozen_v1.CANARY_FLOAT32_BYTES_SHA256
            ),
            "legacy_v1_quantized_sha256_reference_only": (
                frozen_v1.CANARY_QUANTIZED_EMBEDDING_SHA256
            ),
            "cross_hardware_byte_identity_claimed": False,
            "legacy_hashes_are_acceptance_oracle": False,
        },
        "target_runtime_closure": _runtime_closure(),
        "formal_source_or_rows_accessed": False,
        "labels_accessed": False,
        "network_calls": 0,
        "qualification_only_builder": True,
    }


def _encode_manifest(body: Mapping[str, object]) -> bytes:
    mutable = dict(body)
    mutable["self_sha256"] = _sha256_bytes(_canonical_bytes(mutable))
    return _canonical_bytes(mutable) + b"\n"


def build_target_manifest_qualification_only(
    *,
    asset_manifest_path: str | Path,
    model_root: str | Path,
) -> bytes:
    """Build deterministic target-local bytes without any benchmark source."""

    try:
        encoder = PortableOfflineMiniLMEncoder(
            asset_manifest_path=asset_manifest_path,
            model_root=model_root,
            run_canary=True,
        )
    except PortableMiniLMError as exc:
        raise GSCLMiniLMPortableError(
            "portable_minilm_qualification_failed"
        ) from exc
    return _encode_manifest(_manifest_body(encoder))


def _fsync_directory(path: Path) -> None:
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_same_inode_if_present(
    path: Path,
    *,
    expected_device: int,
    expected_inode: int,
) -> None:
    try:
        observed = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if (
        observed.st_dev == expected_device
        and observed.st_ino == expected_inode
    ):
        path.unlink()


def _publish_target_manifest_once(
    absolute: Path,
    raw: bytes,
) -> tuple[Path, bytes, dict[str, Any]]:
    """Validate a pending inode, then hardlink-publish without replacement."""

    parent = absolute.parent
    if (
        not parent.is_dir()
        or parent.is_symlink()
        or absolute.exists()
        or absolute.is_symlink()
    ):
        raise GSCLMiniLMPortableError(
            "target_manifest_parent_invalid"
        )
    pending: Path | None = None
    descriptor: int | None = None
    published_identity: tuple[int, int] | None = None
    succeeded = False
    try:
        for _ in range(32):
            candidate = absolute.with_name(
                f".{absolute.name}.pending-{secrets.token_hex(16)}"
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
            raise GSCLMiniLMPortableError(
                "target_manifest_pending_unavailable"
            )
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise GSCLMiniLMPortableError(
                    "target_manifest_write_failed"
                )
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None

        pending_path, pending_raw = _secure_manifest_bytes(pending)
        if pending_path != pending or pending_raw != raw:
            raise GSCLMiniLMPortableError(
                "target_manifest_write_mismatch"
            )
        pending_value = _decode_target_manifest(pending_raw)
        pending_metadata = pending.stat(follow_symlinks=False)
        try:
            os.link(
                pending,
                absolute,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise GSCLMiniLMPortableError(
                "target_manifest_create_failed"
            ) from exc
        published_identity = (
            pending_metadata.st_dev,
            pending_metadata.st_ino,
        )
        final_metadata = absolute.stat(follow_symlinks=False)
        if (
            final_metadata.st_dev,
            final_metadata.st_ino,
        ) != published_identity:
            raise GSCLMiniLMPortableError(
                "target_manifest_publish_identity_mismatch"
            )
        pending.unlink()
        pending = None
        _fsync_directory(parent)

        read_path, observed = _secure_manifest_bytes(absolute)
        if observed != raw:
            raise GSCLMiniLMPortableError(
                "target_manifest_write_mismatch"
            )
        value = _decode_target_manifest(observed)
        if value != pending_value:
            raise GSCLMiniLMPortableError(
                "target_manifest_write_mismatch"
            )
        succeeded = True
        return read_path, observed, value
    except GSCLMiniLMPortableError:
        raise
    except OSError as exc:
        raise GSCLMiniLMPortableError(
            "target_manifest_create_failed"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if not succeeded and published_identity is not None:
            try:
                _unlink_same_inode_if_present(
                    absolute,
                    expected_device=published_identity[0],
                    expected_inode=published_identity[1],
                )
            except OSError:
                pass
        if pending is not None:
            try:
                pending.unlink()
            except FileNotFoundError:
                pass
        if not succeeded:
            try:
                _fsync_directory(parent)
            except OSError:
                pass


def write_target_manifest_qualification_only(
    *,
    target_manifest_path: str | Path,
    asset_manifest_path: str | Path,
    model_root: str | Path,
) -> dict[str, object]:
    """Create one canonical 0600 target manifest without overwriting."""

    absolute = _safe_absolute(
        target_manifest_path, field="target_manifest"
    )
    if (
        not absolute.parent.is_dir()
        or absolute.parent.is_symlink()
        or absolute.exists()
        or absolute.is_symlink()
    ):
        raise GSCLMiniLMPortableError(
            "target_manifest_parent_invalid"
        )
    raw = build_target_manifest_qualification_only(
        asset_manifest_path=asset_manifest_path,
        model_root=model_root,
    )
    read_path, observed, value = _publish_target_manifest_once(
        absolute, raw
    )
    return {
        "schema": GSCL_MINILM_TARGET_SCHEMA,
        "status": "qualified_target_manifest_written_once",
        "target_manifest_path": str(read_path),
        "target_manifest_file_sha256": _sha256_bytes(observed),
        "target_manifest_self_sha256": value["self_sha256"],
        "source_or_rows_accessed": False,
        "labels_accessed": False,
        "network_calls": 0,
    }


class GSCLPortableOfflineMiniLMEncoder:
    """Formal exact GSCL type bound to one qualified target manifest."""

    __slots__ = (
        "_authority_marker",
        "_canary_receipt_json",
        "_encoder",
        "_encoder_canary_receipt_json",
        "_encoder_runtime_receipt_json",
        "_runtime_receipt_json",
        "_target_manifest_file_sha256",
        "_target_manifest_path",
        "canary_receipt",
        "runtime_receipt",
    )

    def __init__(
        self,
        *,
        asset_manifest_path: str | Path,
        model_root: str | Path,
        target_manifest_path: str | Path,
        run_canary: bool = True,
    ) -> None:
        if run_canary is not True:
            raise GSCLMiniLMPortableError(
                "gscl_portable_canary_cannot_be_skipped"
            )
        manifest_path, target_raw = _secure_manifest_bytes(
            target_manifest_path
        )
        target = _decode_target_manifest(target_raw)
        try:
            encoder = PortableOfflineMiniLMEncoder(
                asset_manifest_path=asset_manifest_path,
                model_root=model_root,
                run_canary=True,
            )
        except PortableMiniLMError as exc:
            raise GSCLMiniLMPortableError(
                "portable_minilm_formal_binding_failed"
            ) from exc
        observed_raw = _encode_manifest(_manifest_body(encoder))
        if observed_raw != target_raw:
            raise GSCLMiniLMPortableError(
                "target_runtime_or_canary_drifted"
            )
        self._encoder = encoder
        runtime_receipt = {
            "schema": GSCL_MINILM_RUNTIME_SCHEMA,
            "status": "verified_exact_gscl_target_local_minilm_runtime",
            "target_manifest_path": str(manifest_path),
            "target_manifest_file_sha256": _sha256_bytes(target_raw),
            "target_manifest_self_sha256": target["self_sha256"],
            "portable_runtime_receipt": dict(encoder.runtime_receipt),
            "target_runtime_closure": target[
                "target_runtime_closure"
            ],
            "formal_source_or_rows_accessed": False,
            "labels_accessed": False,
            "network_calls": 0,
        }
        canary_receipt = {
            "schema": GSCL_MINILM_CANARY_SCHEMA,
            "status": "passed_target_local_repeat_exact_canary",
            "portable_canary_receipt": dict(encoder.canary_receipt),
            "target_observed_float32_sha256": target[
                "public_synthetic_canary"
            ]["target_observed_float32_sha256"],
            "target_observed_quantized_sha256": target[
                "public_synthetic_canary"
            ]["target_observed_quantized_sha256"],
            "target_manifest_self_sha256": target["self_sha256"],
            "repeat_count": 2,
            "repeat_byte_exact": True,
            "repeat_elementwise_exact": True,
            "cross_hardware_byte_identity_claimed": False,
            "formal_source_or_rows_accessed": False,
            "labels_accessed": False,
            "network_calls": 0,
        }
        self.runtime_receipt = MappingProxyType(runtime_receipt)
        self.canary_receipt = MappingProxyType(canary_receipt)
        self._target_manifest_path = manifest_path
        self._target_manifest_file_sha256 = _sha256_bytes(target_raw)
        self._runtime_receipt_json = _canonical_json_text(runtime_receipt)
        self._canary_receipt_json = _canonical_json_text(canary_receipt)
        self._encoder_runtime_receipt_json = _canonical_json_text(
            dict(encoder.runtime_receipt)
        )
        self._encoder_canary_receipt_json = _canonical_json_text(
            dict(encoder.canary_receipt)
        )
        self._authority_marker = _GSCL_ENCODER_AUTHORITY_MARKER
        self.validate_internal()

    def validate_internal(self) -> None:
        """Revalidate exact construction and immutable target-file custody."""

        if (
            type(self) is not GSCLPortableOfflineMiniLMEncoder
            or getattr(self, "_authority_marker", None)
            is not _GSCL_ENCODER_AUTHORITY_MARKER
            or type(getattr(self, "runtime_receipt", None))
            is not MappingProxyType
            or type(getattr(self, "canary_receipt", None))
            is not MappingProxyType
            or type(getattr(self, "_encoder", None))
            is not PortableOfflineMiniLMEncoder
        ):
            raise GSCLMiniLMPortableError(
                "formal_encoder_construction_not_authorized"
            )
        try:
            manifest_path, target_raw = _secure_manifest_bytes(
                self._target_manifest_path
            )
            target = _decode_target_manifest(target_raw)
            runtime_json = _canonical_json_text(dict(self.runtime_receipt))
            canary_json = _canonical_json_text(dict(self.canary_receipt))
            encoder_runtime_json = _canonical_json_text(
                dict(self._encoder.runtime_receipt)
            )
            encoder_canary_json = _canonical_json_text(
                dict(self._encoder.canary_receipt)
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise GSCLMiniLMPortableError(
                "formal_encoder_binding_changed"
            ) from exc
        if (
            manifest_path != self._target_manifest_path
            or _sha256_bytes(target_raw)
            != self._target_manifest_file_sha256
            or runtime_json != self._runtime_receipt_json
            or canary_json != self._canary_receipt_json
            or encoder_runtime_json
            != self._encoder_runtime_receipt_json
            or encoder_canary_json
            != self._encoder_canary_receipt_json
            or self.runtime_receipt.get("schema")
            != GSCL_MINILM_RUNTIME_SCHEMA
            or self.canary_receipt.get("schema")
            != GSCL_MINILM_CANARY_SCHEMA
            or self.runtime_receipt.get("target_manifest_file_sha256")
            != self._target_manifest_file_sha256
            or self.runtime_receipt.get("target_manifest_self_sha256")
            != target.get("self_sha256")
            or self.canary_receipt.get("target_manifest_self_sha256")
            != target.get("self_sha256")
            or self.canary_receipt.get("repeat_count") != 2
            or self.canary_receipt.get("repeat_byte_exact") is not True
            or self.canary_receipt.get("repeat_elementwise_exact") is not True
        ):
            raise GSCLMiniLMPortableError(
                "formal_encoder_binding_changed"
            )

    @property
    def tokenizer(self) -> object:
        return self._encoder._model.tokenizer  # type: ignore[attr-defined]

    @property
    def _model(self) -> object:
        return self._encoder._model  # type: ignore[attr-defined]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._encoder.encode(texts)

    def query_paragraph_similarities(
        self, query: str, paragraphs: Sequence[str]
    ) -> tuple[int, ...]:
        return frozen_v1.query_paragraph_similarities(
            self, query, paragraphs
        )


__all__ = [
    "GSCL_MINILM_CANARY_SCHEMA",
    "GSCL_MINILM_RUNTIME_SCHEMA",
    "GSCL_MINILM_TARGET_SCHEMA",
    "GSCLMiniLMPortableError",
    "GSCLPortableOfflineMiniLMEncoder",
    "build_target_manifest_qualification_only",
    "write_target_manifest_qualification_only",
]
