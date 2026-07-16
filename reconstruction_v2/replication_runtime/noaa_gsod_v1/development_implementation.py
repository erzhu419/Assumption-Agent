from __future__ import annotations

"""Finite source-file binding for the formal NOAA development implementation."""

from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .contract import (
    NoaaGsodError,
    payload_hash,
    sha256_file,
    verify_self_hash,
    with_self_hash,
)


IMPLEMENTATION_SET_VERSION = "noaa_gsod_development_implementation_set_v1"
IMPLEMENTATION_HASH_ALGORITHM = "sha256_file_rows_canonical_payload_v1"
IMPLEMENTATION_RELATIVE_PATHS = tuple(
    sorted(
        {
            "assumption_agent/__init__.py",
            "assumption_agent/models.py",
            "replication_runtime/__init__.py",
            "replication_runtime/financial_semantic_v2/__init__.py",
            "replication_runtime/financial_semantic_v2/durable_state.py",
            "replication_runtime/financial_semantic_v2/terminal_audit.py",
            "replication_runtime/noaa_gsod_v1/__init__.py",
            "replication_runtime/noaa_gsod_v1/contract.py",
            "replication_runtime/noaa_gsod_v1/development_freeze.py",
            "replication_runtime/noaa_gsod_v1/development_implementation.py",
            "replication_runtime/noaa_gsod_v1/development_runner.py",
            "replication_runtime/noaa_gsod_v1/development_schemas.py",
            "replication_runtime/noaa_gsod_v1/development_source.py",
            "replication_runtime/noaa_gsod_v1/oracle_sqlite.py",
            "replication_runtime/noaa_gsod_v1/oracle_stdlib.py",
            "replication_runtime/noaa_gsod_v1/pack.py",
            "replication_runtime/noaa_gsod_v1/schemas.py",
            "replication_runtime/noaa_gsod_v1/train_export.py",
            "replication_runtime/noaa_gsod_v1/train_schemas.py",
            "replication_runtime/noaa_gsod_v1/typed_relational.py",
        }
    )
)
IMPLEMENTATION_FILE_ROW_FIELDS = frozenset({"relative_path", "sha256"})
IMPLEMENTATION_SET_FIELDS = frozenset(
    {
        "file_count",
        "files",
        "fixed_relative_path_set_hash",
        "hash_algorithm",
        "implementation_set_hash",
        "implementation_set_version",
    }
)
FIXED_RELATIVE_PATH_SET_HASH = payload_hash(list(IMPLEMENTATION_RELATIVE_PATHS))


def _repository_root(repository_root: str | Path | None) -> Path:
    request = (
        Path(__file__).resolve().parents[2]
        if repository_root is None
        else Path(repository_root)
    )
    if request.is_symlink():
        raise NoaaGsodError("implementation repository root is a symbolic link")
    try:
        root = request.resolve(strict=True)
    except OSError as exc:
        raise NoaaGsodError("implementation repository root is missing") from exc
    if not root.is_dir():
        raise NoaaGsodError("implementation repository root is not a directory")
    return root


def _fixed_regular_file(root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise NoaaGsodError("implementation relative path is not fixed beneath root")
    candidate = root
    for part in pure.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise NoaaGsodError(
                f"implementation dependency is a symbolic link: {relative_path}"
            )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise NoaaGsodError(
            f"implementation dependency is missing: {relative_path}"
        ) from exc
    if root not in resolved.parents or not resolved.is_file():
        raise NoaaGsodError(
            f"implementation dependency is not a regular file: {relative_path}"
        )
    return resolved


def build_development_implementation_set(
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Hash exactly the fixed dependency list; callers cannot add inputs."""

    root = _repository_root(repository_root)
    rows = [
        {
            "relative_path": relative_path,
            "sha256": sha256_file(_fixed_regular_file(root, relative_path)),
        }
        for relative_path in IMPLEMENTATION_RELATIVE_PATHS
    ]
    body: dict[str, Any] = {
        "file_count": len(IMPLEMENTATION_RELATIVE_PATHS),
        "files": rows,
        "fixed_relative_path_set_hash": FIXED_RELATIVE_PATH_SET_HASH,
        "hash_algorithm": IMPLEMENTATION_HASH_ALGORITHM,
        "implementation_set_version": IMPLEMENTATION_SET_VERSION,
    }
    return with_self_hash(body, "implementation_set_hash")


def verify_development_implementation_set(
    payload: Mapping[str, Any],
    *,
    repository_root: str | Path | None = None,
    verify_live_files: bool = False,
) -> dict[str, Any]:
    """Verify receipt structure, optionally comparing the current file bytes.

    Historical freeze verification must leave ``verify_live_files`` false; a
    historical binding describes the materialized implementation, not whatever
    code happens to be installed later.
    """

    receipt = dict(payload)
    verify_self_hash(receipt, "implementation_set_hash")
    if set(receipt) != IMPLEMENTATION_SET_FIELDS:
        raise NoaaGsodError("development implementation set schema mismatch")
    if (
        receipt.get("implementation_set_version") != IMPLEMENTATION_SET_VERSION
        or receipt.get("hash_algorithm") != IMPLEMENTATION_HASH_ALGORITHM
        or receipt.get("fixed_relative_path_set_hash")
        != FIXED_RELATIVE_PATH_SET_HASH
        or type(receipt.get("file_count")) is not int
        or receipt.get("file_count") != len(IMPLEMENTATION_RELATIVE_PATHS)
    ):
        raise NoaaGsodError("development implementation set identity mismatch")
    rows = receipt.get("files")
    if not isinstance(rows, list) or len(rows) != len(IMPLEMENTATION_RELATIVE_PATHS):
        raise NoaaGsodError("development implementation file row count mismatch")
    for expected_path, row in zip(IMPLEMENTATION_RELATIVE_PATHS, rows):
        if not isinstance(row, dict) or set(row) != IMPLEMENTATION_FILE_ROW_FIELDS:
            raise NoaaGsodError("development implementation file row schema mismatch")
        digest = row.get("sha256")
        if (
            row.get("relative_path") != expected_path
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise NoaaGsodError("development implementation file row mismatch")
    if verify_live_files:
        live = build_development_implementation_set(repository_root=repository_root)
        if live != receipt:
            raise NoaaGsodError("live development implementation differs from receipt")
    return receipt


__all__ = [
    "FIXED_RELATIVE_PATH_SET_HASH",
    "IMPLEMENTATION_HASH_ALGORITHM",
    "IMPLEMENTATION_RELATIVE_PATHS",
    "IMPLEMENTATION_SET_VERSION",
    "build_development_implementation_set",
    "verify_development_implementation_set",
]
