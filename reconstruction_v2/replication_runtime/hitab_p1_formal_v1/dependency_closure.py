"""Content-addressed, stdlib-only filesystem closure for HiTab production.

The production runner imports this module before adding either frozen
site-packages root to ``sys.path``.  It consequently must remain independent
of every third-party package and every other project module.

The tree digest commits regular-file bytes, entry names, permission bits,
each encountered symlink's lexical target, and its strictly resolved final
target path and bytes.  A symlink is never followed by a directory walker.
Its target is instead hashed explicitly; directory-link cycles are rejected.
This makes editable installs and package trees content-bound without allowing
``.pth`` files to execute before verification.  External editable source
roots referenced by a ``.pth`` file still require their own explicit receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Mapping


VERSION = "hitab_p1_dependency_closure_v1"
TREE_RECEIPT_SCHEMA = f"{VERSION}_tree_receipt"
FILE_RECEIPT_SCHEMA = f"{VERSION}_regular_file_receipt"
TREE_DIGEST_ALGORITHM = (
    "sha256_canonical_json_recursive_names_modes_regular_bytes_and_"
    "strict_symlink_targets_v1"
)

_TREE_RECEIPT_KEYS = frozenset(
    {
        "algorithm",
        "directory_count",
        "regular_file_count",
        "regular_file_size_bytes",
        "schema",
        "symlink_count",
        "tree_sha256",
    }
)
_FILE_RECEIPT_KEYS = frozenset(
    {
        "content_sha256",
        "mode",
        "schema",
        "size_bytes",
    }
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class DependencyClosureError(RuntimeError):
    """A frozen dependency path is unsafe, unreadable, or drifted."""


@dataclass(frozen=True)
class _NodeReceipt:
    node_sha256: str
    directory_count: int
    regular_file_count: int
    regular_file_size_bytes: int
    symlink_count: int


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
        raise DependencyClosureError(
            "dependency closure value is not canonical JSON"
        ) from exc


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _lstat(path: Path) -> os.stat_result:
    try:
        return os.lstat(path)
    except OSError as exc:
        raise DependencyClosureError(
            "dependency closure entry is unavailable"
        ) from exc


def _absolute_direct_directory(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise DependencyClosureError(
            "dependency tree root is not absolute"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DependencyClosureError(
            "dependency tree root is unavailable"
        ) from exc
    metadata = _lstat(candidate)
    if (
        resolved != candidate
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise DependencyClosureError(
            "dependency tree root is not a direct directory"
        )
    return candidate


def _absolute_direct_file(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise DependencyClosureError(
            "dependency file is not absolute"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DependencyClosureError(
            "dependency file is unavailable"
        ) from exc
    metadata = _lstat(candidate)
    if (
        resolved != candidate
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise DependencyClosureError(
            "dependency file is not a direct regular file"
        )
    return candidate


def _hash_regular_file(
    path: Path,
    *,
    expected_metadata: os.stat_result | None = None,
) -> tuple[str, int, int]:
    before = _lstat(path)
    if expected_metadata is not None and _stat_signature(
        before
    ) != _stat_signature(expected_metadata):
        raise DependencyClosureError(
            "dependency file changed before it was opened"
        )
    if not stat.S_ISREG(before.st_mode):
        raise DependencyClosureError(
            "dependency closure expected a regular file"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DependencyClosureError(
            "dependency file could not be opened safely"
        ) from exc
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _stat_signature(opened) != _stat_signature(before)
        ):
            raise DependencyClosureError(
                "dependency file identity drifted while opened"
            )
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            digest.update(block)
        after_open = os.fstat(descriptor)
    except OSError as exc:
        raise DependencyClosureError(
            "dependency file could not be read safely"
        ) from exc
    finally:
        os.close(descriptor)
    after_path = _lstat(path)
    signature = _stat_signature(before)
    if (
        _stat_signature(after_open) != signature
        or _stat_signature(after_path) != signature
    ):
        raise DependencyClosureError(
            "dependency file changed while it was hashed"
        )
    return (
        digest.hexdigest(),
        before.st_size,
        stat.S_IMODE(before.st_mode),
    )


def regular_file_receipt(path: Path) -> dict[str, object]:
    """Return an exact receipt for one direct, nonsymlink regular file."""

    candidate = _absolute_direct_file(Path(path))
    digest, size, mode = _hash_regular_file(candidate)
    return {
        "content_sha256": digest,
        "mode": mode,
        "schema": FILE_RECEIPT_SCHEMA,
        "size_bytes": size,
    }


def verify_regular_file_receipt(
    path: Path, expected: Mapping[str, object]
) -> dict[str, object]:
    """Recompute and exactly compare a frozen regular-file receipt."""

    value = dict(expected)
    if set(value) != _FILE_RECEIPT_KEYS:
        raise DependencyClosureError(
            "dependency file receipt shape drifted"
        )
    if (
        value.get("schema") != FILE_RECEIPT_SCHEMA
        or not isinstance(value.get("content_sha256"), str)
        or _HEX64.fullmatch(str(value["content_sha256"])) is None
        or type(value.get("mode")) is not int
        or not 0 <= int(value["mode"]) <= 0o7777
        or type(value.get("size_bytes")) is not int
        or int(value["size_bytes"]) < 0
    ):
        raise DependencyClosureError(
            "dependency file receipt values drifted"
        )
    actual = regular_file_receipt(Path(path))
    if actual != value:
        raise DependencyClosureError("dependency file receipt drifted")
    return actual


def _regular_node(
    path: Path, metadata: os.stat_result
) -> _NodeReceipt:
    digest, size, mode = _hash_regular_file(
        path, expected_metadata=metadata
    )
    node = {
        "content_sha256": digest,
        "kind": "regular_file",
        "mode": mode,
        "size_bytes": size,
    }
    return _NodeReceipt(
        node_sha256=_stable_hash(node),
        directory_count=0,
        regular_file_count=1,
        regular_file_size_bytes=size,
        symlink_count=0,
    )


def _directory_node(
    path: Path,
    metadata: os.stat_result,
    *,
    active_directories: frozenset[tuple[int, int]],
) -> _NodeReceipt:
    identity = (metadata.st_dev, metadata.st_ino)
    if identity in active_directories:
        raise DependencyClosureError(
            "dependency tree contains a directory symlink cycle"
        )
    before = _stat_signature(metadata)
    try:
        with os.scandir(path) as iterator:
            names = [entry.name for entry in iterator]
    except OSError as exc:
        raise DependencyClosureError(
            "dependency directory could not be scanned"
        ) from exc
    names.sort(key=os.fsencode)
    rows: list[dict[str, object]] = []
    directory_count = 1
    regular_file_count = 0
    regular_file_size_bytes = 0
    symlink_count = 0
    next_active = active_directories | {identity}
    for name in names:
        child = _node(
            path / name,
            active_directories=next_active,
        )
        rows.append(
            {
                "name_bytes_hex": os.fsencode(name).hex(),
                "node_sha256": child.node_sha256,
            }
        )
        directory_count += child.directory_count
        regular_file_count += child.regular_file_count
        regular_file_size_bytes += child.regular_file_size_bytes
        symlink_count += child.symlink_count
    after = _lstat(path)
    if (
        not stat.S_ISDIR(after.st_mode)
        or _stat_signature(after) != before
    ):
        raise DependencyClosureError(
            "dependency directory changed while it was hashed"
        )
    node = {
        "entries": rows,
        "kind": "directory",
        "mode": stat.S_IMODE(metadata.st_mode),
    }
    return _NodeReceipt(
        node_sha256=_stable_hash(node),
        directory_count=directory_count,
        regular_file_count=regular_file_count,
        regular_file_size_bytes=regular_file_size_bytes,
        symlink_count=symlink_count,
    )


def _symlink_node(
    path: Path,
    metadata: os.stat_result,
    *,
    active_directories: frozenset[tuple[int, int]],
) -> _NodeReceipt:
    before = _stat_signature(metadata)
    try:
        link_bytes = os.readlink(os.fsencode(path))
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DependencyClosureError(
            "dependency symlink is dangling or cyclic"
        ) from exc
    target_metadata = _lstat(resolved)
    if stat.S_ISREG(target_metadata.st_mode):
        target = _regular_node(resolved, target_metadata)
        target_kind = "regular_file"
    elif stat.S_ISDIR(target_metadata.st_mode):
        target = _directory_node(
            resolved,
            target_metadata,
            active_directories=active_directories,
        )
        target_kind = "directory"
    else:
        raise DependencyClosureError(
            "dependency symlink targets a special file"
        )
    after = _lstat(path)
    try:
        after_link_bytes = os.readlink(os.fsencode(path))
        after_resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DependencyClosureError(
            "dependency symlink changed while it was hashed"
        ) from exc
    if (
        not stat.S_ISLNK(after.st_mode)
        or _stat_signature(after) != before
        or after_link_bytes != link_bytes
        or after_resolved != resolved
    ):
        raise DependencyClosureError(
            "dependency symlink changed while it was hashed"
        )
    node = {
        "kind": "symlink",
        "link_bytes_hex": link_bytes.hex(),
        "resolved_target_bytes_hex": os.fsencode(resolved).hex(),
        "target_kind": target_kind,
        "target_node_sha256": target.node_sha256,
    }
    return _NodeReceipt(
        node_sha256=_stable_hash(node),
        directory_count=target.directory_count,
        regular_file_count=target.regular_file_count,
        regular_file_size_bytes=target.regular_file_size_bytes,
        symlink_count=target.symlink_count + 1,
    )


def _node(
    path: Path,
    *,
    active_directories: frozenset[tuple[int, int]],
) -> _NodeReceipt:
    metadata = _lstat(path)
    if stat.S_ISREG(metadata.st_mode):
        return _regular_node(path, metadata)
    if stat.S_ISDIR(metadata.st_mode):
        return _directory_node(
            path,
            metadata,
            active_directories=active_directories,
        )
    if stat.S_ISLNK(metadata.st_mode):
        return _symlink_node(
            path,
            metadata,
            active_directories=active_directories,
        )
    raise DependencyClosureError(
        "dependency tree contains a special filesystem entry"
    )


def tree_receipt(root: Path) -> dict[str, object]:
    """Hash one actual dependency tree without executing package code."""

    candidate = _absolute_direct_directory(Path(root))
    metadata = _lstat(candidate)
    result = _directory_node(
        candidate,
        metadata,
        active_directories=frozenset(),
    )
    return {
        "algorithm": TREE_DIGEST_ALGORITHM,
        "directory_count": result.directory_count,
        "regular_file_count": result.regular_file_count,
        "regular_file_size_bytes": result.regular_file_size_bytes,
        "schema": TREE_RECEIPT_SCHEMA,
        "symlink_count": result.symlink_count,
        "tree_sha256": result.node_sha256,
    }


def verify_tree_receipt(
    root: Path, expected: Mapping[str, object]
) -> dict[str, object]:
    """Recompute and exactly compare one frozen dependency-tree receipt."""

    value = dict(expected)
    if set(value) != _TREE_RECEIPT_KEYS:
        raise DependencyClosureError(
            "dependency tree receipt shape drifted"
        )
    integer_fields = (
        "directory_count",
        "regular_file_count",
        "regular_file_size_bytes",
        "symlink_count",
    )
    if (
        value.get("schema") != TREE_RECEIPT_SCHEMA
        or value.get("algorithm") != TREE_DIGEST_ALGORITHM
        or not isinstance(value.get("tree_sha256"), str)
        or _HEX64.fullmatch(str(value["tree_sha256"])) is None
        or any(
            type(value.get(field)) is not int
            or int(value[field]) < 0
            for field in integer_fields
        )
        or int(value["directory_count"]) < 1
    ):
        raise DependencyClosureError(
            "dependency tree receipt values drifted"
        )
    actual = tree_receipt(Path(root))
    if actual != value:
        raise DependencyClosureError("dependency tree receipt drifted")
    return actual


__all__ = [
    "DependencyClosureError",
    "FILE_RECEIPT_SCHEMA",
    "TREE_DIGEST_ALGORITHM",
    "TREE_RECEIPT_SCHEMA",
    "VERSION",
    "regular_file_receipt",
    "tree_receipt",
    "verify_regular_file_receipt",
    "verify_tree_receipt",
]
