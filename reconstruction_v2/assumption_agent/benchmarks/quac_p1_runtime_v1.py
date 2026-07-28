"""Source-free two-lane runtime orchestration for QuAC P1.

This module owns only label-free execution:

* verify two independent frozen Python runtimes and shared model/source trees;
* encode the complete block corpus, all full queries, and all per-turn queries
  in one deduplicated GPU0 MiniLM bulk call;
* build every frozen QuAC action graph from those embeddings;
* for authorized measurement blocks, submit one GPU1 official-HippoRAG block
  process concurrently with the GPU0 lane and join both only after submission;
* seal private artifacts with exclusive creation, move the official index
  from scratch into a fixed read-only private archive, and return a safe
  aggregate receipt.

There is no source loader and no field for a split, family, qrel, answer,
label, utility, score, evaluator outcome, or query-to-native-context link.
Failures consume the sole work-root attempt.  Nothing in this module retries,
replays, resamples, changes a model, or falls back online.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys
import threading
from typing import Any, Mapping, Protocol, Sequence

from assumption_agent.benchmarks import quac_p1_action_adapter_v1 as action
from replication_runtime.quac_p1_official_v1 import contract as official_contract


VERSION = "quac_p1_runtime_v1"
BLOCK_SCHEMA = f"{VERSION}_private_block_v1"
ATTEMPT_SCHEMA = f"{VERSION}_private_attempt_v1"
ACTION_PACK_SCHEMA = f"{VERSION}_private_action_pack_v1"
MINILM_RECEIPT_SCHEMA = f"{VERSION}_private_minilm_receipt_v1"
SAFE_RESULT_SCHEMA = f"{VERSION}_safe_result_v1"
SAFE_FAILURE_SCHEMA = f"{VERSION}_safe_failure_v1"
VERIFIED_BINDINGS_SCHEMA = f"{VERSION}_verified_bindings_v1"
MAX_BLOCK_QUERIES = 512
MINILM_BATCH_SIZE = 64
MINILM_DEVICE = "cuda:0"
GPU0 = "0"
GPU1 = "1"
BLOCK_ROLES = ("A_form", "A_hold", "M_search")
OFFICIAL_BLOCK_ROLES = frozenset({"A_hold", "M_search"})
OFFICIAL_WORKER_TIMEOUT_SECONDS = 86_400
_PROJECT_IMPORT_ROOT = Path(__file__).resolve().parents[2]
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_BLOCK_KEYS = frozenset({"block_id", "documents", "queries", "schema"})
_DOCUMENT_KEYS = frozenset(
    {
        "context_id",
        "context_window_ordinal",
        "section_title",
        "text",
        "title",
        "unit_id",
    }
)
_QUERY_KEYS = frozenset({"query_id", "question_turns"})
_TURN_KEYS = frozenset({"question_text"})
_NATIVE_THREAD_KEYS = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class QuacP1RuntimeError(RuntimeError):
    """A frozen asset, block, lane, archive, or one-shot boundary drifted."""


def canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise QuacP1RuntimeError(
            "runtime value is not canonical JSON"
        ) from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value, newline=False)).hexdigest()


def _self_hashed(body: Mapping[str, object]) -> dict[str, object]:
    value = dict(body)
    if "self_sha256" in value:
        raise QuacP1RuntimeError("runtime self hash already exists")
    value["self_sha256"] = stable_hash(value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise QuacP1RuntimeError("frozen file cannot be read") from exc
    return digest.hexdigest()


def _opaque_id(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise QuacP1RuntimeError(
            f"{field} must be an opaque lowercase SHA-256 ID"
        )
    return value


def _model_alias(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or _MODEL_ALIAS.fullmatch(value) is None
        or "/" in value
        or "\\" in value
        or ".." in value
    ):
        raise QuacP1RuntimeError(f"{field} model alias drifted")
    return value


def _private_directory(path: Path, *, fresh: bool) -> None:
    try:
        path.mkdir(parents=True, mode=0o700, exist_ok=not fresh)
        os.chmod(path, 0o700)
    except OSError as exc:
        raise QuacP1RuntimeError(
            "private runtime directory cannot be created"
        ) from exc
    if (
        path.is_symlink()
        or not path.is_dir()
        or stat.S_IMODE(path.stat().st_mode) != 0o700
    ):
        raise QuacP1RuntimeError("private runtime directory mode drifted")


def _write_once(
    path: Path,
    raw: bytes,
    *,
    final_mode: int,
) -> str:
    if final_mode not in (0o400, 0o600):
        raise QuacP1RuntimeError("private final mode drifted")
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, final_mode)
    except OSError as exc:
        raise QuacP1RuntimeError(
            "private runtime artifact cannot be written once"
        ) from exc
    try:
        metadata = path.lstat()
        persisted = path.read_bytes()
    except OSError as exc:
        raise QuacP1RuntimeError(
            "private runtime artifact cannot be verified"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != final_mode
        or persisted != raw
    ):
        raise QuacP1RuntimeError(
            "private runtime artifact verification failed"
        )
    return hashlib.sha256(raw).hexdigest()


def _write_json_once(
    path: Path,
    value: Mapping[str, object],
    *,
    final_mode: int,
) -> str:
    return _write_once(
        path,
        canonical_bytes(value),
        final_mode=final_mode,
    )


def _snapshot_tree(root: Path) -> tuple[str, int, int]:
    """Hash a direct, symlink-free regular-file tree."""

    try:
        absolute = root.absolute()
        root_info = absolute.lstat()
    except OSError as exc:
        raise QuacP1RuntimeError("frozen tree is unavailable") from exc
    if (
        absolute.is_symlink()
        or not stat.S_ISDIR(root_info.st_mode)
    ):
        raise QuacP1RuntimeError(
            "frozen tree root must be a direct directory"
        )
    rows: list[dict[str, object]] = []
    file_count = 0
    total_bytes = 0
    try:
        entries = sorted(
            absolute.rglob("*"),
            key=lambda row: row.relative_to(absolute).as_posix(),
        )
        for entry in entries:
            info = entry.lstat()
            relative = entry.relative_to(absolute).as_posix()
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                raise QuacP1RuntimeError(
                    "frozen tree contains a symbolic link"
                )
            if stat.S_ISDIR(info.st_mode):
                rows.append(
                    {"kind": "directory", "mode": mode, "path": relative}
                )
                continue
            if not stat.S_ISREG(info.st_mode):
                raise QuacP1RuntimeError(
                    "frozen tree contains a special file"
                )
            size = info.st_size
            rows.append(
                {
                    "kind": "file",
                    "mode": mode,
                    "path": relative,
                    "sha256": _sha256_file(entry),
                    "size": size,
                }
            )
            file_count += 1
            total_bytes += size
    except OSError as exc:
        raise QuacP1RuntimeError("frozen tree cannot be traversed") from exc
    if file_count < 1:
        raise QuacP1RuntimeError("frozen tree contains no regular file")
    return stable_hash(rows), file_count, total_bytes


def _seal_private_tree_once(
    source_root: Path,
    archive_root: Path,
) -> tuple[str, int, int]:
    """Atomically retain one recursively read-only private tree.

    The source tree is first validated without changing it, then every regular
    file and descendant directory is hardened in place.  Its complete snapshot
    is taken before a same-filesystem ``os.replace`` into the fixed archive
    path.  The moved root is immediately hardened and the whole tree is
    reverified.  The old scratch name must disappear.
    """

    source = source_root.absolute()
    archive = archive_root.absolute()
    if (
        source == archive
        or archive.exists()
        or archive.is_symlink()
    ):
        raise QuacP1RuntimeError(
            "private tree archive path is not fresh"
        )
    try:
        source_info = source.lstat()
        archive_parent_info = archive.parent.lstat()
    except OSError as exc:
        raise QuacP1RuntimeError(
            "private tree archive boundary is unavailable"
        ) from exc
    if (
        source.is_symlink()
        or not stat.S_ISDIR(source_info.st_mode)
        or archive.parent.is_symlink()
        or not stat.S_ISDIR(archive_parent_info.st_mode)
    ):
        raise QuacP1RuntimeError(
            "private tree archive boundary drifted"
        )
    if source_info.st_dev != archive_parent_info.st_dev:
        raise QuacP1RuntimeError(
            "private tree archive must use the source filesystem"
        )

    try:
        entries = sorted(
            source.rglob("*"),
            key=lambda row: row.relative_to(source).as_posix(),
        )
        directories: list[Path] = []
        regular_files: list[Path] = []
        for entry in entries:
            info = entry.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise QuacP1RuntimeError(
                    "private tree contains a symbolic link"
                )
            if stat.S_ISDIR(info.st_mode):
                directories.append(entry)
            elif stat.S_ISREG(info.st_mode):
                regular_files.append(entry)
            else:
                raise QuacP1RuntimeError(
                    "private tree contains a special file"
                )
    except OSError as exc:
        raise QuacP1RuntimeError(
            "private tree cannot be traversed"
        ) from exc
    if not regular_files:
        raise QuacP1RuntimeError(
            "private tree contains no regular file"
        )

    try:
        for entry in regular_files:
            os.chmod(entry, 0o400, follow_symlinks=False)
        for entry in sorted(
            directories,
            key=lambda row: len(row.relative_to(source).parts),
            reverse=True,
        ):
            os.chmod(entry, 0o500, follow_symlinks=False)
    except OSError as exc:
        raise QuacP1RuntimeError(
            "private tree cannot be hardened"
        ) from exc

    def verify_modes(root: Path, *, expected_root_mode: int) -> None:
        try:
            root_metadata = root.lstat()
            observed_entries = sorted(
                root.rglob("*"),
                key=lambda row: row.relative_to(root).as_posix(),
            )
            if (
                root.is_symlink()
                or not stat.S_ISDIR(root_metadata.st_mode)
                or stat.S_IMODE(root_metadata.st_mode)
                != expected_root_mode
            ):
                raise QuacP1RuntimeError(
                    "private tree root mode drifted"
                )
            for entry in observed_entries:
                info = entry.lstat()
                if stat.S_ISLNK(info.st_mode):
                    raise QuacP1RuntimeError(
                        "private tree contains a symbolic link"
                    )
                expected_mode = (
                    0o500 if stat.S_ISDIR(info.st_mode) else 0o400
                )
                if (
                    not (
                        stat.S_ISDIR(info.st_mode)
                        or stat.S_ISREG(info.st_mode)
                    )
                    or stat.S_IMODE(info.st_mode) != expected_mode
                ):
                    raise QuacP1RuntimeError(
                        "private tree entry mode drifted"
                    )
        except OSError as exc:
            raise QuacP1RuntimeError(
                "private tree modes cannot be verified"
            ) from exc

    verify_modes(source, expected_root_mode=0o700)
    before = _snapshot_tree(source)
    try:
        os.replace(source, archive)
        os.chmod(archive, 0o500, follow_symlinks=False)
    except OSError as exc:
        raise QuacP1RuntimeError(
            "private tree archive move failed"
        ) from exc
    if source.exists() or source.is_symlink():
        raise QuacP1RuntimeError(
            "private tree scratch path survived archive move"
        )
    verify_modes(archive, expected_root_mode=0o500)
    after = _snapshot_tree(archive)
    if after != before:
        raise QuacP1RuntimeError(
            "private tree archive verification mismatched"
        )
    return before


@dataclass(frozen=True)
class FrozenTreeBinding:
    """Exact path and complete regular-file-tree identity."""

    path: str
    tree_sha256: str
    file_count: int
    total_bytes: int

    @classmethod
    def capture(cls, path: Path) -> "FrozenTreeBinding":
        absolute = path.absolute()
        digest, count, total = _snapshot_tree(absolute)
        return cls(str(absolute), digest, count, total)

    def verify(self) -> dict[str, object]:
        if (
            not isinstance(self.path, str)
            or not Path(self.path).is_absolute()
            or not isinstance(self.tree_sha256, str)
            or _HEX64.fullmatch(self.tree_sha256) is None
            or type(self.file_count) is not int
            or self.file_count < 1
            or type(self.total_bytes) is not int
            or self.total_bytes < 0
        ):
            raise QuacP1RuntimeError("frozen tree binding shape drifted")
        observed = _snapshot_tree(Path(self.path))
        expected = (
            self.tree_sha256,
            self.file_count,
            self.total_bytes,
        )
        if observed != expected:
            raise QuacP1RuntimeError("frozen tree binding mismatched")
        return {
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "tree_sha256": self.tree_sha256,
        }

    def semantic_payload(self) -> dict[str, object]:
        return {
            "file_count": self.file_count,
            "path": self.path,
            "total_bytes": self.total_bytes,
            "tree_sha256": self.tree_sha256,
        }


@dataclass(frozen=True)
class FrozenExecutableBinding:
    """Lexical executable path plus resolved regular-file identity."""

    path: str
    realpath: str
    sha256: str
    size_bytes: int

    @classmethod
    def capture(cls, path: Path) -> "FrozenExecutableBinding":
        lexical = path.absolute()
        try:
            resolved = lexical.resolve(strict=True)
            info = resolved.stat()
        except OSError as exc:
            raise QuacP1RuntimeError(
                "Python executable is unavailable"
            ) from exc
        if not stat.S_ISREG(info.st_mode) or not info.st_mode & 0o111:
            raise QuacP1RuntimeError(
                "Python executable target is not executable"
            )
        return cls(
            path=str(lexical),
            realpath=str(resolved),
            sha256=_sha256_file(resolved),
            size_bytes=info.st_size,
        )

    def verify(self) -> dict[str, object]:
        if (
            not isinstance(self.path, str)
            or not Path(self.path).is_absolute()
            or not isinstance(self.realpath, str)
            or not Path(self.realpath).is_absolute()
            or not isinstance(self.sha256, str)
            or _HEX64.fullmatch(self.sha256) is None
            or type(self.size_bytes) is not int
            or self.size_bytes <= 0
        ):
            raise QuacP1RuntimeError(
                "Python executable binding shape drifted"
            )
        try:
            resolved = Path(self.path).resolve(strict=True)
            info = resolved.stat()
        except OSError as exc:
            raise QuacP1RuntimeError(
                "Python executable binding is unavailable"
            ) from exc
        if (
            str(resolved) != self.realpath
            or not stat.S_ISREG(info.st_mode)
            or not info.st_mode & 0o111
            or info.st_size != self.size_bytes
            or _sha256_file(resolved) != self.sha256
        ):
            raise QuacP1RuntimeError(
                "Python executable binding mismatched"
            )
        return {
            "file_sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    def semantic_payload(self) -> dict[str, object]:
        return {
            "path": self.path,
            "realpath": self.realpath,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class PythonRuntimeBinding:
    """One lexical Python plus its independent frozen import tree."""

    executable: FrozenExecutableBinding
    import_tree: FrozenTreeBinding
    identity_sha256: str

    @classmethod
    def capture(
        cls,
        *,
        executable: Path,
        import_tree: Path,
    ) -> "PythonRuntimeBinding":
        executable_binding = FrozenExecutableBinding.capture(executable)
        tree_binding = FrozenTreeBinding.capture(import_tree)
        body = {
            "executable": executable_binding.semantic_payload(),
            "import_tree": tree_binding.semantic_payload(),
        }
        return cls(
            executable_binding,
            tree_binding,
            stable_hash(body),
        )

    def verify(self) -> dict[str, object]:
        if (
            not isinstance(self.executable, FrozenExecutableBinding)
            or not isinstance(self.import_tree, FrozenTreeBinding)
            or not isinstance(self.identity_sha256, str)
            or _HEX64.fullmatch(self.identity_sha256) is None
        ):
            raise QuacP1RuntimeError("Python runtime binding shape drifted")
        body = {
            "executable": self.executable.semantic_payload(),
            "import_tree": self.import_tree.semantic_payload(),
        }
        if stable_hash(body) != self.identity_sha256:
            raise QuacP1RuntimeError(
                "Python runtime semantic binding drifted"
            )
        executable_receipt = self.executable.verify()
        tree_receipt = self.import_tree.verify()
        return {
            "executable": executable_receipt,
            "identity_sha256": self.identity_sha256,
            "import_tree": tree_receipt,
        }


@dataclass(frozen=True)
class RuntimeBindings:
    """Two independent runtimes and three shared frozen asset trees."""

    gpu0_python: PythonRuntimeBinding
    gpu1_python: PythonRuntimeBinding
    gpu1_overlay_import_tree: FrozenTreeBinding
    gpu1_base_import_tree: FrozenTreeBinding
    minilm_asset: FrozenTreeBinding
    llm_asset: FrozenTreeBinding
    hipporag_source: FrozenTreeBinding
    minilm_alias: str = "minilm"
    llm_alias: str = "smollm2"

    def semantic_payload(self) -> dict[str, object]:
        if (
            not isinstance(self.gpu0_python, PythonRuntimeBinding)
            or not isinstance(self.gpu1_python, PythonRuntimeBinding)
            or not isinstance(
                self.gpu1_overlay_import_tree,
                FrozenTreeBinding,
            )
            or not isinstance(
                self.gpu1_base_import_tree,
                FrozenTreeBinding,
            )
            or not isinstance(self.minilm_asset, FrozenTreeBinding)
            or not isinstance(self.llm_asset, FrozenTreeBinding)
            or not isinstance(self.hipporag_source, FrozenTreeBinding)
        ):
            raise QuacP1RuntimeError("runtime binding shape drifted")
        minilm_alias = _model_alias(self.minilm_alias, "MiniLM")
        llm_alias = _model_alias(self.llm_alias, "LLM")
        if (
            minilm_alias == llm_alias
            or self.minilm_asset.path == self.llm_asset.path
        ):
            raise QuacP1RuntimeError(
                "frozen model assets or aliases are not distinct"
            )
        if (
            self.gpu0_python.executable.path
            == self.gpu1_python.executable.path
            or self.gpu0_python.import_tree.path
            == self.gpu1_python.import_tree.path
            or self.gpu1_overlay_import_tree.path
            in {
                self.gpu0_python.import_tree.path,
                self.gpu1_python.import_tree.path,
            }
            or self.gpu1_base_import_tree.path
            in {
                self.gpu0_python.import_tree.path,
                self.gpu1_python.import_tree.path,
                self.gpu1_overlay_import_tree.path,
            }
        ):
            raise QuacP1RuntimeError(
                "GPU0 and GPU1 Python runtimes are not independent"
            )
        return {
            "assets": {
                "hipporag_source": self.hipporag_source.semantic_payload(),
                "llm_asset": self.llm_asset.semantic_payload(),
                "minilm_asset": self.minilm_asset.semantic_payload(),
            },
            "gpu0_python_identity_sha256": (
                self.gpu0_python.identity_sha256
            ),
            "gpu1_python_identity_sha256": (
                self.gpu1_python.identity_sha256
            ),
            "gpu1_overlay_import_tree": (
                self.gpu1_overlay_import_tree.semantic_payload()
            ),
            "gpu1_base_import_tree": (
                self.gpu1_base_import_tree.semantic_payload()
            ),
            "llm_alias": llm_alias,
            "minilm_alias": minilm_alias,
        }

    def verify(self) -> dict[str, object]:
        """Perform the expensive full-tree verification exactly once."""

        semantic = self.semantic_payload()
        gpu0 = self.gpu0_python.verify()
        gpu1 = self.gpu1_python.verify()
        assets = {
            "gpu1_base_import_tree": (
                self.gpu1_base_import_tree.verify()
            ),
            "gpu1_overlay_import_tree": (
                self.gpu1_overlay_import_tree.verify()
            ),
            "hipporag_source": self.hipporag_source.verify(),
            "llm_asset": self.llm_asset.verify(),
            "minilm_asset": self.minilm_asset.verify(),
        }
        return {
            "assets": assets,
            "binding_sha256": stable_hash(semantic),
            "gpu0_python": gpu0,
            "gpu1_python": gpu1,
            "llm_alias": semantic["llm_alias"],
            "minilm_alias": semantic["minilm_alias"],
        }


_VERIFIED_BINDINGS_AUTHORITY = object()


class VerifiedRuntimeBindings:
    """Immutable in-process proof of one pre-source full verification."""

    __slots__ = (
        "_authority",
        "_bindings",
        "_canonical_receipt",
        "_token_sha256",
    )

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise QuacP1RuntimeError(
            "verified runtime bindings require the verifier factory"
        )

    @classmethod
    def _issue(
        cls,
        *,
        bindings: RuntimeBindings,
        receipt: Mapping[str, object],
    ) -> "VerifiedRuntimeBindings":
        instance = object.__new__(cls)
        raw = canonical_bytes(receipt)
        object.__setattr__(
            instance,
            "_authority",
            _VERIFIED_BINDINGS_AUTHORITY,
        )
        object.__setattr__(instance, "_bindings", bindings)
        object.__setattr__(instance, "_canonical_receipt", raw)
        object.__setattr__(
            instance,
            "_token_sha256",
            hashlib.sha256(raw).hexdigest(),
        )
        return instance

    def __setattr__(self, _name: str, _value: object) -> None:
        raise QuacP1RuntimeError(
            "verified runtime bindings are immutable"
        )

    @property
    def token_sha256(self) -> str:
        return self._token_sha256

    @property
    def canonical_receipt(self) -> bytes:
        return bytes(self._canonical_receipt)

    def require(
        self,
        bindings: RuntimeBindings,
    ) -> dict[str, object]:
        """Cheaply bind a block to the identical preverified object."""

        try:
            authority = self._authority
            bound = self._bindings
            raw = self._canonical_receipt
            token = self._token_sha256
        except AttributeError as exc:
            raise QuacP1RuntimeError(
                "verified runtime binding token is malformed"
            ) from exc
        if (
            authority is not _VERIFIED_BINDINGS_AUTHORITY
            or bindings is not bound
            or not isinstance(raw, bytes)
            or hashlib.sha256(raw).hexdigest() != token
        ):
            raise QuacP1RuntimeError(
                "verified runtime binding token mismatched"
            )
        try:
            parsed = json.loads(raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QuacP1RuntimeError(
                "verified runtime receipt is malformed"
            ) from exc
        if (
            not isinstance(parsed, dict)
            or raw != canonical_bytes(parsed)
            or parsed.get("schema") != VERIFIED_BINDINGS_SCHEMA
            or parsed.get("full_tree_verification_count") != 1
            or parsed.get("binding_sha256")
            != stable_hash(bindings.semantic_payload())
            or parsed.get("self_sha256")
            != stable_hash(
                {
                    key: value
                    for key, value in parsed.items()
                    if key != "self_sha256"
                }
            )
        ):
            raise QuacP1RuntimeError(
                "verified runtime receipt binding drifted"
            )
        return parsed


def verify_runtime_bindings_once(
    bindings: RuntimeBindings,
    *,
    source_access_count: int,
) -> VerifiedRuntimeBindings:
    """Run the sole full-tree pre-source verification and issue its token."""

    if not isinstance(bindings, RuntimeBindings):
        raise QuacP1RuntimeError("RuntimeBindings is required")
    if type(source_access_count) is not int or source_access_count != 0:
        raise QuacP1RuntimeError(
            "runtime verification must precede every source access"
        )
    observed = bindings.verify()
    receipt = _self_hashed(
        {
            "binding_sha256": observed["binding_sha256"],
            "full_tree_verification_count": 1,
            "runtime_receipt": observed,
            "schema": VERIFIED_BINDINGS_SCHEMA,
            "source_access_count_at_verification": source_access_count,
        }
    )
    return VerifiedRuntimeBindings._issue(
        bindings=bindings,
        receipt=receipt,
    )


@dataclass(frozen=True)
class RuntimeQuery:
    """One opaque query with current-to-previous question turns only."""

    query_id: str
    question_turns: tuple[action.QuestionTurn, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "query_id",
            _opaque_id(self.query_id, "query_id"),
        )
        turns = tuple(self.question_turns)
        if (
            not 1 <= len(turns) <= action.MAX_DIALOGUE_TURNS
            or any(not isinstance(row, action.QuestionTurn) for row in turns)
        ):
            raise QuacP1RuntimeError(
                "runtime query turns drifted"
            )
        object.__setattr__(self, "question_turns", turns)


@dataclass(frozen=True)
class RuntimeBlock:
    """Complete anonymized block corpus and query batch."""

    block_id: str
    documents: tuple[action.BlockDocument, ...]
    queries: tuple[RuntimeQuery, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "block_id",
            _opaque_id(self.block_id, "block_id"),
        )
        documents = tuple(self.documents)
        queries = tuple(self.queries)
        if (
            len(documents) < action.evaluator.TOP_K
            or any(
                not isinstance(row, action.BlockDocument)
                for row in documents
            )
        ):
            raise QuacP1RuntimeError(
                "runtime block documents drifted"
            )
        if (
            not 1 <= len(queries) <= MAX_BLOCK_QUERIES
            or any(not isinstance(row, RuntimeQuery) for row in queries)
        ):
            raise QuacP1RuntimeError("runtime block queries drifted")
        document_ids = tuple(row.unit_id for row in documents)
        query_ids = tuple(row.query_id for row in queries)
        for row in documents:
            _opaque_id(row.unit_id, "unit_id")
            _opaque_id(row.context_id, "context_id")
        if (
            document_ids != tuple(sorted(document_ids))
            or len(set(document_ids)) != len(document_ids)
            or query_ids != tuple(sorted(query_ids))
            or len(set(query_ids)) != len(query_ids)
        ):
            raise QuacP1RuntimeError(
                "runtime opaque IDs are not canonical"
            )
        object.__setattr__(self, "documents", documents)
        object.__setattr__(self, "queries", queries)


def block_payload(block: RuntimeBlock) -> dict[str, object]:
    if not isinstance(block, RuntimeBlock):
        raise QuacP1RuntimeError("RuntimeBlock is required")
    value = {
        "block_id": block.block_id,
        "documents": [
            {
                "context_id": row.context_id,
                "context_window_ordinal": row.context_window_ordinal,
                "section_title": row.section_title,
                "text": row.text,
                "title": row.title,
                "unit_id": row.unit_id,
            }
            for row in block.documents
        ],
        "queries": [
            {
                "query_id": row.query_id,
                "question_turns": [
                    {"question_text": turn.question_text}
                    for turn in row.question_turns
                ],
            }
            for row in block.queries
        ],
        "schema": BLOCK_SCHEMA,
    }
    validate_block_payload(value)
    return value


def validate_block_payload(value: object) -> RuntimeBlock:
    """Parse a strict label-free private block envelope."""

    if (
        not isinstance(value, Mapping)
        or set(value) != _BLOCK_KEYS
        or value.get("schema") != BLOCK_SCHEMA
    ):
        raise QuacP1RuntimeError("private block envelope drifted")
    raw_documents = value.get("documents")
    raw_queries = value.get("queries")
    if not isinstance(raw_documents, list) or not isinstance(
        raw_queries, list
    ):
        raise QuacP1RuntimeError("private block arrays drifted")
    documents: list[action.BlockDocument] = []
    for ordinal, row in enumerate(raw_documents):
        if not isinstance(row, Mapping) or set(row) != _DOCUMENT_KEYS:
            raise QuacP1RuntimeError(
                f"private document {ordinal} shape drifted"
            )
        documents.append(
            action.BlockDocument(
                unit_id=row["unit_id"],  # type: ignore[arg-type]
                context_id=row["context_id"],  # type: ignore[arg-type]
                title=row["title"],  # type: ignore[arg-type]
                section_title=row["section_title"],  # type: ignore[arg-type]
                context_window_ordinal=row[  # type: ignore[arg-type]
                    "context_window_ordinal"
                ],
                text=row["text"],  # type: ignore[arg-type]
            )
        )
    queries: list[RuntimeQuery] = []
    for ordinal, row in enumerate(raw_queries):
        if not isinstance(row, Mapping) or set(row) != _QUERY_KEYS:
            raise QuacP1RuntimeError(
                f"private query {ordinal} shape drifted"
            )
        raw_turns = row.get("question_turns")
        if not isinstance(raw_turns, list):
            raise QuacP1RuntimeError(
                f"private query {ordinal} turns drifted"
            )
        turns = []
        for turn in raw_turns:
            if not isinstance(turn, Mapping) or set(turn) != _TURN_KEYS:
                raise QuacP1RuntimeError(
                    f"private query {ordinal} turn shape drifted"
                )
            turns.append(
                action.QuestionTurn(
                    question_text=turn["question_text"],  # type: ignore[arg-type]
                )
            )
        queries.append(
            RuntimeQuery(
                query_id=row["query_id"],  # type: ignore[arg-type]
                question_turns=tuple(turns),
            )
        )
    return RuntimeBlock(
        block_id=value["block_id"],  # type: ignore[arg-type]
        documents=tuple(documents),
        queries=tuple(queries),
    )


class MiniLMEncoderProtocol(Protocol):
    """Injectable one-call normalized float32 encoder."""

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        device: str,
        normalize_embeddings: bool,
        dtype: str,
    ) -> Sequence[Sequence[float]]: ...


class LocalMiniLMGpu0Encoder:
    """Lazy local all-MiniLM encoder pinned by :class:`RuntimeBindings`."""

    def __init__(self, model_root: Path) -> None:
        self.model_root = model_root
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            required_environment = {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "CUDA_VISIBLE_DEVICES": GPU0,
                "HF_HUB_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
                **{key: "1" for key in _NATIVE_THREAD_KEYS},
            }
            if any(
                os.environ.get(key) != expected
                for key, expected in required_environment.items()
            ):
                raise QuacP1RuntimeError(
                    "local MiniLM GPU0 offline environment drifted"
                )
            try:
                from sentence_transformers import SentenceTransformer
                import torch

                if (
                    not torch.cuda.is_available()
                    or torch.cuda.device_count() != 1
                ):
                    raise QuacP1RuntimeError(
                        "local MiniLM must see exactly GPU0"
                    )

                self._model = SentenceTransformer(
                    str(self.model_root),
                    device=MINILM_DEVICE,
                    local_files_only=True,
                    trust_remote_code=False,
                )
            except BaseException as exc:
                raise QuacP1RuntimeError(
                    "local MiniLM GPU0 load failed"
                ) from exc
        return self._model

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        device: str,
        normalize_embeddings: bool,
        dtype: str,
    ) -> Sequence[Sequence[float]]:
        if (
            batch_size != MINILM_BATCH_SIZE
            or device != MINILM_DEVICE
            or normalize_embeddings is not True
            or dtype != "float32"
        ):
            raise QuacP1RuntimeError(
                "local MiniLM execution contract drifted"
            )
        try:
            matrix = self._load().encode(  # type: ignore[union-attr]
                list(texts),
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                precision="float32",
                device=device,
            )
        except BaseException as exc:
            raise QuacP1RuntimeError(
                "local MiniLM GPU0 encode failed"
            ) from exc
        if str(getattr(matrix, "dtype", None)) != "float32":
            raise QuacP1RuntimeError(
                "local MiniLM output dtype drifted"
            )
        try:
            return tuple(
                tuple(float(value) for value in row)
                for row in matrix
            )
        except (TypeError, ValueError) as exc:
            raise QuacP1RuntimeError(
                "local MiniLM output drifted"
            ) from exc


@dataclass(frozen=True)
class OfficialLaunchRequest:
    """One exact GPU1 process request for an injected outer launcher."""

    private_input: Mapping[str, object]
    input_path: Path
    output_path: Path
    index_root: Path
    attempt_path: Path
    environment: Mapping[str, str]
    runtime_bindings: RuntimeBindings
    verified_bindings: VerifiedRuntimeBindings
    python_binding: PythonRuntimeBinding
    gpu1_overlay_import_tree_path: Path
    gpu1_base_import_tree_path: Path
    minilm_asset_path: Path
    llm_asset_path: Path
    hipporag_source_path: Path
    minilm_alias: str
    llm_alias: str


class OfficialLaneProtocol(Protocol):
    """Outer launcher that invokes one real official block worker process."""

    def __call__(
        self,
        request: OfficialLaunchRequest,
    ) -> Mapping[str, object]: ...


def _direct_directory(path: Path, field: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise QuacP1RuntimeError(f"{field} directory is unavailable") from exc
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise QuacP1RuntimeError(f"{field} directory binding drifted")
    return path


def _read_official_private_output(
    *,
    path: Path,
    expected_input: Mapping[str, object],
) -> dict[str, object]:
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
        parsed = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QuacP1RuntimeError(
            "official worker output is unavailable"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or not isinstance(parsed, dict)
        or raw != official_contract.canonical_bytes(parsed)
    ):
        raise QuacP1RuntimeError(
            "official worker output metadata drifted"
        )
    official_contract.validate_output(
        parsed,
        expected_input=expected_input,
    )
    return parsed


def _open_private_log(path: Path) -> Any:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "wb")
    except OSError as exc:
        raise QuacP1RuntimeError(
            "official private log cannot be created once"
        ) from exc


class LocalOfficialGpu1Lane:
    """Strict one-process production launcher for the frozen GPU1 worker.

    It constructs short local model aliases, supplies a closed offline
    environment and explicit import closure, invokes the official worker once
    with a fixed timeout, and returns only a fully validated private output.
    The caller owns index commitment and cleanup.
    """

    def __init__(
        self,
        *,
        process_runner: Any = subprocess.run,
        timeout_seconds: int = OFFICIAL_WORKER_TIMEOUT_SECONDS,
    ) -> None:
        if timeout_seconds != OFFICIAL_WORKER_TIMEOUT_SECONDS:
            raise QuacP1RuntimeError(
                "official worker timeout cannot be changed"
            )
        if not callable(process_runner):
            raise QuacP1RuntimeError(
                "official process runner must be callable"
            )
        self._process_runner = process_runner
        self._timeout_seconds = timeout_seconds

    def __call__(
        self,
        request: OfficialLaunchRequest,
    ) -> Mapping[str, object]:
        if not isinstance(request, OfficialLaunchRequest):
            raise QuacP1RuntimeError(
                "official launch request type drifted"
            )
        request.verified_bindings.require(request.runtime_bindings)
        if (
            request.python_binding
            is not request.runtime_bindings.gpu1_python
            or request.minilm_asset_path
            != Path(request.runtime_bindings.minilm_asset.path)
            or request.llm_asset_path
            != Path(request.runtime_bindings.llm_asset.path)
            or request.hipporag_source_path
            != Path(request.runtime_bindings.hipporag_source.path)
            or request.gpu1_overlay_import_tree_path
            != Path(
                request.runtime_bindings.gpu1_overlay_import_tree.path
            )
            or request.gpu1_base_import_tree_path
            != Path(
                request.runtime_bindings.gpu1_base_import_tree.path
            )
            or request.minilm_alias
            != request.runtime_bindings.minilm_alias
            or request.llm_alias
            != request.runtime_bindings.llm_alias
        ):
            raise QuacP1RuntimeError(
                "official request escaped the preverified bindings"
            )
        minilm_root = _direct_directory(
            request.minilm_asset_path,
            "MiniLM asset",
        )
        llm_root = _direct_directory(
            request.llm_asset_path,
            "LLM asset",
        )
        hipporag_root = _direct_directory(
            request.hipporag_source_path,
            "HippoRAG source",
        )
        overlay_root = _direct_directory(
            request.gpu1_overlay_import_tree_path,
            "GPU1 overlay import tree",
        )
        base_root = _direct_directory(
            request.gpu1_base_import_tree_path,
            "GPU1 base import tree",
        )
        if (
            minilm_root == llm_root
            or request.input_path.parent != request.index_root.parent
            or request.output_path.parent != request.index_root.parent
            or request.input_path == request.output_path
            or request.index_root.exists()
            or request.index_root.is_symlink()
            or request.output_path.exists()
            or request.output_path.is_symlink()
        ):
            raise QuacP1RuntimeError(
                "official worker path binding drifted"
            )
        scratch = _direct_directory(
            request.index_root.parent,
            "official scratch",
        )
        expected_environment = _child_environment(
            runtime=request.python_binding,
            scratch=scratch,
            physical_gpu=GPU1,
            hipporag_source=hipporag_root,
            overlay_import_tree=overlay_root,
            base_import_tree=base_root,
        )
        if dict(request.environment) != expected_environment:
            raise QuacP1RuntimeError(
                "official closed environment drifted"
            )
        try:
            input_metadata = request.input_path.lstat()
            attempt_metadata = request.attempt_path.lstat()
            input_raw = request.input_path.read_bytes()
            parsed_input = json.loads(input_raw.decode("ascii"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QuacP1RuntimeError(
                "official input or prior attempt is unavailable"
            ) from exc
        if (
            request.input_path.is_symlink()
            or not stat.S_ISREG(input_metadata.st_mode)
            or stat.S_IMODE(input_metadata.st_mode) != 0o600
            or request.attempt_path.is_symlink()
            or not stat.S_ISREG(attempt_metadata.st_mode)
            or stat.S_IMODE(attempt_metadata.st_mode) != 0o400
            or not isinstance(parsed_input, dict)
            or input_raw != official_contract.canonical_bytes(parsed_input)
            or parsed_input != dict(request.private_input)
        ):
            raise QuacP1RuntimeError(
                "official input or attempt metadata drifted"
            )
        official_contract.validate_input(parsed_input)

        alias_root = scratch / "model_aliases"
        _private_directory(alias_root, fresh=True)
        aliases = {
            _model_alias(request.minilm_alias, "MiniLM"): minilm_root,
            _model_alias(request.llm_alias, "LLM"): llm_root,
        }
        if len(aliases) != 2:
            raise QuacP1RuntimeError(
                "official model aliases collided"
            )
        for alias, target in aliases.items():
            link = alias_root / alias
            try:
                os.symlink(str(target), link, target_is_directory=True)
                metadata = link.lstat()
                resolved = link.resolve(strict=True)
            except OSError as exc:
                raise QuacP1RuntimeError(
                    "official model alias cannot be created"
                ) from exc
            if (
                not stat.S_ISLNK(metadata.st_mode)
                or os.readlink(link) != str(target)
                or not os.path.samefile(resolved, target)
            ):
                raise QuacP1RuntimeError(
                    "official model alias binding drifted"
                )

        command = [
            request.python_binding.executable.path,
            "-S",
            "-B",
            "-m",
            "replication_runtime.quac_p1_official_v1.worker",
            "--input",
            str(request.input_path),
            "--output",
            str(request.output_path),
            "--index-root",
            str(request.index_root),
            "--llm-model",
            request.llm_alias,
            "--embedding-model",
            request.minilm_alias,
        ]
        if str(minilm_root) in command or str(llm_root) in command:
            raise QuacP1RuntimeError(
                "absolute model path escaped into official argv"
            )
        private_root = _direct_directory(
            request.attempt_path.parent,
            "official private custody",
        )
        stdout_path = private_root / "official.stdout.private.bin"
        stderr_path = private_root / "official.stderr.private.bin"
        try:
            with _open_private_log(stdout_path) as stdout_handle, (
                _open_private_log(stderr_path)
            ) as stderr_handle:
                completed = self._process_runner(
                    command,
                    check=False,
                    cwd=alias_root,
                    env=expected_environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    timeout=self._timeout_seconds,
                )
                stdout_handle.flush()
                stderr_handle.flush()
                os.fsync(stdout_handle.fileno())
                os.fsync(stderr_handle.fileno())
        except subprocess.TimeoutExpired as exc:
            raise QuacP1RuntimeError(
                "official worker timed out; no retry is permitted"
            ) from exc
        except OSError as exc:
            raise QuacP1RuntimeError(
                "official worker launch failed; no retry is permitted"
            ) from exc
        finally:
            for path in (stdout_path, stderr_path):
                if path.exists() and not path.is_symlink():
                    try:
                        os.chmod(path, 0o400)
                    except OSError:
                        pass
        returncode = getattr(completed, "returncode", None)
        if type(returncode) is not int or returncode != 0:
            raise QuacP1RuntimeError(
                "official worker exited unsuccessfully; no retry is permitted"
            )
        return _read_official_private_output(
            path=request.output_path,
            expected_input=parsed_input,
        )


class ActionAdapterProtocol(Protocol):
    """Narrow compatibility seam for the frozen action adapter."""

    def required_serializations(
        self,
        documents: Sequence[action.BlockDocument],
        turns: Sequence[action.QuestionTurn],
    ) -> Sequence[tuple[str, str]]: ...

    def build(
        self,
        *,
        documents: Sequence[action.BlockDocument],
        turns: Sequence[action.QuestionTurn],
        embeddings: Sequence[action.MiniLmEmbedding],
    ) -> action.ActionAdapterResult: ...

    def payload(
        self,
        result: action.ActionAdapterResult,
    ) -> Mapping[str, object]: ...


class FrozenActionAdapter:
    """Thin seam over :mod:`quac_p1_action_adapter_v1`."""

    @staticmethod
    def required_serializations(
        documents: Sequence[action.BlockDocument],
        turns: Sequence[action.QuestionTurn],
    ) -> Sequence[tuple[str, str]]:
        return action.required_embedding_serializations(documents, turns)

    @staticmethod
    def build(
        *,
        documents: Sequence[action.BlockDocument],
        turns: Sequence[action.QuestionTurn],
        embeddings: Sequence[action.MiniLmEmbedding],
    ) -> action.ActionAdapterResult:
        return action.build_action_graph(
            action.ActionAdapterInput(
                documents=tuple(documents),
                question_turns=tuple(turns),
                minilm_embeddings=tuple(embeddings),
            )
        )

    @staticmethod
    def payload(
        result: action.ActionAdapterResult,
    ) -> Mapping[str, object]:
        return action.canonical_action_payload(result)


@dataclass(frozen=True)
class MiniLaneResult:
    actions: Mapping[str, action.ActionAdapterResult]
    private_payload: Mapping[str, object]
    embedding_set_sha256: str
    unique_embedding_count: int


@dataclass(frozen=True)
class BlockRuntimeResult:
    actions: Mapping[str, action.ActionAdapterResult]
    official_top5: Mapping[str, tuple[str, ...]] | None
    safe_receipt: Mapping[str, object]


def _embedding_rows(
    *,
    encoder: MiniLMEncoderProtocol,
    requests: Sequence[tuple[str, str]],
) -> tuple[action.MiniLmEmbedding, ...]:
    texts = tuple(text for _digest, text in requests)
    if len(set(texts)) != len(texts):
        raise QuacP1RuntimeError(
            "bulk MiniLM request contains duplicate exact text"
        )
    matrix = encoder.encode(
        texts,
        batch_size=MINILM_BATCH_SIZE,
        device=MINILM_DEVICE,
        normalize_embeddings=True,
        dtype="float32",
    )
    rows = tuple(matrix)
    if len(rows) != len(requests):
        raise QuacP1RuntimeError("bulk MiniLM row count drifted")
    result = tuple(
        action.MiniLmEmbedding(
            serialization_sha256=digest,
            vector=tuple(vector),
        )
        for (digest, _text), vector in zip(requests, rows, strict=True)
    )
    return result


def _embedding_commitment(
    embeddings: Sequence[action.MiniLmEmbedding],
) -> str:
    digest = hashlib.sha256()
    for row in embeddings:
        encoded = row.serialization_sha256.encode("ascii")
        digest.update(encoded)
        for value in row.vector:
            numeric = float(value)
            if not math.isfinite(numeric):
                raise QuacP1RuntimeError(
                    "MiniLM embedding commitment became nonfinite"
                )
            digest.update(struct.pack("<f", numeric))
    return digest.hexdigest()


def _run_minilm_lane(
    *,
    block: RuntimeBlock,
    encoder: MiniLMEncoderProtocol,
    adapter: ActionAdapterProtocol,
) -> MiniLaneResult:
    ordered_requests: list[tuple[str, str]] = []
    observed: dict[str, str] = {}
    query_request_hashes: dict[str, tuple[str, ...]] = {}
    for query in block.queries:
        requests = tuple(
            adapter.required_serializations(
                block.documents,
                query.question_turns,
            )
        )
        expected_requests = action.required_embedding_serializations(
            block.documents,
            query.question_turns,
        )
        if requests != expected_requests:
            raise QuacP1RuntimeError(
                "action adapter omitted or changed a frozen serialization"
            )
        hashes = []
        for digest, text in requests:
            if (
                not isinstance(digest, str)
                or _HEX64.fullmatch(digest) is None
                or not isinstance(text, str)
            ):
                raise QuacP1RuntimeError(
                    "action adapter embedding request drifted"
                )
            prior = observed.get(digest)
            if prior is not None and prior != text:
                raise QuacP1RuntimeError(
                    "action adapter embedding hash collision"
                )
            if prior is None:
                observed[digest] = text
                ordered_requests.append((digest, text))
            hashes.append(digest)
        query_request_hashes[query.query_id] = tuple(hashes)
    embeddings = _embedding_rows(
        encoder=encoder,
        requests=tuple(ordered_requests),
    )
    by_hash = {
        row.serialization_sha256: row
        for row in embeddings
    }
    if len(by_hash) != len(embeddings):
        raise QuacP1RuntimeError(
            "bulk MiniLM returned duplicate serialization rows"
        )
    actions: dict[str, action.ActionAdapterResult] = {}
    private_rows = []
    for query in block.queries:
        required = query_request_hashes[query.query_id]
        try:
            query_embeddings = tuple(by_hash[digest] for digest in required)
        except KeyError as exc:
            raise QuacP1RuntimeError(
                "bulk MiniLM result omitted a requested serialization"
            ) from exc
        result = adapter.build(
            documents=block.documents,
            turns=query.question_turns,
            embeddings=query_embeddings,
        )
        if not isinstance(result, action.ActionAdapterResult):
            raise QuacP1RuntimeError(
                "action adapter result type drifted"
            )
        actions[query.query_id] = result
        payload = dict(adapter.payload(result))
        private_rows.append(
            {
                "action": payload,
                "action_sha256": stable_hash(payload),
                "query_id": query.query_id,
            }
        )
    expected_ids = {row.query_id for row in block.queries}
    if set(actions) != expected_ids:
        raise QuacP1RuntimeError(
            "action adapter query ID set drifted"
        )
    private_payload = {
        "block_id": block.block_id,
        "rows": private_rows,
        "schema": ACTION_PACK_SCHEMA,
    }
    return MiniLaneResult(
        actions=actions,
        private_payload=private_payload,
        embedding_set_sha256=_embedding_commitment(embeddings),
        unique_embedding_count=len(embeddings),
    )


def _official_input(block: RuntimeBlock) -> dict[str, object]:
    return official_contract.build_input(
        block_id=block.block_id,
        units=[
            {
                # The official contract adds the unique canonical-JSON
                # envelope.  Supplying its inner text here keeps that
                # document byte-identical to the MiniLM serialization.
                "text": action.official_inner_unit_text(document),
                "unit_id": document.unit_id,
            }
            for document in block.documents
        ],
        queries=[
            {
                "query_id": query.query_id,
                "text": action.serialize_full_query(query.question_turns),
            }
            for query in block.queries
        ],
    )


def _child_environment(
    *,
    runtime: PythonRuntimeBinding,
    scratch: Path,
    physical_gpu: str,
    hipporag_source: Path,
    overlay_import_tree: Path,
    base_import_tree: Path,
) -> dict[str, str]:
    if physical_gpu not in (GPU0, GPU1):
        raise QuacP1RuntimeError("physical GPU binding drifted")
    _direct_directory(hipporag_source, "HippoRAG source")
    _direct_directory(
        overlay_import_tree,
        "GPU1 overlay import tree",
    )
    _direct_directory(
        base_import_tree,
        "GPU1 base import tree",
    )
    environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_MODULE_LOADING": "LAZY",
        "CUDA_VISIBLE_DEVICES": physical_gpu,
        "HF_HOME": str(scratch / "cache"),
        "HF_HUB_OFFLINE": "1",
        "HOME": str(scratch / "home"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": f"{Path(runtime.executable.path).parent}:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join(
            (
                str(_PROJECT_IMPORT_ROOT),
                runtime.import_tree.path,
                str(hipporag_source),
                str(overlay_import_tree),
                str(base_import_tree),
            )
        ),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "TMP": str(scratch / "tmp"),
        "TEMP": str(scratch / "tmp"),
        "TMPDIR": str(scratch / "tmp"),
    }
    for key in _NATIVE_THREAD_KEYS:
        environment[key] = "1"
    return environment


def _load_or_write_official_output(
    *,
    output_path: Path,
    returned: Mapping[str, object],
) -> dict[str, object]:
    returned_dict = dict(returned)
    if output_path.exists() or output_path.is_symlink():
        try:
            info = output_path.lstat()
            raw = output_path.read_bytes()
            parsed = json.loads(raw.decode("ascii"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QuacP1RuntimeError(
                "official private output is unavailable"
            ) from exc
        if (
            output_path.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or not isinstance(parsed, dict)
            or raw != official_contract.canonical_bytes(parsed)
            or parsed != returned_dict
        ):
            raise QuacP1RuntimeError(
                "official private output metadata drifted"
            )
        return parsed
    _write_once(
        output_path,
        official_contract.canonical_bytes(returned_dict),
        final_mode=0o600,
    )
    return returned_dict


def _safe_failure(
    *,
    work_root: Path,
    stage: str,
    exc: BaseException,
    attempt_sha256: str | None,
) -> None:
    path = work_root / "runtime.safe.json"
    if path.exists() or path.is_symlink():
        return
    receipt = _self_hashed(
        {
            "API_or_online_evaluation_call_count": 0,
            "attempt_file_sha256": attempt_sha256,
            "exception_message_sha256": hashlib.sha256(
                str(exc).encode("utf-8")
            ).hexdigest(),
            "exception_type_sha256": hashlib.sha256(
                type(exc).__qualname__.encode("utf-8")
            ).hexdigest(),
            "retry_replay_resample_or_fallback_authorized": False,
            "schema": SAFE_FAILURE_SCHEMA,
            "stage": stage,
            "status": "implementation_or_infrastructure_invalid",
        }
    )
    try:
        _write_json_once(path, receipt, final_mode=0o400)
    except QuacP1RuntimeError:
        return


def run_block(
    *,
    block_role: str,
    block: RuntimeBlock,
    work_root: Path,
    bindings: RuntimeBindings,
    verified_bindings: VerifiedRuntimeBindings,
    encoder: MiniLMEncoderProtocol,
    official_lane: OfficialLaneProtocol | None,
    action_adapter: ActionAdapterProtocol | None = None,
) -> BlockRuntimeResult:
    """Consume one lifecycle-bound block attempt.

    ``A_form`` builds only complete label-free Agent graphs before its late
    labels are opened.  ``A_hold`` requires the concurrent official baseline.
    ``M_search`` has the identical two-lane behavior, but may only be passed
    here after the trusted acquisition broker consumes its promotion
    capability; this runtime has no source or M materialization surface.
    """

    if not isinstance(block, RuntimeBlock):
        raise QuacP1RuntimeError("RuntimeBlock is required")
    if block_role not in BLOCK_ROLES:
        raise QuacP1RuntimeError("runtime block role drifted")
    official_required = block_role in OFFICIAL_BLOCK_ROLES
    if official_required != (official_lane is not None):
        raise QuacP1RuntimeError(
            "official lane does not match the frozen block lifecycle"
        )
    if work_root.exists() or work_root.is_symlink():
        raise QuacP1RuntimeError(
            "block runtime work root is not fresh; retry is forbidden"
        )
    if not isinstance(verified_bindings, VerifiedRuntimeBindings):
        raise QuacP1RuntimeError(
            "VerifiedRuntimeBindings is required"
        )
    verified_receipt = verified_bindings.require(bindings)
    runtime_receipt = verified_receipt.get("runtime_receipt")
    if not isinstance(runtime_receipt, Mapping):
        raise QuacP1RuntimeError(
            "verified runtime receipt disappeared"
        )
    binding_receipt = dict(runtime_receipt)
    if isinstance(encoder, LocalMiniLMGpu0Encoder):
        try:
            encoder_root = encoder.model_root.resolve(strict=True)
            frozen_root = Path(bindings.minilm_asset.path).resolve(
                strict=True
            )
            active_python = Path(sys.executable).resolve(strict=True)
        except OSError as exc:
            raise QuacP1RuntimeError(
                "local MiniLM model binding is unavailable"
            ) from exc
        if (
            not os.path.samefile(encoder_root, frozen_root)
            or str(active_python)
            != bindings.gpu0_python.executable.realpath
            or bindings.gpu0_python.import_tree.path not in sys.path
        ):
            raise QuacP1RuntimeError(
                "local MiniLM escaped its frozen asset or Python runtime"
            )
    adapter = action_adapter or FrozenActionAdapter()
    _private_directory(work_root, fresh=True)
    _private_directory(work_root / "private", fresh=True)
    _private_directory(work_root / "scratch", fresh=True)
    for lane in ("gpu0", "gpu1"):
        lane_root = work_root / "scratch" / lane
        _private_directory(lane_root, fresh=True)
        for child in ("cache", "home", "tmp"):
            _private_directory(lane_root / child, fresh=True)

    attempt_sha: str | None = None
    stage = "claim_attempt_before_worker"
    attempt_path = work_root / "private" / "attempt.private.json"
    attempt = _self_hashed(
        {
            "API_or_online_evaluation_authorized": False,
            "asset_binding_sha256": binding_receipt["binding_sha256"],
            "binding_verification_token_sha256": (
                verified_bindings.token_sha256
            ),
            "block_role": block_role,
            "block_input_sha256": stable_hash(block_payload(block)),
            "official_lane_authorized": official_required,
            "retry_replay_resample_or_fallback_authorized": False,
            "schema": ATTEMPT_SCHEMA,
        }
    )
    attempt_sha = _write_json_once(
        attempt_path,
        attempt,
        final_mode=0o400,
    )
    try:
        stage = "seal_label_free_block_input"
        block_input_sha = _write_json_once(
            work_root / "private" / "block.private.json",
            block_payload(block),
            final_mode=0o400,
        )
        official_input: dict[str, object] | None = None
        official_request: OfficialLaunchRequest | None = None
        if official_required:
            official_input = _official_input(block)
            official_input_path = (
                work_root / "scratch" / "gpu1" / "input.private.json"
            )
            _write_once(
                official_input_path,
                official_contract.canonical_bytes(official_input),
                final_mode=0o600,
            )
            official_request = OfficialLaunchRequest(
                private_input=official_input,
                input_path=official_input_path,
                output_path=(
                    work_root
                    / "scratch"
                    / "gpu1"
                    / "output.private.json"
                ),
                index_root=(
                    work_root / "scratch" / "gpu1" / "official_index"
                ),
                attempt_path=attempt_path,
                environment=_child_environment(
                    runtime=bindings.gpu1_python,
                    scratch=work_root / "scratch" / "gpu1",
                    physical_gpu=GPU1,
                    hipporag_source=Path(
                        bindings.hipporag_source.path
                    ),
                    overlay_import_tree=Path(
                        bindings.gpu1_overlay_import_tree.path
                    ),
                    base_import_tree=Path(
                        bindings.gpu1_base_import_tree.path
                    ),
                ),
                runtime_bindings=bindings,
                verified_bindings=verified_bindings,
                python_binding=bindings.gpu1_python,
                gpu1_overlay_import_tree_path=Path(
                    bindings.gpu1_overlay_import_tree.path
                ),
                gpu1_base_import_tree_path=Path(
                    bindings.gpu1_base_import_tree.path
                ),
                minilm_asset_path=Path(bindings.minilm_asset.path),
                llm_asset_path=Path(bindings.llm_asset.path),
                hipporag_source_path=Path(
                    bindings.hipporag_source.path
                ),
                minilm_alias=bindings.minilm_alias,
                llm_alias=bindings.llm_alias,
            )

        mini_result: MiniLaneResult
        returned_official: Mapping[str, object] | None = None
        parallel_barrier_passed = False
        stage = "submit_label_free_model_lanes"
        if official_required:
            assert official_lane is not None
            assert official_request is not None
            launch_gate = threading.Event()

            def mini_task() -> MiniLaneResult:
                launch_gate.wait()
                return _run_minilm_lane(
                    block=block,
                    encoder=encoder,
                    adapter=adapter,
                )

            def official_task() -> Mapping[str, object]:
                launch_gate.wait()
                return official_lane(official_request)

            with ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="quac-p1-two-lane",
            ) as pool:
                mini_future = pool.submit(mini_task)
                official_future = pool.submit(official_task)
                launch_gate.set()
                parallel_barrier_passed = True
                try:
                    mini_result = mini_future.result()
                    returned_official = official_future.result()
                except BaseException:
                    mini_future.cancel()
                    official_future.cancel()
                    raise
        else:
            mini_result = _run_minilm_lane(
                block=block,
                encoder=encoder,
                adapter=adapter,
            )

        stage = "seal_minilm_and_actions"
        action_pack_sha = _write_json_once(
            work_root / "private" / "actions.private.json",
            mini_result.private_payload,
            final_mode=0o400,
        )
        minilm_receipt = _self_hashed(
            {
                "action_count": len(mini_result.actions),
                "action_pack_file_sha256": action_pack_sha,
                "batch_size": MINILM_BATCH_SIZE,
                "device": MINILM_DEVICE,
                "dtype": "float32",
                "embedding_set_sha256": (
                    mini_result.embedding_set_sha256
                ),
                "encode_call_count": 1,
                "normalize_embeddings": True,
                "schema": MINILM_RECEIPT_SCHEMA,
                "unique_embedding_count": (
                    mini_result.unique_embedding_count
                ),
            }
        )
        minilm_receipt_sha = _write_json_once(
            work_root / "private" / "minilm.private.json",
            minilm_receipt,
            final_mode=0o400,
        )

        official_top5: dict[str, tuple[str, ...]] | None = None
        official_output_sha: str | None = None
        official_full_rankings_sha: str | None = None
        index_receipt: dict[str, object] = {
            "cleanup_verified": True,
            "file_count": 0,
            "total_bytes": 0,
            "tree_sha256": None,
        }
        if official_required:
            stage = "validate_official_and_cleanup_index"
            assert official_input is not None
            assert official_request is not None
            assert returned_official is not None
            official_output = _load_or_write_official_output(
                output_path=official_request.output_path,
                returned=returned_official,
            )
            validated = official_contract.validate_output(
                official_output,
                expected_input=official_input,
            )
            runtime_receipt = validated.get("runtime")
            if (
                not isinstance(runtime_receipt, Mapping)
                or runtime_receipt.get("index_call_count") != 1
                or runtime_receipt.get("retrieve_call_count") != 1
            ):
                raise QuacP1RuntimeError(
                    "official one-index/one-retrieve receipt drifted"
                )
            expected_query_ids = {
                query.query_id for query in block.queries
            }
            rows = validated.get("rows")
            if not isinstance(rows, list):
                raise QuacP1RuntimeError(
                    "official result rows disappeared"
                )
            official_top5 = {}
            for row in rows:
                if not isinstance(row, Mapping):
                    raise QuacP1RuntimeError(
                        "official result row drifted"
                    )
                query_id = row.get("query_id")
                top5 = row.get("top5_unit_ids")
                if (
                    not isinstance(query_id, str)
                    or query_id in official_top5
                    or not isinstance(top5, list)
                ):
                    raise QuacP1RuntimeError(
                        "official query ID mapping drifted"
                    )
                official_top5[query_id] = tuple(top5)
            if set(official_top5) != expected_query_ids:
                raise QuacP1RuntimeError(
                    "official query ID set has missing or extra rows"
                )
            if (
                not official_request.index_root.is_dir()
                or official_request.index_root.is_symlink()
            ):
                raise QuacP1RuntimeError(
                    "official ephemeral index is unavailable"
                )
            tree_sha, file_count, total_bytes = _seal_private_tree_once(
                official_request.index_root,
                work_root / "private" / "official_index.private",
            )
            index_receipt = {
                "cleanup_verified": True,
                "file_count": file_count,
                "total_bytes": total_bytes,
                "tree_sha256": tree_sha,
            }
            official_output_sha = _write_json_once(
                work_root
                / "private"
                / "official_output.private.json",
                validated,
                final_mode=0o400,
            )
            official_full_rankings_sha = str(
                validated["full_rankings_sha256"]
            )

        stage = "write_safe_runtime_terminal"
        safe_receipt = _self_hashed(
            {
                "API_or_online_evaluation_call_count": 0,
                "action_count": len(mini_result.actions),
                "action_pack_file_sha256": action_pack_sha,
                "asset_binding_sha256": binding_receipt["binding_sha256"],
                "binding_verification_token_sha256": (
                    verified_bindings.token_sha256
                ),
                "attempt_count": 1,
                "attempt_file_sha256": attempt_sha,
                "block_input_file_sha256": block_input_sha,
                "block_role": block_role,
                "corpus_count": len(block.documents),
                "index_cleanup": index_receipt,
                "label_family_qrel_or_answer_input_count": 0,
                "logical_action_query_count": len(block.queries),
                "max_concurrent_physical_model_lanes": (
                    2 if official_required else 1
                ),
                "minilm_encode_call_count": 1,
                "minilm_receipt_file_sha256": minilm_receipt_sha,
                "official_full_rankings_sha256": (
                    official_full_rankings_sha
                ),
                "official_index_call_count": (
                    1 if official_required else 0
                ),
                "official_output_file_sha256": official_output_sha,
                "official_required": official_required,
                "official_retrieve_call_count": (
                    1 if official_required else 0
                ),
                "parallel_submission_barrier_passed": (
                    parallel_barrier_passed
                    if official_required
                    else None
                ),
                "query_count": len(block.queries),
                "retry_replay_resample_or_fallback_count": 0,
                "schema": SAFE_RESULT_SCHEMA,
                "status": "passed_label_free_block_runtime",
                "unique_embedding_count": (
                    mini_result.unique_embedding_count
                ),
            }
        )
        _write_json_once(
            work_root / "runtime.safe.json",
            safe_receipt,
            final_mode=0o400,
        )
        return BlockRuntimeResult(
            actions=mini_result.actions,
            official_top5=official_top5,
            safe_receipt=safe_receipt,
        )
    except BaseException as exc:
        _safe_failure(
            work_root=work_root,
            stage=stage,
            exc=exc,
            attempt_sha256=attempt_sha,
        )
        raise


__all__ = [
    "ACTION_PACK_SCHEMA",
    "ATTEMPT_SCHEMA",
    "BLOCK_ROLES",
    "BLOCK_SCHEMA",
    "GPU0",
    "GPU1",
    "OFFICIAL_BLOCK_ROLES",
    "OFFICIAL_WORKER_TIMEOUT_SECONDS",
    "MINILM_BATCH_SIZE",
    "MINILM_DEVICE",
    "SAFE_FAILURE_SCHEMA",
    "SAFE_RESULT_SCHEMA",
    "VERIFIED_BINDINGS_SCHEMA",
    "ActionAdapterProtocol",
    "BlockRuntimeResult",
    "FrozenActionAdapter",
    "FrozenExecutableBinding",
    "FrozenTreeBinding",
    "LocalOfficialGpu1Lane",
    "LocalMiniLMGpu0Encoder",
    "MiniLMEncoderProtocol",
    "OfficialLaneProtocol",
    "OfficialLaunchRequest",
    "PythonRuntimeBinding",
    "QuacP1RuntimeError",
    "RuntimeBindings",
    "RuntimeBlock",
    "RuntimeQuery",
    "VerifiedRuntimeBindings",
    "block_payload",
    "canonical_bytes",
    "run_block",
    "stable_hash",
    "validate_block_payload",
    "verify_runtime_bindings_once",
]
