"""One source-free production-path integration canary for QuAC P1.

The canary has no dataset loader or formal-source path.  Its only input is a
strict canonical configuration containing a fresh work root plus project,
effect-design, and local runtime bindings.  It constructs five fixed synthetic
documents and one fixed current/previous question pair, verifies the frozen
runtime before any source access, and invokes the production label-free
``A_hold`` runtime once.  That runtime concurrently submits one GPU0 MiniLM
lane and one GPU1 official-HippoRAG lane, builds one ephemeral official index,
and permits no retry.

Only an aggregate safe terminal (or aggregate no-retry failure) is written at
this outer boundary.  Synthetic text, opaque IDs, rankings, embeddings, and
private action payloads remain in the inner private runtime directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping, Sequence

from assumption_agent.benchmarks import quac_p1_action_adapter_v1 as action
from assumption_agent.benchmarks import quac_p1_runtime_v1 as runtime


VERSION = "quac_p1_source_free_canary_v1"
STUDY_ID = "QUAC_P1_RJMC_DIALOGUE_EVIDENCE_L5_V1"
CONFIG_SCHEMA = f"{VERSION}_config_v1"
ATTEMPT_SCHEMA = f"{VERSION}_attempt_v1"
SAFE_TERMINAL_SCHEMA = f"{VERSION}_safe_terminal_v1"
SAFE_FAILURE_SCHEMA = f"{VERSION}_safe_failure_v1"
ATTEMPT_FILENAME = "canary_attempt.private.json"
TERMINAL_FILENAME = "canary.safe.json"
INNER_RUNTIME_DIRECTORY = "block_runtime"
EXPECTED_DESIGN_SELF_SHA256 = (
    "def417300b3c25f127517eef1cdd61760757762f08cc5a9b9877b261036dace2"
)
EXPECTED_HIPPORAG_NORMATIVE_CONTENT_SHA256 = (
    "342505c3aaa8dc5e57718e8ac695ac28f60aa66837ba717f52d6f7b536527b1f"
)
EXPECTED_HIPPORAG_NORMATIVE_FILE_COUNT = 60
EXPECTED_HIPPORAG_NORMATIVE_SIZE_BYTES = 332110
ASSET_FREEZE_SCHEMA = f"{VERSION}_asset_freeze_v1"
ASSET_FREEZE_STATUS = "frozen_prospective_assets_before_canary"
SYNTHETIC_DOCUMENT_COUNT = 5
SYNTHETIC_QUERY_COUNT = 1
SYNTHETIC_UNIQUE_EMBEDDING_COUNT = 8

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DESIGN_PATH = (
    _PROJECT_ROOT / "manifests" / "quac_p1_effect_execution_design_v1.json"
)
_PROJECT_FILES = {
    "assumption_agent_initializer_file_sha256": (
        "assumption_agent/__init__.py"
    ),
    "assumption_agent_models_file_sha256": "assumption_agent/models.py",
    "benchmarks_initializer_file_sha256": (
        "assumption_agent/benchmarks/__init__.py"
    ),
    "action_adapter_file_sha256": (
        "assumption_agent/benchmarks/quac_p1_action_adapter_v1.py"
    ),
    "evaluator_file_sha256": (
        "assumption_agent/benchmarks/quac_rjmc_evaluator_v1.py"
    ),
    "formal_acquisition_file_sha256": (
        "assumption_agent/benchmarks/quac_p1_formal_acquisition_v1.py"
    ),
    "formal_controller_file_sha256": (
        "assumption_agent/benchmarks/quac_p1_formal_controller_v1.py"
    ),
    "maud_official_initializer_file_sha256": (
        "replication_runtime/maud_extraction_p2_official_v1/__init__.py"
    ),
    "maud_official_worker_file_sha256": (
        "replication_runtime/maud_extraction_p2_official_v1/worker.py"
    ),
    "official_contract_file_sha256": (
        "replication_runtime/quac_p1_official_v1/contract.py"
    ),
    "official_worker_file_sha256": (
        "replication_runtime/quac_p1_official_v1/worker.py"
    ),
    "quac_official_initializer_file_sha256": (
        "replication_runtime/quac_p1_official_v1/__init__.py"
    ),
    "replication_runtime_initializer_file_sha256": (
        "replication_runtime/__init__.py"
    ),
    "runtime_file_sha256": (
        "assumption_agent/benchmarks/quac_p1_runtime_v1.py"
    ),
    "source_free_canary_file_sha256": (
        "replication_runtime/quac_p1_source_free_canary_v1.py"
    ),
}
_PROJECT_BINDING_KEYS = frozenset(_PROJECT_FILES)
_DESIGN_BINDING_KEYS = frozenset(
    {
        "effect_execution_design_file_sha256",
        "effect_execution_design_self_sha256",
    }
)
_CONFIG_KEYS = frozenset(
    {
        "asset_freeze_binding",
        "design_binding",
        "project_binding",
        "runtime_bindings",
        "schema",
        "self_sha256",
        "work_root",
    }
)
_ASSET_FREEZE_BINDING_KEYS = frozenset(
    {"file_sha256", "path", "schema", "self_sha256"}
)
_ASSET_FREEZE_KEYS = frozenset(
    {
        "effect_execution_design_self_sha256",
        "normative_hipporag_source_content_receipt",
        "runtime_binding_sha256",
        "runtime_bindings",
        "schema",
        "self_sha256",
        "status",
        "study_id",
    }
)
_TREE_KEYS = frozenset(
    {"file_count", "path", "total_bytes", "tree_sha256"}
)
_EXECUTABLE_KEYS = frozenset(
    {"path", "realpath", "sha256", "size_bytes"}
)
_PYTHON_KEYS = frozenset(
    {"executable", "identity_sha256", "import_tree"}
)
_RUNTIME_BINDING_KEYS = frozenset(
    {
        "gpu0_python",
        "gpu1_base_import_tree",
        "gpu1_overlay_import_tree",
        "gpu1_python",
        "hipporag_source",
        "llm_alias",
        "llm_asset",
        "minilm_alias",
        "minilm_asset",
    }
)
_RUNTIME_SAFE_KEYS = frozenset(
    {
        "API_or_online_evaluation_call_count",
        "action_count",
        "action_pack_file_sha256",
        "asset_binding_sha256",
        "attempt_count",
        "attempt_file_sha256",
        "binding_verification_token_sha256",
        "block_input_file_sha256",
        "block_role",
        "corpus_count",
        "index_cleanup",
        "label_family_qrel_or_answer_input_count",
        "logical_action_query_count",
        "max_concurrent_physical_model_lanes",
        "minilm_encode_call_count",
        "minilm_receipt_file_sha256",
        "official_full_rankings_sha256",
        "official_index_call_count",
        "official_output_file_sha256",
        "official_required",
        "official_retrieve_call_count",
        "parallel_submission_barrier_passed",
        "query_count",
        "retry_replay_resample_or_fallback_count",
        "schema",
        "self_sha256",
        "status",
        "unique_embedding_count",
    }
)
_INDEX_KEYS = frozenset(
    {"cleanup_verified", "file_count", "total_bytes", "tree_sha256"}
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class QuacP1SourceFreeCanaryError(RuntimeError):
    """The source-free canary config, binding, or one-shot run drifted."""


def canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise QuacP1SourceFreeCanaryError(
            "canary value is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise QuacP1SourceFreeCanaryError(
            "bound file cannot be read"
        ) from exc
    return digest.hexdigest()


def normative_hipporag_content_receipt(
    root: Path,
) -> dict[str, object]:
    """Reproduce the design's content-only, bytecode-excluding tree hash."""

    lexical = root.absolute()
    if lexical.is_symlink() or not lexical.is_dir():
        raise QuacP1SourceFreeCanaryError(
            "normative HippoRAG content root drifted"
        )
    rows: list[dict[str, object]] = []
    try:
        for current, directories, files in os.walk(
            lexical,
            followlinks=False,
        ):
            base = Path(current)
            for name in directories:
                if (base / name).is_symlink():
                    raise QuacP1SourceFreeCanaryError(
                        "normative HippoRAG tree contains a symlink"
                    )
            directories[:] = [
                name for name in directories if name != "__pycache__"
            ]
            for name in files:
                path = base / name
                if path.suffix in {".pyc", ".pyo"}:
                    continue
                metadata = path.lstat()
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or not stat.S_ISREG(metadata.st_mode)
                ):
                    raise QuacP1SourceFreeCanaryError(
                        "normative HippoRAG tree contains a non-file"
                    )
                rows.append(
                    {
                        "path": path.relative_to(lexical).as_posix(),
                        "sha256": _sha256_file(path),
                        "size_bytes": metadata.st_size,
                    }
                )
    except OSError as exc:
        raise QuacP1SourceFreeCanaryError(
            "normative HippoRAG tree cannot be inspected"
        ) from exc
    rows.sort(key=lambda row: str(row["path"]))
    return {
        "file_count": len(rows),
        "size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "tree_sha256": stable_hash(rows),
    }


def _hex64(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise QuacP1SourceFreeCanaryError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _exact_dict(
    value: object,
    keys: frozenset[str],
    field: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise QuacP1SourceFreeCanaryError(f"{field} shape drifted")
    return value


def _positive_int(value: object, field: str) -> int:
    if type(value) is not int or value < 1:
        raise QuacP1SourceFreeCanaryError(f"{field} must be positive")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise QuacP1SourceFreeCanaryError(
            f"{field} must be nonnegative"
        )
    return value


def _absolute_path(value: object, field: str) -> Path:
    if not isinstance(value, str):
        raise QuacP1SourceFreeCanaryError(f"{field} path drifted")
    path = Path(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or ".." in path.parts
    ):
        raise QuacP1SourceFreeCanaryError(f"{field} path drifted")
    return path


def _self_hashed(body: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in body:
        raise QuacP1SourceFreeCanaryError(
            "self hash already exists"
        )
    return {**body, "self_sha256": stable_hash(body)}


def _write_once(path: Path, value: Mapping[str, object]) -> str:
    raw = canonical_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o400)
        metadata = path.lstat()
        persisted = path.read_bytes()
    except OSError as exc:
        raise QuacP1SourceFreeCanaryError(
            "canary artifact cannot be created exactly once"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or persisted != raw
    ):
        raise QuacP1SourceFreeCanaryError(
            "canary artifact verification failed"
        )
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ProjectBinding:
    file_sha256: tuple[tuple[str, str], ...]

    @classmethod
    def capture(cls) -> "ProjectBinding":
        return cls(
            tuple(
                (
                    key,
                    _sha256_file(_PROJECT_ROOT / relative),
                )
                for key, relative in sorted(_PROJECT_FILES.items())
            )
        )

    @classmethod
    def from_payload(cls, value: object) -> "ProjectBinding":
        row = _exact_dict(
            value,
            _PROJECT_BINDING_KEYS,
            "project binding",
        )
        return cls(
            tuple(
                (key, _hex64(row[key], key))
                for key in sorted(_PROJECT_BINDING_KEYS)
            )
        )

    def __post_init__(self) -> None:
        rows = tuple(self.file_sha256)
        if (
            tuple(key for key, _value in rows)
            != tuple(sorted(_PROJECT_BINDING_KEYS))
            or len(rows) != len(_PROJECT_BINDING_KEYS)
        ):
            raise QuacP1SourceFreeCanaryError(
                "project binding registry drifted"
            )
        for key, value in rows:
            _hex64(value, key)
        object.__setattr__(self, "file_sha256", rows)

    def payload(self) -> dict[str, object]:
        return dict(self.file_sha256)

    def verify(self) -> None:
        expected = dict(self.file_sha256)
        for key, relative in _PROJECT_FILES.items():
            path = _PROJECT_ROOT / relative
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise QuacP1SourceFreeCanaryError(
                    "project binding file is unavailable"
                ) from exc
            if (
                path.is_symlink()
                or not stat.S_ISREG(metadata.st_mode)
                or _sha256_file(path) != expected[key]
            ):
                raise QuacP1SourceFreeCanaryError(
                    f"{key} project binding mismatched"
                )


@dataclass(frozen=True)
class DesignBinding:
    file_sha256: str
    self_sha256: str

    @classmethod
    def capture(cls) -> "DesignBinding":
        try:
            value = json.loads(_DESIGN_PATH.read_text("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QuacP1SourceFreeCanaryError(
                "effect design cannot be read"
            ) from exc
        if not isinstance(value, dict):
            raise QuacP1SourceFreeCanaryError(
                "effect design shape drifted"
            )
        return cls(
            file_sha256=_sha256_file(_DESIGN_PATH),
            self_sha256=_hex64(
                value.get("self_sha256"),
                "effect design self",
            ),
        )

    @classmethod
    def from_payload(cls, value: object) -> "DesignBinding":
        row = _exact_dict(
            value,
            _DESIGN_BINDING_KEYS,
            "design binding",
        )
        return cls(
            file_sha256=_hex64(
                row["effect_execution_design_file_sha256"],
                "effect design file",
            ),
            self_sha256=_hex64(
                row["effect_execution_design_self_sha256"],
                "effect design self",
            ),
        )

    def __post_init__(self) -> None:
        _hex64(self.file_sha256, "effect design file")
        _hex64(self.self_sha256, "effect design self")
        if self.self_sha256 != EXPECTED_DESIGN_SELF_SHA256:
            raise QuacP1SourceFreeCanaryError(
                "effect design identity drifted"
            )

    def payload(self) -> dict[str, object]:
        return {
            "effect_execution_design_file_sha256": self.file_sha256,
            "effect_execution_design_self_sha256": self.self_sha256,
        }

    def verify(self) -> None:
        try:
            metadata = _DESIGN_PATH.lstat()
            raw = _DESIGN_PATH.read_bytes()
            value = json.loads(raw.decode("utf-8"))
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise QuacP1SourceFreeCanaryError(
                "effect design cannot be verified"
            ) from exc
        if (
            _DESIGN_PATH.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or not isinstance(value, dict)
            or value.get("schema")
            != "quac_p1_effect_execution_design_v1"
            or hashlib.sha256(raw).hexdigest() != self.file_sha256
        ):
            raise QuacP1SourceFreeCanaryError(
                "effect design file binding mismatched"
            )
        official = value.get("official_HippoRAG")
        if (
            not isinstance(official, Mapping)
            or official.get("source_tree_sha256")
            != EXPECTED_HIPPORAG_NORMATIVE_CONTENT_SHA256
        ):
            raise QuacP1SourceFreeCanaryError(
                "effect design official HippoRAG source binding drifted"
            )
        observed_self = value.get("self_sha256")
        body = {
            key: item
            for key, item in value.items()
            if key != "self_sha256"
        }
        if (
            observed_self != self.self_sha256
            or stable_hash(body) != observed_self
        ):
            raise QuacP1SourceFreeCanaryError(
                "effect design self binding mismatched"
            )


def _tree_payload(value: runtime.FrozenTreeBinding) -> dict[str, object]:
    return {
        "file_count": value.file_count,
        "path": value.path,
        "total_bytes": value.total_bytes,
        "tree_sha256": value.tree_sha256,
    }


def _executable_payload(
    value: runtime.FrozenExecutableBinding,
) -> dict[str, object]:
    return {
        "path": value.path,
        "realpath": value.realpath,
        "sha256": value.sha256,
        "size_bytes": value.size_bytes,
    }


def _python_payload(
    value: runtime.PythonRuntimeBinding,
) -> dict[str, object]:
    return {
        "executable": _executable_payload(value.executable),
        "identity_sha256": value.identity_sha256,
        "import_tree": _tree_payload(value.import_tree),
    }


def runtime_bindings_payload(
    value: runtime.RuntimeBindings,
) -> dict[str, object]:
    if not isinstance(value, runtime.RuntimeBindings):
        raise QuacP1SourceFreeCanaryError(
            "RuntimeBindings is required"
        )
    value.semantic_payload()
    return {
        "gpu0_python": _python_payload(value.gpu0_python),
        "gpu1_base_import_tree": _tree_payload(
            value.gpu1_base_import_tree
        ),
        "gpu1_overlay_import_tree": _tree_payload(
            value.gpu1_overlay_import_tree
        ),
        "gpu1_python": _python_payload(value.gpu1_python),
        "hipporag_source": _tree_payload(value.hipporag_source),
        "llm_alias": value.llm_alias,
        "llm_asset": _tree_payload(value.llm_asset),
        "minilm_alias": value.minilm_alias,
        "minilm_asset": _tree_payload(value.minilm_asset),
    }


def _parse_tree(value: object, field: str) -> runtime.FrozenTreeBinding:
    row = _exact_dict(value, _TREE_KEYS, field)
    return runtime.FrozenTreeBinding(
        path=str(_absolute_path(row["path"], field)),
        tree_sha256=_hex64(row["tree_sha256"], field),
        file_count=_positive_int(row["file_count"], field),
        total_bytes=_nonnegative_int(row["total_bytes"], field),
    )


def _parse_executable(
    value: object,
    field: str,
) -> runtime.FrozenExecutableBinding:
    row = _exact_dict(value, _EXECUTABLE_KEYS, field)
    return runtime.FrozenExecutableBinding(
        path=str(_absolute_path(row["path"], field)),
        realpath=str(_absolute_path(row["realpath"], field)),
        sha256=_hex64(row["sha256"], field),
        size_bytes=_positive_int(row["size_bytes"], field),
    )


def _parse_python(
    value: object,
    field: str,
) -> runtime.PythonRuntimeBinding:
    row = _exact_dict(value, _PYTHON_KEYS, field)
    return runtime.PythonRuntimeBinding(
        executable=_parse_executable(
            row["executable"],
            f"{field} executable",
        ),
        import_tree=_parse_tree(
            row["import_tree"],
            f"{field} import tree",
        ),
        identity_sha256=_hex64(
            row["identity_sha256"],
            f"{field} identity",
        ),
    )


def parse_runtime_bindings(value: object) -> runtime.RuntimeBindings:
    row = _exact_dict(
        value,
        _RUNTIME_BINDING_KEYS,
        "runtime bindings",
    )
    if not isinstance(row["minilm_alias"], str) or not isinstance(
        row["llm_alias"], str
    ):
        raise QuacP1SourceFreeCanaryError(
            "runtime model aliases drifted"
        )
    bindings = runtime.RuntimeBindings(
        gpu0_python=_parse_python(
            row["gpu0_python"],
            "GPU0 Python",
        ),
        gpu1_python=_parse_python(
            row["gpu1_python"],
            "GPU1 Python",
        ),
        gpu1_base_import_tree=_parse_tree(
            row["gpu1_base_import_tree"],
            "GPU1 base import tree",
        ),
        gpu1_overlay_import_tree=_parse_tree(
            row["gpu1_overlay_import_tree"],
            "GPU1 overlay import tree",
        ),
        minilm_asset=_parse_tree(
            row["minilm_asset"],
            "MiniLM asset",
        ),
        llm_asset=_parse_tree(
            row["llm_asset"],
            "LLM asset",
        ),
        hipporag_source=_parse_tree(
            row["hipporag_source"],
            "HippoRAG source",
        ),
        minilm_alias=row["minilm_alias"],
        llm_alias=row["llm_alias"],
    )
    bindings.semantic_payload()
    return bindings


def build_asset_freeze_payload(
    bindings: runtime.RuntimeBindings,
) -> dict[str, object]:
    """Build the separately persisted prospective runtime authority."""

    payload = runtime_bindings_payload(bindings)
    body = {
        "effect_execution_design_self_sha256": (
            EXPECTED_DESIGN_SELF_SHA256
        ),
        "normative_hipporag_source_content_receipt": {
            "file_count": EXPECTED_HIPPORAG_NORMATIVE_FILE_COUNT,
            "size_bytes": EXPECTED_HIPPORAG_NORMATIVE_SIZE_BYTES,
            "tree_sha256": (
                EXPECTED_HIPPORAG_NORMATIVE_CONTENT_SHA256
            ),
        },
        "runtime_binding_sha256": runtime.stable_hash(
            bindings.semantic_payload()
        ),
        "runtime_bindings": payload,
        "schema": ASSET_FREEZE_SCHEMA,
        "status": ASSET_FREEZE_STATUS,
        "study_id": STUDY_ID,
    }
    return {**body, "self_sha256": stable_hash(body)}


@dataclass(frozen=True)
class AssetFreezeBinding:
    path: Path
    file_sha256: str
    self_sha256: str

    @classmethod
    def capture(cls, path: Path) -> "AssetFreezeBinding":
        absolute = _absolute_path(str(path), "asset freeze")
        try:
            raw = absolute.read_bytes()
            value = json.loads(raw.decode("ascii"))
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise QuacP1SourceFreeCanaryError(
                "asset freeze cannot be captured"
            ) from exc
        if (
            type(value) is not dict
            or raw != canonical_bytes(value)
            or value.get("schema") != ASSET_FREEZE_SCHEMA
        ):
            raise QuacP1SourceFreeCanaryError(
                "asset freeze capture drifted"
            )
        return cls(
            path=absolute,
            file_sha256=hashlib.sha256(raw).hexdigest(),
            self_sha256=_hex64(
                value.get("self_sha256"),
                "asset freeze self",
            ),
        )

    @classmethod
    def from_payload(cls, value: object) -> "AssetFreezeBinding":
        row = _exact_dict(
            value,
            _ASSET_FREEZE_BINDING_KEYS,
            "asset freeze binding",
        )
        if row["schema"] != ASSET_FREEZE_SCHEMA:
            raise QuacP1SourceFreeCanaryError(
                "asset freeze binding schema drifted"
            )
        return cls(
            path=_absolute_path(row["path"], "asset freeze"),
            file_sha256=_hex64(
                row["file_sha256"],
                "asset freeze file",
            ),
            self_sha256=_hex64(
                row["self_sha256"],
                "asset freeze self",
            ),
        )

    def __post_init__(self) -> None:
        _absolute_path(str(self.path), "asset freeze")
        _hex64(self.file_sha256, "asset freeze file")
        _hex64(self.self_sha256, "asset freeze self")

    def payload(self) -> dict[str, object]:
        return {
            "file_sha256": self.file_sha256,
            "path": str(self.path),
            "schema": ASSET_FREEZE_SCHEMA,
            "self_sha256": self.self_sha256,
        }

    def verify(self, bindings: runtime.RuntimeBindings) -> None:
        try:
            metadata = self.path.lstat()
            raw = self.path.read_bytes()
            value = json.loads(raw.decode("ascii"))
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise QuacP1SourceFreeCanaryError(
                "asset freeze cannot be verified"
            ) from exc
        body = (
            {
                key: item
                for key, item in value.items()
                if key != "self_sha256"
            }
            if isinstance(value, dict)
            else {}
        )
        expected_normative = {
            "file_count": EXPECTED_HIPPORAG_NORMATIVE_FILE_COUNT,
            "size_bytes": EXPECTED_HIPPORAG_NORMATIVE_SIZE_BYTES,
            "tree_sha256": (
                EXPECTED_HIPPORAG_NORMATIVE_CONTENT_SHA256
            ),
        }
        observed_normative = normative_hipporag_content_receipt(
            Path(bindings.hipporag_source.path)
        )
        if (
            self.path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or type(value) is not dict
            or set(value) != _ASSET_FREEZE_KEYS
            or raw != canonical_bytes(value)
            or hashlib.sha256(raw).hexdigest() != self.file_sha256
            or value.get("self_sha256") != self.self_sha256
            or stable_hash(body) != self.self_sha256
            or value.get("schema") != ASSET_FREEZE_SCHEMA
            or value.get("status") != ASSET_FREEZE_STATUS
            or value.get("study_id") != STUDY_ID
            or value.get("effect_execution_design_self_sha256")
            != EXPECTED_DESIGN_SELF_SHA256
            or value.get("normative_hipporag_source_content_receipt")
            != expected_normative
            or observed_normative != expected_normative
            or value.get("runtime_bindings")
            != runtime_bindings_payload(bindings)
            or value.get("runtime_binding_sha256")
            != runtime.stable_hash(bindings.semantic_payload())
        ):
            raise QuacP1SourceFreeCanaryError(
                "prospective asset freeze binding mismatched"
            )


@dataclass(frozen=True)
class SourceFreeCanaryConfig:
    work_root: Path
    project_binding: ProjectBinding
    design_binding: DesignBinding
    asset_freeze_binding: AssetFreezeBinding
    runtime_bindings: runtime.RuntimeBindings
    self_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.work_root, Path):
            raise QuacP1SourceFreeCanaryError(
                "canary work root type drifted"
            )
        _absolute_path(str(self.work_root), "canary work root")
        if (
            not isinstance(self.project_binding, ProjectBinding)
            or not isinstance(self.design_binding, DesignBinding)
            or not isinstance(
                self.asset_freeze_binding,
                AssetFreezeBinding,
            )
            or not isinstance(
                self.runtime_bindings,
                runtime.RuntimeBindings,
            )
        ):
            raise QuacP1SourceFreeCanaryError(
                "canary binding type drifted"
            )
        _hex64(self.self_sha256, "canary config self")
        if self.self_sha256 != stable_hash(self.body()):
            raise QuacP1SourceFreeCanaryError(
                "canary config self hash drifted"
            )

    def body(self) -> dict[str, object]:
        return {
            "asset_freeze_binding": self.asset_freeze_binding.payload(),
            "design_binding": self.design_binding.payload(),
            "project_binding": self.project_binding.payload(),
            "runtime_bindings": runtime_bindings_payload(
                self.runtime_bindings
            ),
            "schema": CONFIG_SCHEMA,
            "work_root": str(self.work_root),
        }

    def payload(self) -> dict[str, object]:
        return {**self.body(), "self_sha256": self.self_sha256}


def build_config_payload(
    *,
    work_root: Path,
    bindings: runtime.RuntimeBindings,
    asset_freeze_path: Path,
) -> dict[str, object]:
    """Capture a source-free canonical config payload before the canary."""

    project = ProjectBinding.capture()
    design = DesignBinding.capture()
    asset_freeze = AssetFreezeBinding.capture(asset_freeze_path)
    body = {
        "asset_freeze_binding": asset_freeze.payload(),
        "design_binding": design.payload(),
        "project_binding": project.payload(),
        "runtime_bindings": runtime_bindings_payload(bindings),
        "schema": CONFIG_SCHEMA,
        "work_root": str(
            _absolute_path(str(work_root), "canary work root")
        ),
    }
    return {**body, "self_sha256": stable_hash(body)}


def parse_config(value: object) -> SourceFreeCanaryConfig:
    row = _exact_dict(value, _CONFIG_KEYS, "canary config")
    if row["schema"] != CONFIG_SCHEMA:
        raise QuacP1SourceFreeCanaryError(
            "canary config schema drifted"
        )
    body = {
        key: item
        for key, item in row.items()
        if key != "self_sha256"
    }
    supplied_self = _hex64(
        row["self_sha256"],
        "canary config self",
    )
    if stable_hash(body) != supplied_self:
        raise QuacP1SourceFreeCanaryError(
            "canary config self hash drifted"
        )
    return SourceFreeCanaryConfig(
        work_root=_absolute_path(
            row["work_root"],
            "canary work root",
        ),
        project_binding=ProjectBinding.from_payload(
            row["project_binding"]
        ),
        design_binding=DesignBinding.from_payload(
            row["design_binding"]
        ),
        asset_freeze_binding=AssetFreezeBinding.from_payload(
            row["asset_freeze_binding"]
        ),
        runtime_bindings=parse_runtime_bindings(
            row["runtime_bindings"]
        ),
        self_sha256=supplied_self,
    )


def load_config(path: Path) -> SourceFreeCanaryConfig:
    path = _absolute_path(str(path), "canary config")
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise QuacP1SourceFreeCanaryError(
            "canary config cannot be read"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or type(value) is not dict
        or raw != canonical_bytes(value)
    ):
        raise QuacP1SourceFreeCanaryError(
            "canary config is not a direct mode-0400 canonical file"
        )
    return parse_config(value)


def _synthetic_id(label: str) -> str:
    return hashlib.sha256(
        f"{VERSION}:{label}".encode("ascii")
    ).hexdigest()


def synthetic_block() -> runtime.RuntimeBlock:
    """Return the one immutable public synthetic integration fixture."""

    rows = (
        (
            "observatory",
            "Synthetic Observatory",
            "Calibration",
            (
                "The synthetic observatory installed a blue spectrometer "
                "in 2004. Its calibration ledger is stored in Archive Delta."
            ),
        ),
        (
            "bridge",
            "Synthetic River Bridge",
            "Engineering",
            (
                "The synthetic river bridge uses three steel arches and a "
                "separate maintenance notebook."
            ),
        ),
        (
            "garden",
            "Synthetic Botanical Garden",
            "Collections",
            (
                "The synthetic garden catalogues alpine moss in Gallery "
                "Seven and records watering intervals."
            ),
        ),
        (
            "library",
            "Synthetic Civic Library",
            "Archives",
            (
                "The synthetic library keeps municipal maps in Cabinet "
                "Nine beside a public reading room."
            ),
        ),
        (
            "station",
            "Synthetic Weather Station",
            "Sensors",
            (
                "The synthetic weather station measures wind direction "
                "with a rooftop vane and publishes a weekly summary."
            ),
        ),
    )
    documents = tuple(
        sorted(
            (
                action.BlockDocument(
                    unit_id=_synthetic_id(f"unit:{label}"),
                    context_id=_synthetic_id(f"context:{label}"),
                    title=title,
                    section_title=section,
                    context_window_ordinal=0,
                    text=text,
                )
                for label, title, section, text in rows
            ),
            key=lambda item: item.unit_id,
        )
    )
    query = runtime.RuntimeQuery(
        query_id=_synthetic_id("query"),
        question_turns=(
            action.QuestionTurn(
                "Where is the observatory calibration ledger stored?"
            ),
            action.QuestionTurn(
                "Which instrument was installed at the observatory?"
            ),
        ),
    )
    return runtime.RuntimeBlock(
        block_id=_synthetic_id("block"),
        documents=documents,
        queries=(query,),
    )


def _fresh_work_root(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise QuacP1SourceFreeCanaryError(
            "canary work root is not fresh; retry is forbidden"
        )
    try:
        parent = path.parent
        parent_metadata = parent.lstat()
        if (
            parent.is_symlink()
            or not stat.S_ISDIR(parent_metadata.st_mode)
        ):
            raise QuacP1SourceFreeCanaryError(
                "canary work-root parent drifted"
            )
        os.mkdir(path, mode=0o700)
        os.chmod(path, 0o700)
        metadata = path.lstat()
    except OSError as exc:
        raise QuacP1SourceFreeCanaryError(
            "canary work root cannot be claimed"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise QuacP1SourceFreeCanaryError(
            "canary work-root metadata drifted"
        )


def _verify_token(
    token: object,
    bindings: runtime.RuntimeBindings,
) -> tuple[Mapping[str, object], str]:
    require = getattr(token, "require", None)
    token_sha256 = getattr(token, "token_sha256", None)
    if not callable(require):
        raise QuacP1SourceFreeCanaryError(
            "verified runtime token surface drifted"
        )
    receipt = require(bindings)
    if not isinstance(receipt, Mapping):
        raise QuacP1SourceFreeCanaryError(
            "verified runtime receipt disappeared"
        )
    expected_binding = runtime.stable_hash(
        bindings.semantic_payload()
    )
    body = {
        key: item
        for key, item in receipt.items()
        if key != "self_sha256"
    }
    runtime_receipt = receipt.get("runtime_receipt")
    if (
        receipt.get("schema") != runtime.VERIFIED_BINDINGS_SCHEMA
        or receipt.get("source_access_count_at_verification") != 0
        or receipt.get("full_tree_verification_count") != 1
        or receipt.get("binding_sha256") != expected_binding
        or receipt.get("self_sha256") != runtime.stable_hash(body)
        or not isinstance(runtime_receipt, Mapping)
        or runtime_receipt.get("binding_sha256") != expected_binding
    ):
        raise QuacP1SourceFreeCanaryError(
            "verified runtime receipt semantics drifted"
        )
    return receipt, _hex64(
        token_sha256,
        "runtime verification token",
    )


def _validate_runtime_result(
    *,
    result: runtime.BlockRuntimeResult,
    block: runtime.RuntimeBlock,
    binding_sha256: str,
    token_sha256: str,
) -> Mapping[str, object]:
    if not isinstance(result, runtime.BlockRuntimeResult):
        raise QuacP1SourceFreeCanaryError(
            "block runtime result type drifted"
        )
    expected_query_ids = {row.query_id for row in block.queries}
    expected_unit_ids = {row.unit_id for row in block.documents}
    official = result.official_top5
    safe = result.safe_receipt
    if (
        set(result.actions) != expected_query_ids
        or not isinstance(official, Mapping)
        or set(official) != expected_query_ids
        or any(
            len(tuple(top5)) != action.evaluator.TOP_K
            or set(top5) != expected_unit_ids
            for top5 in official.values()
        )
        or not isinstance(safe, Mapping)
        or set(safe) != _RUNTIME_SAFE_KEYS
    ):
        raise QuacP1SourceFreeCanaryError(
            "two-lane result registry drifted"
        )
    index = safe.get("index_cleanup")
    safe_body = {
        key: item
        for key, item in safe.items()
        if key != "self_sha256"
    }
    if (
        safe.get("schema") != runtime.SAFE_RESULT_SCHEMA
        or safe.get("status") != "passed_label_free_block_runtime"
        or safe.get("block_role") != "A_hold"
        or safe.get("action_count") != SYNTHETIC_QUERY_COUNT
        or safe.get("query_count") != SYNTHETIC_QUERY_COUNT
        or safe.get("logical_action_query_count")
        != SYNTHETIC_QUERY_COUNT
        or safe.get("corpus_count") != SYNTHETIC_DOCUMENT_COUNT
        or safe.get("unique_embedding_count")
        != SYNTHETIC_UNIQUE_EMBEDDING_COUNT
        or safe.get("minilm_encode_call_count") != 1
        or safe.get("official_index_call_count") != 1
        or safe.get("official_retrieve_call_count") != 1
        or safe.get("official_required") is not True
        or safe.get("max_concurrent_physical_model_lanes") != 2
        or safe.get("parallel_submission_barrier_passed") is not True
        or safe.get("attempt_count") != 1
        or safe.get("retry_replay_resample_or_fallback_count") != 0
        or safe.get("API_or_online_evaluation_call_count") != 0
        or safe.get("label_family_qrel_or_answer_input_count") != 0
        or safe.get("asset_binding_sha256") != binding_sha256
        or safe.get("binding_verification_token_sha256")
        != token_sha256
        or safe.get("self_sha256") != runtime.stable_hash(safe_body)
        or not isinstance(index, Mapping)
        or set(index) != _INDEX_KEYS
        or index.get("cleanup_verified") is not True
        or type(index.get("file_count")) is not int
        or index.get("file_count", 0) < 1
        or type(index.get("total_bytes")) is not int
        or index.get("total_bytes", 0) < 1
    ):
        raise QuacP1SourceFreeCanaryError(
            "two-lane aggregate receipt drifted"
        )
    for field in (
        "action_pack_file_sha256",
        "attempt_file_sha256",
        "block_input_file_sha256",
        "minilm_receipt_file_sha256",
        "official_full_rankings_sha256",
        "official_output_file_sha256",
    ):
        _hex64(safe.get(field), f"runtime {field}")
    _hex64(index.get("tree_sha256"), "official index tree")
    return safe


def _safe_failure(
    *,
    config: SourceFreeCanaryConfig,
    stage: str,
    exc: BaseException,
    attempt_file_sha256: str,
) -> None:
    path = config.work_root / TERMINAL_FILENAME
    if path.exists() or path.is_symlink():
        return
    value = _self_hashed(
        {
            "API_or_online_evaluation_call_count": 0,
            "aggregate_only_public_receipt": True,
            "attempt_count": 1,
            "canary_attempt_file_sha256": attempt_file_sha256,
            "config_self_sha256": config.self_sha256,
            "exception_message_sha256": hashlib.sha256(
                str(exc).encode("utf-8", errors="replace")
            ).hexdigest(),
            "exception_type_sha256": hashlib.sha256(
                type(exc).__qualname__.encode("utf-8")
            ).hexdigest(),
            "failure_stage": stage,
            "formal_source_access_count": 0,
            "retry_replay_resample_or_fallback_count": 0,
            "schema": SAFE_FAILURE_SCHEMA,
            "status": "failed_source_free_canary_no_retry",
            "study_id": STUDY_ID,
        }
    )
    try:
        _write_once(path, value)
    except QuacP1SourceFreeCanaryError:
        return


def run_source_free_canary_once(
    config: SourceFreeCanaryConfig,
    *,
    encoder: runtime.MiniLMEncoderProtocol | None = None,
    official_lane: runtime.OfficialLaneProtocol | None = None,
    verified_bindings_token: object | None = None,
    verify_bindings_once: Callable[..., object] = (
        runtime.verify_runtime_bindings_once
    ),
    run_block_once: Callable[..., runtime.BlockRuntimeResult] = (
        runtime.run_block
    ),
    asset_authority_verifier: Callable[
        [SourceFreeCanaryConfig], None
    ] | None = None,
) -> Mapping[str, object]:
    """Run the single source-free production-path canary attempt."""

    if not isinstance(config, SourceFreeCanaryConfig):
        raise QuacP1SourceFreeCanaryError(
            "SourceFreeCanaryConfig is required"
        )
    if config.self_sha256 != stable_hash(config.body()):
        raise QuacP1SourceFreeCanaryError(
            "canary config mutated"
        )
    _fresh_work_root(config.work_root)
    attempt = _self_hashed(
        {
            "API_or_online_evaluation_authorized": False,
            "config_self_sha256": config.self_sha256,
            "formal_source_capability_present": False,
            "retry_replay_resample_or_fallback_authorized": False,
            "schema": ATTEMPT_SCHEMA,
            "source_access_count": 0,
            "study_id": STUDY_ID,
        }
    )
    attempt_file_sha256 = _write_once(
        config.work_root / ATTEMPT_FILENAME,
        attempt,
    )
    stage = "verify_project_and_effect_design_bindings"
    try:
        config.project_binding.verify()
        config.design_binding.verify()
        if asset_authority_verifier is None:
            config.asset_freeze_binding.verify(
                config.runtime_bindings
            )
        else:
            asset_authority_verifier(config)
        stage = "verify_runtime_bindings_before_source"
        token = verified_bindings_token
        if token is None:
            token = verify_bindings_once(
                config.runtime_bindings,
                source_access_count=0,
            )
        verified_receipt, token_sha256 = _verify_token(
            token,
            config.runtime_bindings,
        )
        binding_sha256 = _hex64(
            verified_receipt.get("binding_sha256"),
            "verified runtime binding",
        )
        stage = "construct_fixed_source_free_fixture"
        block = synthetic_block()
        if (
            len(block.documents) != SYNTHETIC_DOCUMENT_COUNT
            or len(block.queries) != SYNTHETIC_QUERY_COUNT
        ):
            raise QuacP1SourceFreeCanaryError(
                "fixed synthetic fixture drifted"
            )
        active_encoder = encoder
        if active_encoder is None:
            active_encoder = runtime.LocalMiniLMGpu0Encoder(
                Path(config.runtime_bindings.minilm_asset.path)
            )
        active_official_lane = official_lane
        if active_official_lane is None:
            active_official_lane = runtime.LocalOfficialGpu1Lane()
        stage = "run_two_physical_lanes_single_official_index"
        result = run_block_once(
            block_role="A_hold",
            block=block,
            work_root=(
                config.work_root / INNER_RUNTIME_DIRECTORY
            ),
            bindings=config.runtime_bindings,
            verified_bindings=token,
            encoder=active_encoder,
            official_lane=active_official_lane,
        )
        stage = "validate_aggregate_two_lane_terminal"
        inner_safe = _validate_runtime_result(
            result=result,
            block=block,
            binding_sha256=binding_sha256,
            token_sha256=token_sha256,
        )
        terminal = _self_hashed(
            {
                "API_or_online_evaluation_call_count": 0,
                "aggregate_only_public_receipt": True,
                "asset_freeze_self_sha256": (
                    config.asset_freeze_binding.self_sha256
                ),
                "canary_attempt_file_sha256": attempt_file_sha256,
                "config_self_sha256": config.self_sha256,
                "effect_execution_design_self_sha256": (
                    config.design_binding.self_sha256
                ),
                "formal_source_access_count": 0,
                "max_concurrent_physical_model_lanes": 2,
                "minilm_encode_call_count": 1,
                "official_index_call_count": 1,
                "official_retrieve_call_count": 1,
                "parallel_submission_barrier_passed": True,
                "project_binding_sha256": stable_hash(
                    config.project_binding.payload()
                ),
                "retry_replay_resample_or_fallback_count": 0,
                "runtime_binding_sha256": binding_sha256,
                "runtime_safe_terminal_self_sha256": inner_safe[
                    "self_sha256"
                ],
                "runtime_verification_token_sha256": token_sha256,
                "schema": SAFE_TERMINAL_SCHEMA,
                "source_path_loader_label_qrel_answer_input_count": 0,
                "status": (
                    "passed_source_free_two_lane_single_index_canary"
                ),
                "study_id": STUDY_ID,
                "synthetic_document_count": SYNTHETIC_DOCUMENT_COUNT,
                "synthetic_query_count": SYNTHETIC_QUERY_COUNT,
            }
        )
        _write_once(
            config.work_root / TERMINAL_FILENAME,
            terminal,
        )
        return terminal
    except BaseException as exc:
        _safe_failure(
            config=config,
            stage=stage,
            exc=exc,
            attempt_file_sha256=attempt_file_sha256,
        )
        if isinstance(exc, QuacP1SourceFreeCanaryError):
            raise
        raise QuacP1SourceFreeCanaryError(
            "source-free canary failed closed; retry is forbidden"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=VERSION,
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        run_source_free_canary_once(config)
    except BaseException:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ASSET_FREEZE_SCHEMA",
    "ASSET_FREEZE_STATUS",
    "ATTEMPT_FILENAME",
    "ATTEMPT_SCHEMA",
    "AssetFreezeBinding",
    "CONFIG_SCHEMA",
    "DesignBinding",
    "EXPECTED_DESIGN_SELF_SHA256",
    "EXPECTED_HIPPORAG_NORMATIVE_CONTENT_SHA256",
    "EXPECTED_HIPPORAG_NORMATIVE_FILE_COUNT",
    "EXPECTED_HIPPORAG_NORMATIVE_SIZE_BYTES",
    "INNER_RUNTIME_DIRECTORY",
    "ProjectBinding",
    "QuacP1SourceFreeCanaryError",
    "SAFE_FAILURE_SCHEMA",
    "SAFE_TERMINAL_SCHEMA",
    "STUDY_ID",
    "SYNTHETIC_DOCUMENT_COUNT",
    "SYNTHETIC_QUERY_COUNT",
    "SYNTHETIC_UNIQUE_EMBEDDING_COUNT",
    "SourceFreeCanaryConfig",
    "TERMINAL_FILENAME",
    "VERSION",
    "build_config_payload",
    "build_asset_freeze_payload",
    "canonical_bytes",
    "load_config",
    "main",
    "normative_hipporag_content_receipt",
    "parse_config",
    "parse_runtime_bindings",
    "run_source_free_canary_once",
    "runtime_bindings_payload",
    "stable_hash",
    "synthetic_block",
]
