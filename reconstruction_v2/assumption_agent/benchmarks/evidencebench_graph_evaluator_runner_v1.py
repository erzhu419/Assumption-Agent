"""Stage-isolated EvidenceBench graph/evaluator runner.

This module never locates, downloads, or parses the EvidenceBench dataset.  It
accepts only acquisition-produced stage packs with separate label-free and
label schemas.  All MiniLM, official HippoRAG, typed-action, and official
postflight work reaches a terminal barrier before the relevant label loader is
called.  There is deliberately no F-search label loader.

Formal hashes live only in a future, externally self-hashed implementation
freeze.  This file hard-codes its path, schema, and required interface roles,
but never its own future hash or a future Git HEAD.  Formal verification uses
two fixed read-only Git commands (``rev-parse HEAD`` and ``ls-tree`` restricted
to freeze-listed files or canonical committed prior public receipts).  It
never passes a source, secret, private-pack, work, or current-stage output path
to Git.  Unit tests use only synthetic packs, a mock encoder, and a mock
official runtime.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from replication_runtime.qasper_minilm_v1 import (
    OfflineMiniLMEncoder,
    quantized_cosine_similarity,
)

from .evidencebench_typed_scientific_graph_v1 import (
    AFormationSelection,
    ActionTrace,
    CoverageComponents,
    FSearchSelection,
    FormationItem,
    SourceNode,
    TypedEdge,
    build_common_candidate_table,
    build_typed_scientific_graph,
    coverage_components,
    embedding_text,
    evaluator_registry,
    exact_magnitude_preserving_sign_flip,
    execute_all_recipes,
    execute_recipe,
    has_identifiable_transition,
    item_utility,
    recipe_registry,
    score_all_evaluators,
    select_a_evaluator,
    select_f_recipe,
)
from .musique_formal_runtime_binding_v2 import (
    PreparedFormalRuntimeV2,
    prepare_formal_runtime_v2,
)


VERSION = "evidencebench_graph_evaluator_runner_v1"

PENDING = "__PENDING__"
IMPLEMENTATION_FREEZE_RELATIVE_PATH = Path(
    "manifests/evidencebench_implementation_freeze_v1.json"
)
IMPLEMENTATION_FREEZE_SCHEMA = "evidencebench_implementation_freeze_v1"
IMPLEMENTATION_FREEZE_HASH_FIELD = "implementation_freeze_sha256"
EXPECTED_BINDING_INTERFACES: dict[str, dict[str, str]] = {
    "design": {
        "relative_path": "manifests/evidencebench_graph_evaluator_design_v1.json",
        "schema": "evidencebench_graph_evaluator_design_v1",
        "semantic_hash_field": "design_sha256",
    },
    "custody": {
        "relative_path": "manifests/evidencebench_graph_evaluator_source_custody_v1.json",
        "schema": "evidencebench_graph_evaluator_source_custody_v1",
        "semantic_hash_field": "custody_sha256",
    },
    "source_access": {
        "relative_path": "manifests/evidencebench_graph_evaluator_source_access_v1.json",
        "schema": "evidencebench_graph_evaluator_source_access_v1",
        "semantic_hash_field": "source_access_sha256",
    },
    "graph_core": {
        "relative_path": (
            "assumption_agent/benchmarks/"
            "evidencebench_typed_scientific_graph_v1.py"
        ),
        "version": "evidencebench_typed_scientific_graph_v1",
    },
    "graph_core_test": {
        "relative_path": "tests/test_evidencebench_typed_scientific_graph_v1.py",
    },
    "acquisition_runner": {
        "relative_path": (
            "assumption_agent/benchmarks/evidencebench_direct_acquisition_v1.py"
        ),
        "version": "evidencebench_direct_acquisition_v1",
    },
    "acquisition_test": {
        "relative_path": "tests/test_evidencebench_direct_acquisition_v1.py",
    },
    "evaluator_runner": {
        "relative_path": (
            "assumption_agent/benchmarks/"
            "evidencebench_graph_evaluator_runner_v1.py"
        ),
        "version": VERSION,
    },
    "evaluator_test": {
        "relative_path": "tests/test_evidencebench_graph_evaluator_runner_v1.py",
    },
}

MINILM_ASSET_SHA256 = "921d9b1945581130e03c53f448092c3de3b30714431c6cac9b3b32c2ec10abad"
MINILM_ASSET_FILE_SHA256 = (
    "62b85c7752f2e46932fb9fb13ae2f3aac9eb750a33c8f07102739040feb6cc75"
)

LABEL_FREE_SCHEMA = "evidencebench_direct_v1_label_free_block"
LABEL_FREE_ITEM_SCHEMA = "evidencebench_direct_v1_label_free_item"
LABEL_SCHEMA = "evidencebench_direct_v1_label_block"
LABEL_ITEM_SCHEMA = "evidencebench_direct_v1_label_item"
BLOCKS = ("A_form", "F_search", "A_hold", "M_search")
LABEL_BLOCKS = ("A_form", "A_hold", "M_search")
BLOCK_COUNT = 64
NODE_COUNT = 32
TOP_K = 5
OFFICIAL_CONCURRENCY_CAP = 8
LOCAL_ITEM_CONCURRENCY_CAP = 64
MAX_PRIVATE_BYTES = 64 * 1024 * 1024
MAX_PUBLIC_RECEIPT_BYTES = 1024 * 1024
ENCODER_DIMENSION = 384
R0 = "R0_HIPPO_TOP5"
RECIPE_IDS = tuple(recipe.recipe_id for recipe in recipe_registry())
EVALUATOR_IDS = tuple(evaluator.evaluator_id for evaluator in evaluator_registry())
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")

CANONICAL_ACQUISITION_RECEIPT_RELATIVE_PATH = Path(
    "manifests/evidencebench_direct_acquisition_v1.json"
)
ACQUISITION_RECEIPT_SCHEMA = "evidencebench_direct_acquisition_v1"
CANONICAL_PRIVATE_PACKS: dict[str, tuple[Path, Path]] = {
    "A_form": (
        Path(
            "artifacts/evidencebench_direct_acquisition_v1/"
            "A_form.label_free.private.json"
        ),
        Path(
            "artifacts/evidencebench_direct_acquisition_v1/"
            "A_form.labels.private.json"
        ),
    ),
    "F_search": (
        Path(
            "artifacts/evidencebench_direct_acquisition_v1/"
            "F_search.label_free.private.json"
        ),
        Path(
            "artifacts/evidencebench_direct_acquisition_v1/"
            "F_search.labels.sealed.json"
        ),
    ),
    "A_hold": (
        Path(
            "artifacts/evidencebench_direct_acquisition_v1/"
            "A_hold.label_free.sealed.json"
        ),
        Path(
            "artifacts/evidencebench_direct_acquisition_v1/"
            "A_hold.labels.sealed.json"
        ),
    ),
    "M_search": (
        Path(
            "artifacts/evidencebench_direct_acquisition_v1/"
            "M_search.label_free.sealed.json"
        ),
        Path(
            "artifacts/evidencebench_direct_acquisition_v1/"
            "M_search.labels.sealed.json"
        ),
    ),
}
CANONICAL_STAGE_PATHS: dict[str, dict[str, Path]] = {
    "formation": {
        "root": Path(
            "artifacts/evidencebench_graph_evaluator_runner_v1/formation"
        ),
        "receipt": Path(
            "manifests/evidencebench_graph_evaluator_formation_v1.json"
        ),
        "failure": Path(
            "manifests/evidencebench_graph_evaluator_formation_failure_v1.json"
        ),
    },
    "A_hold": {
        "root": Path(
            "artifacts/evidencebench_graph_evaluator_runner_v1/A_hold"
        ),
        "receipt": Path(
            "manifests/evidencebench_graph_evaluator_A_hold_v1.json"
        ),
        "failure": Path(
            "manifests/evidencebench_graph_evaluator_A_hold_failure_v1.json"
        ),
    },
    "M_search": {
        "root": Path(
            "artifacts/evidencebench_graph_evaluator_runner_v1/M_search"
        ),
        "receipt": Path(
            "manifests/evidencebench_graph_evaluator_M_search_v1.json"
        ),
        "failure": Path(
            "manifests/evidencebench_graph_evaluator_M_search_failure_v1.json"
        ),
    },
}

LABEL_FREE_BLOCK_KEYS = frozenset(
    {"schema", "block", "count", "rows", "block_sha256"}
)
LABEL_FREE_ITEM_KEYS = frozenset(
    {
        "schema",
        "block",
        "ordinal",
        "item_commitment_sha256",
        "paper_commitment_sha256",
        "component_commitment_sha256",
        "hypothesis",
        "title",
        "nodes",
    }
)
NODE_KEYS = frozenset({"span_i", "start", "end", "identity_text"})
LABEL_BLOCK_KEYS = frozenset(
    {"schema", "block", "count", "rows", "block_sha256"}
)
LABEL_ITEM_KEYS = frozenset(
    {
        "schema",
        "block",
        "ordinal",
        "item_commitment_sha256",
        "gold_aspect_node_indices",
    }
)


class EvidenceBenchGraphEvaluatorRunnerError(RuntimeError):
    """A frozen pack, runtime, action, stage, or receipt contract drifted."""


class EncoderProtocol(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class OfficialRuntimeProtocol(Protocol):
    @property
    def safe_binding(self) -> Mapping[str, Any]: ...

    def retrieve(
        self,
        *,
        question: str,
        paragraphs: Sequence[Mapping[str, object]],
        work_root: Path,
    ) -> tuple[int, ...]: ...

    def fresh_reverify(self) -> Mapping[str, Any]: ...


ProgressHook = Callable[[str, str | None, int | None], None]


@dataclass(frozen=True)
class PrivateNode:
    span_i: int
    start: int
    end: int
    identity_text: str

    def source_node(self) -> SourceNode:
        return SourceNode(self.span_i, self.start, self.end, self.identity_text)


@dataclass(frozen=True)
class LabelFreeItem:
    block: str
    ordinal: int
    item_commitment_sha256: str
    paper_commitment_sha256: str
    component_commitment_sha256: str
    hypothesis: str
    nodes: tuple[PrivateNode, ...]

    @property
    def source_nodes(self) -> tuple[SourceNode, ...]:
        return tuple(node.source_node() for node in self.nodes)

    @property
    def paragraphs(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "idx": node.span_i,
                "title": "EvidenceBench_paper",
                "paragraph_text": node.identity_text,
            }
            for node in self.nodes
        )


@dataclass(frozen=True)
class LabelFreeBlock:
    block: str
    block_sha256: str
    file_sha256: str
    rows: tuple[LabelFreeItem, ...]


@dataclass(frozen=True)
class LabelItem:
    block: str
    ordinal: int
    item_commitment_sha256: str
    gold_aspect_node_indices: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class LabelBlock:
    block: str
    block_sha256: str
    file_sha256: str
    rows: tuple[LabelItem, ...]


@dataclass(frozen=True)
class AcquisitionReceiptBinding:
    acquisition_sha256: str
    file_sha256: str
    git_blob_sha1: str
    verified_at_git_HEAD: str
    commitments_by_block: Mapping[str, Mapping[str, object]]
    payload: Mapping[str, object]


@dataclass(frozen=True)
class PriorStageReceiptBinding:
    stage: str
    receipt_sha256: str
    file_sha256: str
    git_blob_sha1: str
    verified_at_git_HEAD: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class LocalTensor:
    raw_top5: tuple[int, int, int, int, int]
    query_node_similarities: tuple[int, ...]
    node_node_similarities: tuple[tuple[int, ...], ...]
    typed_edges: tuple[TypedEdge, ...]


@dataclass(frozen=True)
class FullItemActions:
    block: str
    ordinal: int
    raw_top5: tuple[int, int, int, int, int]
    official_top5: tuple[int, int, int, int, int]
    traces_by_recipe: Mapping[str, ActionTrace]
    components_by_recipe: Mapping[str, CoverageComponents]
    evaluator_score_table_sha256: str


@dataclass(frozen=True)
class MeasurementItemActions:
    block: str
    ordinal: int
    raw_top5: tuple[int, int, int, int, int]
    official_top5: tuple[int, int, int, int, int]
    agent_top5: tuple[int, int, int, int, int]
    common_scan_sha256: str


@dataclass(frozen=True)
class FormationOutcome:
    a_block: LabelFreeBlock
    f_block: LabelFreeBlock
    a_labels: LabelBlock
    a_selection: AFormationSelection
    f_selection: FSearchSelection
    identifiable_transition: bool
    a_arm_aggregates: Mapping[str, Mapping[str, int]]
    a_label_histograms: Mapping[str, Mapping[str, int]]
    action_table_sha256: str
    runtime_binding_sha256: str


@dataclass(frozen=True)
class MeasurementOutcome:
    block: LabelFreeBlock
    labels: LabelBlock
    selected_evaluator_id: str | None
    selected_recipe_id: str
    arm_aggregates: Mapping[str, Mapping[str, int]]
    label_histograms: Mapping[str, Mapping[str, int]]
    delta_vector_sha256: str
    exact_test: Mapping[str, object]
    action_table_sha256: str
    runtime_binding_sha256: str


def _noop_progress(
    _event: str, _block: str | None, _ordinal: int | None
) -> None:
    return None


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "value is not canonical JSON"
        ) from exc


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            f"{field} must be lowercase sha256"
        )
    return value


def _assert_no_symlink_components(path: Path, field: str) -> Path:
    absolute = path.expanduser().absolute()
    for candidate in (*reversed(absolute.parents), absolute):
        if candidate.is_symlink():
            raise EvidenceBenchGraphEvaluatorRunnerError(
                f"{field} contains a symbolic link"
            )
    return absolute


def _canonical_project_root(project_root: Path) -> Path:
    absolute = _assert_no_symlink_components(
        project_root.expanduser().absolute(), "project root"
    )
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "project root is unavailable"
        ) from exc
    if resolved != absolute or not resolved.is_dir():
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "project root is not canonical"
        )
    return resolved


def _block_hash(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("block_sha256", None)
    return _semantic_hash(body)


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def _find_git_root(project_root: Path) -> Path:
    for candidate in (project_root, *project_root.parents):
        marker = candidate / ".git"
        if (marker.is_dir() or marker.is_file()) and not marker.is_symlink():
            return candidate
    raise EvidenceBenchGraphEvaluatorRunnerError(
        "Git worktree root is unavailable"
    )


def _run_read_only_git(
    git_root: Path, arguments: Sequence[str]
) -> subprocess.CompletedProcess[str]:
    allowed = {"rev-parse", "ls-tree"}
    if not arguments or arguments[0] not in allowed:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "Git verification command is not allowlisted"
        )
    try:
        result = subprocess.run(
            ("git", "-C", str(git_root), *arguments),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=30,
            shell=False,
        )
    except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "read-only Git verification failed"
        ) from exc
    if result.returncode != 0 or result.stderr.strip():
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "read-only Git verification was not terminal-clean"
        )
    return result


def _head_blob_table(
    *, project_root: Path, relative_paths: Sequence[str]
) -> tuple[str, dict[str, str], dict[str, object]]:
    git_root = _find_git_root(project_root)
    head_result = _run_read_only_git(git_root, ("rev-parse", "HEAD"))
    head = head_result.stdout.strip()
    if SHA1_RE.fullmatch(head) is None:
        raise EvidenceBenchGraphEvaluatorRunnerError("Git HEAD is malformed")
    project_prefix = project_root.resolve(strict=True).relative_to(
        git_root.resolve(strict=True)
    )
    git_paths: list[str] = []
    project_by_git_path: dict[str, str] = {}
    for relative in relative_paths:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "implementation-freeze path is unsafe"
            )
        git_path = (project_prefix / candidate).as_posix()
        git_paths.append(git_path)
        project_by_git_path[git_path] = relative
    if len(set(git_paths)) != len(git_paths):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "implementation-freeze paths are not unique"
        )
    tree_result = _run_read_only_git(
        git_root, ("ls-tree", "-r", head, "--", *git_paths)
    )
    observed: dict[str, str] = {}
    for line in tree_result.stdout.splitlines():
        try:
            metadata, git_path = line.split("\t", 1)
            mode, object_type, object_id = metadata.split(" ", 2)
        except ValueError as exc:
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "Git tree output is malformed"
            ) from exc
        if (
            git_path not in project_by_git_path
            or mode not in {"100644", "100755"}
            or object_type != "blob"
            or SHA1_RE.fullmatch(object_id) is None
        ):
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "Git tree binding is invalid"
            )
        observed[project_by_git_path[git_path]] = object_id
    if set(observed) != set(relative_paths):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "implementation freeze or listed path is absent from HEAD"
        )
    audit = {
        "HEAD": head,
        "commands": [
            {"command": "rev-parse_HEAD", "returncode": head_result.returncode},
            {
                "command": "ls-tree_r_HEAD_restricted_paths",
                "returncode": tree_result.returncode,
                "path_count": len(git_paths),
            },
        ],
        "source_secret_private_or_current_stage_output_path_passed": False,
    }
    return head, observed, audit


def _read_committed_public_prerequisite(
    *,
    project_root: Path,
    relative_path: Path,
    expected_current_head: str,
) -> tuple[dict[str, Any], bytes, str, str, dict[str, object]]:
    absolute = _assert_no_symlink_components(
        project_root / relative_path, "canonical public prerequisite"
    )
    if (
        not absolute.is_file()
        or not 1 <= absolute.stat().st_size <= MAX_PUBLIC_RECEIPT_BYTES
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "canonical public prerequisite is unavailable"
        )
    payload, raw = _read_json_object(
        absolute, "canonical public prerequisite"
    )
    head, blobs, audit = _head_blob_table(
        project_root=project_root,
        relative_paths=(relative_path.as_posix(),),
    )
    blob = _git_blob_sha1(raw)
    if head != expected_current_head or blobs[relative_path.as_posix()] != blob:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "canonical public prerequisite does not byte-match current HEAD"
        )
    audit = {
        **audit,
        "committed_public_prior_output_path_count": 1,
        "current_stage_output_path_passed": False,
    }
    return payload, raw, _sha256_bytes(raw), blob, audit


def _read_json_object(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            f"{field} is unreadable"
        ) from exc
    if not isinstance(payload, dict):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            f"{field} root must be an object"
        )
    return payload, raw


def verify_design_binding(project_root: Path) -> dict[str, Any]:
    """Verify the external freeze and every listed worktree blob against HEAD."""

    root = _canonical_project_root(project_root)
    freeze_path = _assert_no_symlink_components(
        root / IMPLEMENTATION_FREEZE_RELATIVE_PATH,
        "implementation-freeze path",
    )
    if not freeze_path.is_file():
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "implementation freeze is unavailable"
        )
    freeze, freeze_raw = _read_json_object(
        freeze_path, "implementation freeze"
    )
    expected_top_keys = {
        "schema",
        "bindings",
        "source_binding",
        "selection_secret_commitment",
        "freeze_hash_contract",
        IMPLEMENTATION_FREEZE_HASH_FIELD,
    }
    if set(freeze) != expected_top_keys or freeze.get(
        "schema"
    ) != IMPLEMENTATION_FREEZE_SCHEMA:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "implementation-freeze schema drifted"
        )
    declared_freeze = _require_sha256(
        freeze.get(IMPLEMENTATION_FREEZE_HASH_FIELD),
        "implementation-freeze semantic hash",
    )
    freeze_body = dict(freeze)
    freeze_body.pop(IMPLEMENTATION_FREEZE_HASH_FIELD)
    if _semantic_hash(freeze_body) != declared_freeze:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "implementation-freeze self-hash drifted"
        )
    hash_contract = freeze.get("freeze_hash_contract")
    if not isinstance(hash_contract, Mapping) or (
        hash_contract.get("algorithm") != "sha256"
        or hash_contract.get("excluded_top_level_fields")
        != [IMPLEMENTATION_FREEZE_HASH_FIELD]
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "implementation-freeze hash contract drifted"
        )
    selection_commitment = freeze.get("selection_secret_commitment")
    if _require_sha256(
        selection_commitment, "selection-secret commitment"
    ) != selection_commitment:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "selection-secret commitment drifted"
        )
    if not isinstance(freeze.get("source_binding"), Mapping):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "source binding is absent from implementation freeze"
        )
    bindings = freeze.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(
        EXPECTED_BINDING_INTERFACES
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "implementation-freeze roles drifted"
        )
    normalized_bindings: dict[str, dict[str, str]] = {}
    for role, interface in EXPECTED_BINDING_INTERFACES.items():
        raw_binding = bindings.get(role)
        if not isinstance(raw_binding, Mapping):
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "implementation binding row is invalid"
            )
        relative = raw_binding.get("relative_path")
        file_sha = raw_binding.get("file_sha256")
        blob_sha = raw_binding.get("git_blob_sha1")
        if (
            relative != interface["relative_path"]
            or _require_sha256(file_sha, f"{role} file hash") != file_sha
            or not isinstance(blob_sha, str)
            or SHA1_RE.fullmatch(blob_sha) is None
        ):
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "implementation binding row drifted"
            )
        if "version" in interface and raw_binding.get("version") != interface[
            "version"
        ]:
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "implementation version binding drifted"
            )
        if "schema" in interface and raw_binding.get("schema") != interface[
            "schema"
        ]:
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "manifest schema binding drifted"
            )
        normalized_bindings[role] = {
            str(key): str(value) for key, value in raw_binding.items()
        }
    restricted_paths = (
        IMPLEMENTATION_FREEZE_RELATIVE_PATH.as_posix(),
        *(row["relative_path"] for row in normalized_bindings.values()),
    )
    head, head_blobs, git_audit = _head_blob_table(
        project_root=root, relative_paths=restricted_paths
    )
    if _git_blob_sha1(freeze_raw) != head_blobs[
        IMPLEMENTATION_FREEZE_RELATIVE_PATH.as_posix()
    ]:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "implementation freeze does not match HEAD"
        )
    manifest_payloads: dict[str, dict[str, Any]] = {}
    for role, binding in normalized_bindings.items():
        path = _assert_no_symlink_components(
            root / binding["relative_path"], f"{role} binding path"
        )
        if not path.is_file():
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "implementation binding file is unavailable"
            )
        raw = path.read_bytes()
        if (
            _sha256_bytes(raw) != binding["file_sha256"]
            or _git_blob_sha1(raw) != binding["git_blob_sha1"]
            or head_blobs[binding["relative_path"]]
            != binding["git_blob_sha1"]
        ):
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "implementation file or HEAD blob drifted"
            )
        interface = EXPECTED_BINDING_INTERFACES[role]
        if "schema" in interface:
            payload, _payload_raw = _read_json_object(path, f"{role} manifest")
            hash_field = interface["semantic_hash_field"]
            declared = _require_sha256(
                payload.get(hash_field), f"{role} semantic hash"
            )
            body = dict(payload)
            body.pop(hash_field)
            if (
                payload.get("schema") != interface["schema"]
                or _semantic_hash(body) != declared
                or binding.get("semantic_sha256") != declared
            ):
                raise EvidenceBenchGraphEvaluatorRunnerError(
                    "bound manifest self-hash drifted"
                )
            manifest_payloads[role] = payload
    design = manifest_payloads["design"]
    rendered_design = _canonical_bytes(design).decode("ascii")
    for required in (
        IMPLEMENTATION_FREEZE_RELATIVE_PATH.as_posix(),
        IMPLEMENTATION_FREEZE_SCHEMA,
        EXPECTED_BINDING_INTERFACES["acquisition_runner"]["relative_path"],
        EXPECTED_BINDING_INTERFACES["acquisition_runner"]["version"],
        EXPECTED_BINDING_INTERFACES["evaluator_runner"]["relative_path"],
        VERSION,
        LABEL_FREE_SCHEMA,
        LABEL_SCHEMA,
        LABEL_FREE_ITEM_SCHEMA,
        LABEL_ITEM_SCHEMA,
    ):
        if required not in rendered_design:
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "design interface declaration drifted"
            )
    return {
        "implementation_freeze_sha256": declared_freeze,
        "implementation_freeze_file_sha256": _sha256_bytes(freeze_raw),
        "git_HEAD": head,
        "git_verification": git_audit,
        "design_sha256": normalized_bindings["design"]["semantic_sha256"],
        "design_file_sha256": normalized_bindings["design"]["file_sha256"],
        "custody_sha256": normalized_bindings["custody"]["semantic_sha256"],
        "custody_file_sha256": normalized_bindings["custody"]["file_sha256"],
        "source_access_sha256": normalized_bindings["source_access"][
            "semantic_sha256"
        ],
        "source_access_file_sha256": normalized_bindings["source_access"][
            "file_sha256"
        ],
        "graph_core_sha256": normalized_bindings["graph_core"]["file_sha256"],
        "acquisition_runner_sha256": normalized_bindings[
            "acquisition_runner"
        ]["file_sha256"],
        "evaluator_runner_sha256": normalized_bindings["evaluator_runner"][
            "file_sha256"
        ],
        "freeze_bindings": normalized_bindings,
        "freeze_source_binding": dict(freeze["source_binding"]),
        "selection_secret_commitment": selection_commitment,
    }


def _load_canonical_acquisition_receipt(
    *,
    project_root: Path,
    protocol_binding: Mapping[str, Any],
) -> AcquisitionReceiptBinding:
    payload, raw, file_sha, git_blob, _git_audit = (
        _read_committed_public_prerequisite(
            project_root=project_root,
            relative_path=CANONICAL_ACQUISITION_RECEIPT_RELATIVE_PATH,
            expected_current_head=str(protocol_binding["git_HEAD"]),
        )
    )
    if payload.get("schema") != ACQUISITION_RECEIPT_SCHEMA or payload.get(
        "status"
    ) != "private_four_block_pack_formed":
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "canonical acquisition receipt is not a formed four-block pack"
        )
    body = dict(payload)
    declared = _require_sha256(
        body.pop("acquisition_sha256", None), "acquisition receipt hash"
    )
    if _semantic_hash(body) != declared:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "canonical acquisition receipt self-hash drifted"
        )
    protocol = payload.get("protocol_bindings")
    if not isinstance(protocol, Mapping):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "acquisition protocol binding is absent"
        )
    freeze = protocol.get("implementation_freeze")
    if not isinstance(freeze, Mapping) or (
        freeze.get("semantic_sha256")
        != protocol_binding["implementation_freeze_sha256"]
        or freeze.get("file_sha256")
        != protocol_binding["implementation_freeze_file_sha256"]
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "acquisition implementation-freeze chain drifted"
        )
    acquired_head = protocol.get("git_HEAD")
    if not isinstance(acquired_head, Mapping) or (
        SHA1_RE.fullmatch(str(acquired_head.get("head_commit"))) is None
        or acquired_head.get("all_freeze_listed_files_byte_match_HEAD") is not True
        or acquired_head.get(
            "source_secret_private_or_output_path_passed_to_git"
        )
        is not False
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "acquisition-time Git verification drifted"
        )
    freeze_bindings = protocol_binding.get("freeze_bindings")
    acquired_files = protocol.get("protocol_files")
    if (
        not isinstance(freeze_bindings, Mapping)
        or not isinstance(acquired_files, Mapping)
        or set(acquired_files) != set(freeze_bindings)
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "acquisition protocol file role set drifted"
        )
    for role, frozen in freeze_bindings.items():
        acquired = acquired_files.get(role)
        if not isinstance(frozen, Mapping) or not isinstance(acquired, Mapping):
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "acquisition protocol file row is invalid"
            )
        for field in (
            "relative_path",
            "file_sha256",
            "git_blob_sha1",
            "schema",
            "version",
            "semantic_sha256",
        ):
            if field in frozen and acquired.get(field) != frozen.get(field):
                raise EvidenceBenchGraphEvaluatorRunnerError(
                    "acquisition protocol file binding drifted"
                )
    frozen_source = protocol_binding.get("freeze_source_binding")
    source_identity = protocol.get("source_identity")
    public_source = payload.get("source")
    if not isinstance(frozen_source, Mapping) or not isinstance(
        source_identity, Mapping
    ) or not isinstance(public_source, Mapping):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "acquisition source chain is absent"
        )
    source_pairs = (
        ("repository", "repository", "repository"),
        ("commit", "commit", "repository_fixed_commit"),
        ("relative_path", "path", "repository_path"),
        ("git_blob_sha1", "git_blob_sha1", "repository_git_blob_sha1"),
        ("byte_size", "byte_size", "source_file_byte_size"),
    )
    for frozen_key, identity_key, public_key in source_pairs:
        expected = frozen_source.get(frozen_key)
        if (
            source_identity.get(identity_key) != expected
            or public_source.get(public_key) != expected
        ):
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "acquisition source identity drifted"
            )
    if public_source.get("source_file_sha256") != frozen_source.get(
        "whole_file_sha256"
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "acquisition whole-source hash drifted"
        )
    safety = payload.get("safety")
    if not isinstance(safety, Mapping) or (
        safety.get("selection_completed_before_action_or_score") is not True
        or safety.get("performance_scores_computed") != 0
        or safety.get("model_calls") != 0
        or safety.get("network_calls") != 0
        or safety.get("online_evaluator_calls") != 0
        or safety.get("public_item_content_or_identifiers") != 0
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "acquisition safety receipt drifted"
        )
    attempt = payload.get("attempt")
    if not isinstance(attempt, Mapping) or (
        attempt.get("marker_durable_before_source_JSON_open") is not True
        or attempt.get("preregistered_formal_invocation_count") != 1
        or attempt.get("observed_marker_consuming_attempt_count") != 1
        or attempt.get("attempt_marker_creation_count") != 1
        or attempt.get("source_JSON_open_attempt_count") != 1
        or attempt.get("same_source_replay_count") != 0
        or attempt.get("resample_count") != 0
        or attempt.get("secret_rotation_count") != 0
        or attempt.get("parser_or_model_worker_count") != 0
        or attempt.get("retry_replay_resample_authorized") is not False
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "acquisition one-shot attempt receipt drifted"
        )
    aggregate = payload.get("aggregate")
    paper_counts = (
        aggregate.get("paper_counts")
        if isinstance(aggregate, Mapping)
        else None
    )
    if not isinstance(paper_counts, Mapping) or (
        paper_counts.get("capacity_satisfied") is not True
        or paper_counts.get("required") != BLOCK_COUNT * len(BLOCKS)
        or type(paper_counts.get("eligible")) is not int
        or paper_counts["eligible"] < BLOCK_COUNT * len(BLOCKS)
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "acquisition aggregate capacity receipt drifted"
        )
    blocks = payload.get("blocks")
    if not isinstance(blocks, Mapping) or (
        blocks.get("block_order") != list(BLOCKS)
        or blocks.get("block_size") != BLOCK_COUNT
        or blocks.get("selected_item_count") != BLOCK_COUNT * len(BLOCKS)
        or blocks.get("global_paper_disjointness") is not True
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "acquisition block envelope drifted"
        )
    rows = blocks.get("private_file_commitments")
    if not isinstance(rows, list) or len(rows) != len(BLOCKS):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "acquisition private-file commitments are incomplete"
        )
    exact_keys = {
        "block",
        "count",
        "item_commitment_root_sha256",
        "label_free_file_sha256",
        "label_file_sha256",
        "label_free_block_sha256",
        "label_block_sha256",
    }
    commitments: dict[str, Mapping[str, object]] = {}
    for expected_block, row in zip(BLOCKS, rows):
        if not isinstance(row, Mapping) or set(row) != exact_keys or (
            row.get("block") != expected_block
            or row.get("count") != BLOCK_COUNT
        ):
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "acquisition private-file commitment row drifted"
            )
        for field in exact_keys - {"block", "count"}:
            _require_sha256(row.get(field), f"acquisition {field}")
        commitments[expected_block] = dict(row)
    return AcquisitionReceiptBinding(
        acquisition_sha256=declared,
        file_sha256=file_sha,
        git_blob_sha1=git_blob,
        verified_at_git_HEAD=str(protocol_binding["git_HEAD"]),
        commitments_by_block=commitments,
        payload=payload,
    )


def _verify_loaded_pack_commitment(
    *,
    acquisition: AcquisitionReceiptBinding,
    block: str,
    view: LabelFreeBlock,
    labels: LabelBlock | None,
) -> None:
    commitment = acquisition.commitments_by_block.get(block)
    if not isinstance(commitment, Mapping) or (
        view.file_sha256 != commitment.get("label_free_file_sha256")
        or view.block_sha256 != commitment.get("label_free_block_sha256")
        or _semantic_hash(
            [row.item_commitment_sha256 for row in view.rows]
        )
        != commitment.get("item_commitment_root_sha256")
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "label-free pack does not match canonical acquisition receipt"
        )
    if labels is not None and (
        labels.file_sha256 != commitment.get("label_file_sha256")
        or labels.block_sha256 != commitment.get("label_block_sha256")
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "label pack does not match canonical acquisition receipt"
        )


def _canonical_stage_absolutes(
    project_root: Path, stage: str
) -> dict[str, Path]:
    if stage not in CANONICAL_STAGE_PATHS:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "canonical stage identity is invalid"
        )
    return {
        key: _assert_no_symlink_components(
            project_root / relative, f"canonical {stage} {key}"
        )
        for key, relative in CANONICAL_STAGE_PATHS[stage].items()
    }


def _preflight_canonical_stage_outputs(paths: Mapping[str, Path]) -> None:
    if set(paths) != {"root", "receipt", "failure"}:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "canonical stage path map drifted"
        )
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "canonical stage destination already exists; replay is forbidden"
        )


def _read_private_json(path: Path, field: str) -> tuple[dict[str, Any], str]:
    absolute = _assert_no_symlink_components(path, field)
    if not absolute.is_file():
        raise EvidenceBenchGraphEvaluatorRunnerError(f"{field} is unavailable")
    info = absolute.stat()
    if (
        stat.S_IMODE(info.st_mode) != 0o600
        or not 1 <= info.st_size <= MAX_PRIVATE_BYTES
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            f"{field} mode or size is invalid"
        )
    raw = absolute.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            f"{field} is invalid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            f"{field} root must be an object"
        )
    return payload, _sha256_bytes(raw)


def _parse_node(raw: object, expected_i: int, previous_end: int) -> PrivateNode:
    if not isinstance(raw, Mapping) or set(raw) != NODE_KEYS:
        raise EvidenceBenchGraphEvaluatorRunnerError("private node schema drifted")
    span_i = raw.get("span_i")
    start = raw.get("start")
    end = raw.get("end")
    identity_text = raw.get("identity_text")
    if (
        type(span_i) is not int
        or span_i != expected_i
        or type(start) is not int
        or type(end) is not int
        or start < previous_end
        or end <= start
        or not isinstance(identity_text, str)
        or not identity_text
        or "\x00" in identity_text
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "private node content is invalid"
        )
    return PrivateNode(span_i, start, end, identity_text)


def _parse_view_item(
    raw: object, block: str, expected_ordinal: int
) -> LabelFreeItem:
    if not isinstance(raw, Mapping) or set(raw) != LABEL_FREE_ITEM_KEYS:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "label-free item schema drifted"
        )
    if (
        raw.get("schema") != LABEL_FREE_ITEM_SCHEMA
        or raw.get("block") != block
        or type(raw.get("ordinal")) is not int
        or raw.get("ordinal") != expected_ordinal
        or raw.get("title") != "EvidenceBench_paper"
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "label-free item identity drifted"
        )
    hypothesis = raw.get("hypothesis")
    if (
        not isinstance(hypothesis, str)
        or not hypothesis.strip()
        or "\x00" in hypothesis
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "label-free hypothesis is invalid"
        )
    item_commitment = _require_sha256(
        raw.get("item_commitment_sha256"), "item commitment"
    )
    paper_commitment = _require_sha256(
        raw.get("paper_commitment_sha256"), "paper commitment"
    )
    component_commitment = _require_sha256(
        raw.get("component_commitment_sha256"), "component commitment"
    )
    raw_nodes = raw.get("nodes")
    if not isinstance(raw_nodes, list) or len(raw_nodes) != NODE_COUNT:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "label-free node count is invalid"
        )
    nodes: list[PrivateNode] = []
    previous_end = 0
    for node_i, raw_node in enumerate(raw_nodes):
        node = _parse_node(raw_node, node_i, previous_end)
        nodes.append(node)
        previous_end = node.end
    return LabelFreeItem(
        block=block,
        ordinal=expected_ordinal,
        item_commitment_sha256=item_commitment,
        paper_commitment_sha256=paper_commitment,
        component_commitment_sha256=component_commitment,
        hypothesis=hypothesis,
        nodes=tuple(nodes),
    )


def _load_label_free_block(path: Path, expected_block: str) -> LabelFreeBlock:
    if expected_block not in BLOCKS:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "unknown label-free block"
        )
    payload, file_hash = _read_private_json(
        path, f"{expected_block} label-free pack"
    )
    if set(payload) != LABEL_FREE_BLOCK_KEYS or (
        payload.get("schema") != LABEL_FREE_SCHEMA
        or payload.get("block") != expected_block
        or payload.get("count") != BLOCK_COUNT
        or not isinstance(payload.get("rows"), list)
        or len(payload["rows"]) != BLOCK_COUNT
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "label-free block schema drifted"
        )
    declared = _require_sha256(
        payload.get("block_sha256"), "label-free block hash"
    )
    if _block_hash(payload) != declared:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "label-free block self-hash drifted"
        )
    rows = tuple(
        _parse_view_item(raw, expected_block, ordinal)
        for ordinal, raw in enumerate(payload["rows"])
    )
    if len({row.item_commitment_sha256 for row in rows}) != BLOCK_COUNT:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "label-free item commitments are not unique"
        )
    if len({row.component_commitment_sha256 for row in rows}) != BLOCK_COUNT:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "label-free component commitments are not disjoint"
        )
    return LabelFreeBlock(expected_block, declared, file_hash, rows)


def _parse_aspect_buckets(raw: object) -> tuple[tuple[int, ...], ...]:
    if not isinstance(raw, list) or not raw:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "gold aspect list must be nonempty"
        )
    aspects: list[tuple[int, ...]] = []
    for bucket_set in raw:
        if (
            not isinstance(bucket_set, list)
            or not bucket_set
            or bucket_set != sorted(set(bucket_set))
            or any(
                type(index) is not int or not 0 <= index < NODE_COUNT
                for index in bucket_set
            )
        ):
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "gold aspect bucket set is invalid"
            )
        aspects.append(tuple(bucket_set))
    return tuple(aspects)


def _load_label_block(path: Path, expected_block: str) -> LabelBlock:
    if expected_block not in LABEL_BLOCKS:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "labels are forbidden for this block"
        )
    payload, file_hash = _read_private_json(
        path, f"{expected_block} label pack"
    )
    if set(payload) != LABEL_BLOCK_KEYS or (
        payload.get("schema") != LABEL_SCHEMA
        or payload.get("block") != expected_block
        or payload.get("count") != BLOCK_COUNT
        or not isinstance(payload.get("rows"), list)
        or len(payload["rows"]) != BLOCK_COUNT
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "label block schema drifted"
        )
    declared = _require_sha256(
        payload.get("block_sha256"), "label block hash"
    )
    if _block_hash(payload) != declared:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "label block self-hash drifted"
        )
    rows: list[LabelItem] = []
    for ordinal, raw in enumerate(payload["rows"]):
        if not isinstance(raw, Mapping) or set(raw) != LABEL_ITEM_KEYS or (
            raw.get("schema") != LABEL_ITEM_SCHEMA
            or raw.get("block") != expected_block
            or raw.get("ordinal") != ordinal
        ):
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "label item schema drifted"
            )
        rows.append(
            LabelItem(
                block=expected_block,
                ordinal=ordinal,
                item_commitment_sha256=_require_sha256(
                    raw.get("item_commitment_sha256"),
                    "label item commitment",
                ),
                gold_aspect_node_indices=_parse_aspect_buckets(
                    raw.get("gold_aspect_node_indices")
                ),
            )
        )
    if len({row.item_commitment_sha256 for row in rows}) != BLOCK_COUNT:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "label item commitments are not unique"
        )
    return LabelBlock(expected_block, declared, file_hash, tuple(rows))


def load_a_form_view(path: Path) -> LabelFreeBlock:
    return _load_label_free_block(path, "A_form")


def load_a_form_labels(path: Path) -> LabelBlock:
    return _load_label_block(path, "A_form")


def load_f_search_view(path: Path) -> LabelFreeBlock:
    return _load_label_free_block(path, "F_search")


def load_a_hold_view(path: Path) -> LabelFreeBlock:
    return _load_label_free_block(path, "A_hold")


def load_a_hold_labels(path: Path) -> LabelBlock:
    return _load_label_block(path, "A_hold")


def load_m_search_view(path: Path) -> LabelFreeBlock:
    return _load_label_free_block(path, "M_search")


def load_m_search_labels(path: Path) -> LabelBlock:
    return _load_label_block(path, "M_search")


def _join_labels(
    view: LabelFreeBlock, labels: LabelBlock
) -> tuple[tuple[LabelFreeItem, LabelItem], ...]:
    if view.block != labels.block or len(view.rows) != len(labels.rows):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "view and label blocks do not align"
        )
    joined: list[tuple[LabelFreeItem, LabelItem]] = []
    for item, label in zip(view.rows, labels.rows):
        if (
            item.ordinal != label.ordinal
            or item.item_commitment_sha256 != label.item_commitment_sha256
        ):
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "view and label item commitments drifted"
            )
        joined.append((item, label))
    return tuple(joined)


def _quantized_vector(left: np.ndarray, rows: np.ndarray) -> tuple[int, ...]:
    return tuple(quantized_cosine_similarity(left, row) for row in rows)


def _local_tensor_from_embeddings(
    item: LabelFreeItem,
    query_embedding: np.ndarray,
    node_embeddings: np.ndarray,
) -> LocalTensor:
    query_similarities = _quantized_vector(query_embedding, node_embeddings)
    node_matrix = tuple(
        tuple(
            quantized_cosine_similarity(left, right)
            for right in node_embeddings
        )
        for left in node_embeddings
    )
    raw = tuple(
        sorted(
            range(NODE_COUNT),
            key=lambda index: (-query_similarities[index], index),
        )[:TOP_K]
    )
    return LocalTensor(
        raw_top5=raw,  # type: ignore[arg-type]
        query_node_similarities=query_similarities,
        node_node_similarities=node_matrix,
        typed_edges=build_typed_scientific_graph(item.source_nodes),
    )


def precompute_local_block(
    block: LabelFreeBlock,
    encoder: EncoderProtocol,
    *,
    progress: ProgressHook = _noop_progress,
) -> tuple[LocalTensor, ...]:
    """Batch one block through the frozen offline encoder, then fan out locally."""

    if block.block not in BLOCKS or len(block.rows) != BLOCK_COUNT:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "local tensor block identity drifted"
        )
    flat_texts: list[str] = []
    slices: list[tuple[int, int]] = []
    for item in block.rows:
        start = len(flat_texts)
        flat_texts.append(item.hypothesis)
        flat_texts.extend(
            embedding_text(node.identity_text) for node in item.nodes
        )
        slices.append((start, len(flat_texts)))
    try:
        matrix = np.asarray(encoder.encode(tuple(flat_texts)), dtype=np.float32)
    except Exception as exc:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "offline MiniLM batch failed"
        ) from exc
    if (
        matrix.ndim != 2
        or matrix.shape != (len(flat_texts), ENCODER_DIMENSION)
        or not np.isfinite(matrix).all()
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "offline MiniLM batch shape drifted"
        )
    results: list[LocalTensor | None] = [None] * BLOCK_COUNT
    with ThreadPoolExecutor(max_workers=LOCAL_ITEM_CONCURRENCY_CAP) as pool:
        futures: dict[Future[LocalTensor], int] = {}
        for item, (start, end) in zip(block.rows, slices):
            futures[
                pool.submit(
                    _local_tensor_from_embeddings,
                    item,
                    matrix[start],
                    matrix[start + 1 : end],
                )
            ] = item.ordinal
        try:
            for future in as_completed(futures):
                ordinal = futures[future]
                results[ordinal] = future.result()
                progress("local_tensor_terminal", block.block, ordinal)
        except Exception:
            for future in futures:
                future.cancel()
            raise
    if any(result is None for result in results):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "local tensor completion barrier did not close"
        )
    return tuple(result for result in results if result is not None)


def _validated_official_top5(
    values: object, source_count: int
) -> tuple[int, int, int, int, int]:
    if isinstance(values, (str, bytes)):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "official top5 is malformed"
        )
    try:
        normalized = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "official top5 is malformed"
        ) from exc
    if (
        source_count != NODE_COUNT
        or len(normalized) != TOP_K
        or len(set(normalized)) != TOP_K
        or any(
            type(value) is not int or not 0 <= value < source_count
            for value in normalized
        )
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "official top5 violates the frozen contract"
        )
    return normalized  # type: ignore[return-value]


def _execute_full_item(
    item: LabelFreeItem,
    local: LocalTensor,
    official_top5: tuple[int, int, int, int, int],
) -> FullItemActions:
    table = build_common_candidate_table(
        item.source_nodes,
        local.typed_edges,
        official_top5,
        local.query_node_similarities,
    )
    traces = execute_all_recipes(
        official_top5, table, local.query_node_similarities
    )
    if tuple(trace.recipe_id for trace in traces) != RECIPE_IDS or len(
        {trace.common_scan_sha256 for trace in traces}
    ) != 1:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "nine-recipe common scan drifted"
        )
    trace_map = {trace.recipe_id: trace for trace in traces}
    components = {
        trace.recipe_id: coverage_components(
            item.hypothesis,
            item.source_nodes,
            trace.output_top5,
            official_top5,
            local.typed_edges,
            local.query_node_similarities,
            local.node_node_similarities,
        )
        for trace in traces
    }
    evaluator_table = {
        recipe_id: [
            [evaluator_id, score.numerator, score.denominator]
            for evaluator_id, score in score_all_evaluators(component)
        ]
        for recipe_id, component in components.items()
    }
    if any(
        len(rows) != len(EVALUATOR_IDS) for rows in evaluator_table.values()
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "sixteen-evaluator scan drifted"
        )
    return FullItemActions(
        block=item.block,
        ordinal=item.ordinal,
        raw_top5=local.raw_top5,
        official_top5=official_top5,
        traces_by_recipe=trace_map,
        components_by_recipe=components,
        evaluator_score_table_sha256=_semantic_hash(evaluator_table),
    )


def _execute_measurement_item(
    item: LabelFreeItem,
    local: LocalTensor,
    official_top5: tuple[int, int, int, int, int],
    selected_recipe_id: str,
) -> MeasurementItemActions:
    table = build_common_candidate_table(
        item.source_nodes,
        local.typed_edges,
        official_top5,
        local.query_node_similarities,
    )
    identity = execute_recipe(
        official_top5, table, local.query_node_similarities, R0
    )
    agent = execute_recipe(
        official_top5,
        table,
        local.query_node_similarities,
        selected_recipe_id,
    )
    if identity.output_top5 != official_top5 or (
        identity.common_scan_sha256 != agent.common_scan_sha256
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "paired R0/Agent common scan drifted"
        )
    return MeasurementItemActions(
        block=item.block,
        ordinal=item.ordinal,
        raw_top5=local.raw_top5,
        official_top5=official_top5,
        agent_top5=agent.output_top5,
        common_scan_sha256=agent.common_scan_sha256,
    )


def _runtime_binding_hash(runtime: OfficialRuntimeProtocol) -> str:
    safe = dict(runtime.safe_binding)
    rendered = json.dumps(
        safe, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    if any(
        forbidden in rendered
        for forbidden in ("/home/", "/tmp/", "\\", "identity_text", "gold_")
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "official safe binding contains a private or host field"
        )
    return _semantic_hash(safe)


def _run_official_action_wave(
    blocks: Sequence[LabelFreeBlock],
    tensors: Mapping[str, Sequence[LocalTensor]],
    runtime: OfficialRuntimeProtocol,
    work_root: Path,
    *,
    selected_recipe_id: str | None,
    progress: ProgressHook,
) -> tuple[
    dict[str, tuple[FullItemActions | MeasurementItemActions, ...]], str
]:
    """Run one shared cap-8 completion queue and close its action barrier."""

    if work_root.exists():
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "official work root already exists"
        )
    work_root.mkdir(parents=True, mode=0o700)
    official_parent = work_root / "official"
    official_parent.mkdir(mode=0o700)
    ordered: dict[
        str, list[FullItemActions | MeasurementItemActions | None]
    ] = {block.block: [None] * len(block.rows) for block in blocks}
    official_futures: dict[
        Future[tuple[int, ...]], tuple[LabelFreeItem, LocalTensor]
    ] = {}
    local_futures: dict[
        Future[FullItemActions | MeasurementItemActions], tuple[str, int]
    ] = {}
    with ThreadPoolExecutor(
        max_workers=OFFICIAL_CONCURRENCY_CAP
    ) as official_pool, ThreadPoolExecutor(
        max_workers=LOCAL_ITEM_CONCURRENCY_CAP
    ) as local_pool:
        for block in blocks:
            local_rows = tuple(tensors.get(block.block, ()))
            if len(local_rows) != len(block.rows):
                raise EvidenceBenchGraphEvaluatorRunnerError(
                    "local tensor block length drifted"
                )
            for item, local in zip(block.rows, local_rows):
                item_work = official_parent / f"{block.block}_{item.ordinal:03d}"
                future = official_pool.submit(
                    runtime.retrieve,
                    question=item.hypothesis,
                    paragraphs=item.paragraphs,
                    work_root=item_work,
                )
                official_futures[future] = (item, local)
        try:
            for future in as_completed(official_futures):
                item, local = official_futures[future]
                try:
                    raw_official = future.result()
                except Exception as exc:
                    raise EvidenceBenchGraphEvaluatorRunnerError(
                        "official runtime item failed"
                    ) from exc
                official = _validated_official_top5(
                    raw_official, len(item.nodes)
                )
                progress("official_terminal", item.block, item.ordinal)
                if selected_recipe_id is None:
                    local_future = local_pool.submit(
                        _execute_full_item, item, local, official
                    )
                else:
                    local_future = local_pool.submit(
                        _execute_measurement_item,
                        item,
                        local,
                        official,
                        selected_recipe_id,
                    )
                local_futures[local_future] = (item.block, item.ordinal)
            for future in as_completed(local_futures):
                block_id, ordinal = local_futures[future]
                ordered[block_id][ordinal] = future.result()
                progress("action_terminal", block_id, ordinal)
        except Exception:
            for future in (*official_futures, *local_futures):
                future.cancel()
            raise
    if any(any(row is None for row in rows) for rows in ordered.values()):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "action completion barrier did not close"
        )
    try:
        fresh = dict(runtime.fresh_reverify())
    except Exception as exc:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "official postflight failed"
        ) from exc
    if _semantic_hash(fresh) != _runtime_binding_hash(runtime):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "official postflight binding drifted"
        )
    progress("postflight_terminal", None, None)
    return (
        {
            block_id: tuple(row for row in rows if row is not None)
            for block_id, rows in ordered.items()
        },
        _semantic_hash(fresh),
    )


def _label_histograms(
    joined: Sequence[tuple[LabelFreeItem, LabelItem]],
) -> dict[str, dict[str, int]]:
    aspect_counts: Counter[int] = Counter()
    union_counts: Counter[int] = Counter()
    alternative_counts: Counter[int] = Counter()
    for _item, label in joined:
        aspects = label.gold_aspect_node_indices
        aspect_counts[len(aspects)] += 1
        union_counts[len(set().union(*(set(row) for row in aspects)))] += 1
        alternative_counts[sum(len(row) > 1 for row in aspects)] += 1
    return {
        "aspect_count_histogram": {
            str(key): aspect_counts[key] for key in sorted(aspect_counts)
        },
        "gold_bucket_union_count_histogram": {
            str(key): union_counts[key] for key in sorted(union_counts)
        },
        "aspects_with_alternative_evidence_histogram": {
            str(key): alternative_counts[key]
            for key in sorted(alternative_counts)
        },
    }


def _arm_aggregates(
    joined: Sequence[tuple[LabelFreeItem, LabelItem]],
    outputs: Mapping[str, Sequence[Sequence[int]]],
) -> tuple[dict[str, dict[str, int]], dict[str, tuple[int, ...]]]:
    aggregates: dict[str, dict[str, int]] = {}
    utilities: dict[str, tuple[int, ...]] = {}
    for arm, arm_outputs in outputs.items():
        if len(arm_outputs) != len(joined):
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "arm output count drifted"
            )
        covered_total = 0
        aspect_total = 0
        complete_total = 0
        utility_rows: list[int] = []
        for (item, label), top5 in zip(joined, arm_outputs):
            covered, complete, utility = item_utility(
                top5,
                label.gold_aspect_node_indices,
                source_count=len(item.nodes),
            )
            covered_total += covered
            aspect_total += len(label.gold_aspect_node_indices)
            complete_total += complete
            utility_rows.append(utility)
        aggregates[arm] = {
            "item_count": len(joined),
            "aspect_covered_count": covered_total,
            "aspect_total": aspect_total,
            "complete_count": complete_total,
            "total_U": sum(utility_rows),
        }
        utilities[arm] = tuple(utility_rows)
    return aggregates, utilities


def run_formation_wave(
    a_block: LabelFreeBlock,
    f_block: LabelFreeBlock,
    *,
    a_label_loader: Callable[[], LabelBlock],
    encoder: EncoderProtocol,
    runtime: OfficialRuntimeProtocol,
    work_root: Path,
    progress: ProgressHook = _noop_progress,
) -> FormationOutcome:
    """Execute A-form and F-search actions in one wave, then open only A labels."""

    if a_block.block != "A_form" or f_block.block != "F_search":
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "formation block identities drifted"
        )
    component_commitments = [
        item.component_commitment_sha256
        for block in (a_block, f_block)
        for item in block.rows
    ]
    if len(set(component_commitments)) != 2 * BLOCK_COUNT:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "A_form and F_search components are not disjoint"
        )
    a_tensors = precompute_local_block(a_block, encoder, progress=progress)
    f_tensors = precompute_local_block(f_block, encoder, progress=progress)
    action_map, runtime_hash = _run_official_action_wave(
        (a_block, f_block),
        {"A_form": a_tensors, "F_search": f_tensors},
        runtime,
        work_root,
        selected_recipe_id=None,
        progress=progress,
    )
    a_actions = tuple(action_map["A_form"])
    f_actions = tuple(action_map["F_search"])
    if any(
        not isinstance(row, FullItemActions) for row in (*a_actions, *f_actions)
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "formation action type drifted"
        )
    progress("labels_open", "A_form", None)
    try:
        a_labels = a_label_loader()
    except Exception as exc:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "A_form label loader failed"
        ) from exc
    joined = _join_labels(a_block, a_labels)
    formation_items: list[FormationItem] = []
    for (item, label), action in zip(joined, a_actions):
        assert isinstance(action, FullItemActions)
        utility_by_recipe: dict[str, int] = {}
        complete_by_recipe: dict[str, bool] = {}
        for recipe_id in RECIPE_IDS:
            _covered, complete, utility = item_utility(
                action.traces_by_recipe[recipe_id].output_top5,
                label.gold_aspect_node_indices,
                source_count=len(item.nodes),
            )
            utility_by_recipe[recipe_id] = utility
            complete_by_recipe[recipe_id] = bool(complete)
        formation_items.append(
            FormationItem(
                components_by_recipe=action.components_by_recipe,
                utility_by_recipe=utility_by_recipe,
                complete_by_recipe=complete_by_recipe,
            )
        )
    a_selection = select_a_evaluator(formation_items)
    f_selection = select_f_recipe(
        [
            action.components_by_recipe
            for action in f_actions
            if isinstance(action, FullItemActions)
        ],
        a_selection.evaluator_id,
    )
    selected_f_outputs = tuple(
        action.traces_by_recipe[f_selection.recipe_id].output_top5
        for action in f_actions
        if isinstance(action, FullItemActions)
    )
    official_f_outputs = tuple(
        action.official_top5
        for action in f_actions
        if isinstance(action, FullItemActions)
    )
    identifiable = has_identifiable_transition(
        f_selection.recipe_id, selected_f_outputs, official_f_outputs
    )
    a_outputs = {
        "canonical_RAW": tuple(
            action.raw_top5
            for action in a_actions
            if isinstance(action, FullItemActions)
        ),
        "official_HippoRAG": tuple(
            action.official_top5
            for action in a_actions
            if isinstance(action, FullItemActions)
        ),
        "Agent": tuple(
            action.traces_by_recipe[f_selection.recipe_id].output_top5
            for action in a_actions
            if isinstance(action, FullItemActions)
        ),
    }
    aggregates, _utilities = _arm_aggregates(joined, a_outputs)
    action_summary = [
        {
            "block": action.block,
            "ordinal": action.ordinal,
            "official": list(action.official_top5),
            "raw": list(action.raw_top5),
            "recipe_outputs": {
                recipe_id: list(
                    action.traces_by_recipe[recipe_id].output_top5
                )
                for recipe_id in RECIPE_IDS
            },
            "evaluator_score_table_sha256": (
                action.evaluator_score_table_sha256
            ),
        }
        for action in (*a_actions, *f_actions)
        if isinstance(action, FullItemActions)
    ]
    return FormationOutcome(
        a_block=a_block,
        f_block=f_block,
        a_labels=a_labels,
        a_selection=a_selection,
        f_selection=f_selection,
        identifiable_transition=identifiable,
        a_arm_aggregates=aggregates,
        a_label_histograms=_label_histograms(joined),
        action_table_sha256=_semantic_hash(action_summary),
        runtime_binding_sha256=runtime_hash,
    )


def _verified_exact_test(deltas: Sequence[int]) -> dict[str, object]:
    exact = dict(exact_magnitude_preserving_sign_flip(deltas))
    required = {
        "observed_net_U",
        "p_value_numerator",
        "p_value_denominator",
        "promoted",
    }
    if not required <= set(exact):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "exact sign-flip result schema drifted"
        )
    numerator = exact["p_value_numerator"]
    denominator = exact["p_value_denominator"]
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or numerator < 0
        or denominator <= 0
        or exact["observed_net_U"] != sum(deltas)
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "exact sign-flip result is invalid"
        )
    expected = sum(deltas) > 0 and Fraction(numerator, denominator) <= Fraction(
        1, 10
    )
    if type(exact["promoted"]) is not bool or exact["promoted"] != expected:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "exact promotion criterion drifted"
        )
    return exact


def run_measurement_wave(
    block: LabelFreeBlock,
    *,
    selected_recipe_id: str,
    label_loader: Callable[[], LabelBlock],
    encoder: EncoderProtocol,
    runtime: OfficialRuntimeProtocol,
    work_root: Path,
    selected_evaluator_id: str | None = None,
    progress: ProgressHook = _noop_progress,
) -> MeasurementOutcome:
    if block.block not in {"A_hold", "M_search"}:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "measurement block identity drifted"
        )
    if selected_recipe_id not in RECIPE_IDS or selected_recipe_id == R0:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "measurement Agent recipe is invalid"
        )
    if selected_evaluator_id is not None and (
        selected_evaluator_id not in EVALUATOR_IDS
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "measurement evaluator binding is invalid"
        )
    tensors = precompute_local_block(block, encoder, progress=progress)
    action_map, runtime_hash = _run_official_action_wave(
        (block,),
        {block.block: tensors},
        runtime,
        work_root,
        selected_recipe_id=selected_recipe_id,
        progress=progress,
    )
    actions = tuple(action_map[block.block])
    if any(not isinstance(row, MeasurementItemActions) for row in actions):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "measurement action type drifted"
        )
    progress("labels_open", block.block, None)
    try:
        labels = label_loader()
    except Exception as exc:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            f"{block.block} label loader failed"
        ) from exc
    joined = _join_labels(block, labels)
    outputs = {
        "canonical_RAW": tuple(
            row.raw_top5
            for row in actions
            if isinstance(row, MeasurementItemActions)
        ),
        "official_HippoRAG": tuple(
            row.official_top5
            for row in actions
            if isinstance(row, MeasurementItemActions)
        ),
        "Agent": tuple(
            row.agent_top5
            for row in actions
            if isinstance(row, MeasurementItemActions)
        ),
    }
    aggregates, utilities = _arm_aggregates(joined, outputs)
    deltas = tuple(
        agent - official
        for agent, official in zip(
            utilities["Agent"], utilities["official_HippoRAG"]
        )
    )
    exact = _verified_exact_test(deltas)
    action_summary = [
        {
            "block": row.block,
            "ordinal": row.ordinal,
            "raw": list(row.raw_top5),
            "official": list(row.official_top5),
            "agent": list(row.agent_top5),
            "common_scan_sha256": row.common_scan_sha256,
        }
        for row in actions
        if isinstance(row, MeasurementItemActions)
    ]
    return MeasurementOutcome(
        block=block,
        labels=labels,
        selected_evaluator_id=selected_evaluator_id,
        selected_recipe_id=selected_recipe_id,
        arm_aggregates=aggregates,
        label_histograms=_label_histograms(joined),
        delta_vector_sha256=_semantic_hash(list(deltas)),
        exact_test=exact,
        action_table_sha256=_semantic_hash(action_summary),
        runtime_binding_sha256=runtime_hash,
    )


def run_m_if_authorized(
    *,
    authorized: bool,
    view_loader: Callable[[], LabelFreeBlock],
    label_loader: Callable[[], LabelBlock],
    selected_recipe_id: str,
    encoder: EncoderProtocol,
    runtime: OfficialRuntimeProtocol,
    work_root: Path,
    selected_evaluator_id: str | None = None,
    progress: ProgressHook = _noop_progress,
) -> MeasurementOutcome:
    if authorized is not True:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "M_search is sealed because A_hold did not promote"
        )
    try:
        block = view_loader()
    except Exception as exc:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "M_search view loader failed"
        ) from exc
    if block.block != "M_search":
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "M_search loader returned the wrong block"
        )
    return run_measurement_wave(
        block,
        selected_recipe_id=selected_recipe_id,
        selected_evaluator_id=selected_evaluator_id,
        label_loader=label_loader,
        encoder=encoder,
        runtime=runtime,
        work_root=work_root,
        progress=progress,
    )


def _binding_receipt_fields(
    protocol_binding: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    if protocol_binding is None:
        return {
            "binding_status": "synthetic_test_unbound",
            "implementation_freeze_sha256": PENDING,
            "implementation_freeze_file_sha256": PENDING,
            "git_HEAD": PENDING,
            "git_verification": None,
            "design_sha256": PENDING,
            "design_file_sha256": PENDING,
            "custody_sha256": PENDING,
            "custody_file_sha256": PENDING,
            "source_access_sha256": PENDING,
            "source_access_file_sha256": PENDING,
            "graph_core_sha256": PENDING,
            "acquisition_runner_sha256": PENDING,
            "evaluator_runner_sha256": PENDING,
        }
    required = {
        "implementation_freeze_sha256",
        "implementation_freeze_file_sha256",
        "git_HEAD",
        "git_verification",
        "design_sha256",
        "design_file_sha256",
        "custody_sha256",
        "custody_file_sha256",
        "source_access_sha256",
        "source_access_file_sha256",
        "graph_core_sha256",
        "acquisition_runner_sha256",
        "evaluator_runner_sha256",
    }
    if not required <= set(protocol_binding):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "formal protocol receipt binding is incomplete"
        )
    return {
        "binding_status": "formally_verified_against_HEAD",
        **{key: protocol_binding[key] for key in sorted(required)},
    }


def _acquisition_receipt_fields(
    acquisition: AcquisitionReceiptBinding | None = None,
) -> dict[str, object]:
    if acquisition is None:
        return {
            "acquisition_receipt_relative_path": (
                CANONICAL_ACQUISITION_RECEIPT_RELATIVE_PATH.as_posix()
            ),
            "acquisition_receipt_sha256": PENDING,
            "acquisition_receipt_file_sha256": PENDING,
            "acquisition_receipt_git_blob_sha1": PENDING,
            "acquisition_receipt_verified_at_git_HEAD": PENDING,
        }
    return {
        "acquisition_receipt_relative_path": (
            CANONICAL_ACQUISITION_RECEIPT_RELATIVE_PATH.as_posix()
        ),
        "acquisition_receipt_sha256": acquisition.acquisition_sha256,
        "acquisition_receipt_file_sha256": acquisition.file_sha256,
        "acquisition_receipt_git_blob_sha1": acquisition.git_blob_sha1,
        "acquisition_receipt_verified_at_git_HEAD": (
            acquisition.verified_at_git_HEAD
        ),
    }


def _receipt_with_hash(body: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_sha256" in body:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "receipt body already contains a hash"
        )
    normalized = dict(body)
    return {**normalized, "receipt_sha256": _semantic_hash(normalized)}


def formation_public_receipt(
    outcome: FormationOutcome,
    *,
    protocol_binding: Mapping[str, Any] | None = None,
    acquisition: AcquisitionReceiptBinding | None = None,
) -> dict[str, Any]:
    status = (
        "formation_complete_identifiable"
        if outcome.identifiable_transition
        else "terminal_same_behavior_no_runner_up"
    )
    body: dict[str, Any] = {
        "schema": f"{VERSION}_formation_public_receipt",
        "version": VERSION,
        "stage": "formation",
        "status": status,
        **_binding_receipt_fields(protocol_binding),
        **_acquisition_receipt_fields(acquisition),
        "minilm_asset_sha256": MINILM_ASSET_SHA256,
        "minilm_asset_file_sha256": MINILM_ASSET_FILE_SHA256,
        "official_concurrency_cap": OFFICIAL_CONCURRENCY_CAP,
        "local_item_concurrency_cap": LOCAL_ITEM_CONCURRENCY_CAP,
        "A_form_count": len(outcome.a_block.rows),
        "F_search_count": len(outcome.f_block.rows),
        "A_form_view_file_sha256": outcome.a_block.file_sha256,
        "A_form_label_file_sha256": outcome.a_labels.file_sha256,
        "F_search_view_file_sha256": outcome.f_block.file_sha256,
        "F_search_labels_opened": False,
        "recipe_count": len(RECIPE_IDS),
        "evaluator_count": len(EVALUATOR_IDS),
        "selected_evaluator_id": outcome.a_selection.evaluator_id,
        "selected_recipe_id": outcome.f_selection.recipe_id,
        "identifiable_transition": outcome.identifiable_transition,
        "A_form_arm_aggregates": dict(outcome.a_arm_aggregates),
        "A_form_label_histograms": dict(outcome.a_label_histograms),
        "action_table_sha256": outcome.action_table_sha256,
        "runtime_binding_sha256": outcome.runtime_binding_sha256,
        "A_hold_authorized": outcome.identifiable_transition,
        "runner_up_or_fallback_attempted": False,
        "item_rows_persisted_publicly": False,
    }
    return _receipt_with_hash(body)


def measurement_public_receipt(
    outcome: MeasurementOutcome,
    *,
    protocol_binding: Mapping[str, Any] | None = None,
    acquisition: AcquisitionReceiptBinding | None = None,
) -> dict[str, Any]:
    promoted = bool(outcome.exact_test.get("promoted"))
    if outcome.block.block == "A_hold":
        status = "promoted" if promoted else "valid_nonpromotion"
    else:
        status = "terminal_positive" if promoted else "terminal_negative"
    body: dict[str, Any] = {
        "schema": f"{VERSION}_{outcome.block.block}_public_receipt",
        "version": VERSION,
        "stage": outcome.block.block,
        "status": status,
        **_binding_receipt_fields(protocol_binding),
        **_acquisition_receipt_fields(acquisition),
        "minilm_asset_sha256": MINILM_ASSET_SHA256,
        "minilm_asset_file_sha256": MINILM_ASSET_FILE_SHA256,
        "official_concurrency_cap": OFFICIAL_CONCURRENCY_CAP,
        "local_item_concurrency_cap": LOCAL_ITEM_CONCURRENCY_CAP,
        "item_count": len(outcome.block.rows),
        "view_file_sha256": outcome.block.file_sha256,
        "label_file_sha256": outcome.labels.file_sha256,
        "selected_evaluator_id": outcome.selected_evaluator_id,
        "selected_recipe_id": outcome.selected_recipe_id,
        "arm_aggregates": dict(outcome.arm_aggregates),
        "label_histograms": dict(outcome.label_histograms),
        "delta_vector_sha256": outcome.delta_vector_sha256,
        "exact_test": dict(outcome.exact_test),
        "action_table_sha256": outcome.action_table_sha256,
        "runtime_binding_sha256": outcome.runtime_binding_sha256,
        "M_search_authorized": (
            promoted if outcome.block.block == "A_hold" else False
        ),
        "item_rows_persisted_publicly": False,
    }
    return _receipt_with_hash(body)


def _safe_failure_receipt(
    stage: str,
    failure_class: str,
    marker_sha256: str,
    protocol_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if stage not in {"formation", "A_hold", "M_search"}:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "failure stage is invalid"
        )
    allowed = {
        "private_pack_invalid",
        "offline_embedding_invalid",
        "official_runtime_invalid",
        "typed_action_invalid",
        "label_join_invalid",
        "receipt_persistence_invalid",
        "unexpected_internal_invalid",
    }
    if failure_class not in allowed:
        failure_class = "unexpected_internal_invalid"
    return _receipt_with_hash(
        {
            "schema": f"{VERSION}_stage_failure_receipt",
            "version": VERSION,
            "stage": stage,
            "status": "terminal_infrastructure_invalid_no_replay",
            "failure_class": failure_class,
            "marker_sha256": _require_sha256(
                marker_sha256, "failure marker hash"
            ),
            **_binding_receipt_fields(protocol_binding),
            "private_path_exception_message_or_item_persisted": False,
        }
    )


def _write_exclusive(
    path: Path, payload: Mapping[str, Any], mode: int
) -> str:
    absolute = _assert_no_symlink_components(path, "output path")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_bytes(payload) + b"\n"
    try:
        descriptor = os.open(
            absolute,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            mode,
        )
    except FileExistsError as exc:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "exclusive output already exists; replay is forbidden"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    parent_descriptor = os.open(
        absolute.parent, os.O_RDONLY | os.O_DIRECTORY
    )
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return _sha256_bytes(raw)


def consume_stage_marker(
    path: Path,
    stage: str,
    *,
    protocol_binding: Mapping[str, Any] | None = None,
) -> str:
    if stage not in {"formation", "A_hold", "M_search"}:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "stage marker identity is invalid"
        )
    marker = {
        "schema": f"{VERSION}_one_shot_stage_marker",
        "version": VERSION,
        "stage": stage,
        "design_sha256": (
            PENDING
            if protocol_binding is None
            else protocol_binding["design_sha256"]
        ),
        "implementation_freeze_sha256": (
            PENDING
            if protocol_binding is None
            else protocol_binding["implementation_freeze_sha256"]
        ),
        "git_HEAD": (
            PENDING if protocol_binding is None else protocol_binding["git_HEAD"]
        ),
        "replay_allowed": False,
        "parent_process_stage_entry": True,
    }
    return _write_exclusive(path, marker, 0o600)


def _classify_failure(exc: BaseException) -> str:
    message = str(exc).casefold()
    if "pack" in message or "block" in message or "private" in message:
        return "private_pack_invalid"
    if "minilm" in message or "embedding" in message or "tensor" in message:
        return "offline_embedding_invalid"
    if "official" in message or "hippo" in message or "postflight" in message:
        return "official_runtime_invalid"
    if (
        "label" in message
        or "gold" in message
        or "aspect" in message
        or "join" in message
    ):
        return "label_join_invalid"
    if (
        "recipe" in message
        or "action" in message
        or "coverage" in message
        or "sign-flip" in message
    ):
        return "typed_action_invalid"
    return "unexpected_internal_invalid"


def _persist_stage_failure(
    path: Path,
    stage: str,
    exc: BaseException,
    marker_hash: str,
    protocol_binding: Mapping[str, Any] | None = None,
) -> None:
    receipt = _safe_failure_receipt(
        stage, _classify_failure(exc), marker_hash, protocol_binding
    )
    _write_exclusive(path, receipt, 0o644)


def _load_canonical_prior_stage_receipt(
    *,
    project_root: Path,
    stage: str,
    protocol_binding: Mapping[str, Any],
    acquisition: AcquisitionReceiptBinding,
    formation: PriorStageReceiptBinding | None = None,
) -> PriorStageReceiptBinding:
    if stage not in {"formation", "A_hold"}:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "prior-stage receipt identity is invalid"
        )
    relative = CANONICAL_STAGE_PATHS[stage]["receipt"]
    payload, raw, file_sha, git_blob, _audit = (
        _read_committed_public_prerequisite(
            project_root=project_root,
            relative_path=relative,
            expected_current_head=str(protocol_binding["git_HEAD"]),
        )
    )
    expected_schema = f"{VERSION}_{stage}_public_receipt"
    if payload.get("schema") != expected_schema or payload.get("stage") != stage:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "canonical prior-stage receipt schema drifted"
        )
    body = dict(payload)
    declared = _require_sha256(
        body.pop("receipt_sha256", None), "prior-stage receipt hash"
    )
    if _semantic_hash(body) != declared:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "canonical prior-stage receipt self-hash drifted"
        )
    current_fields = _binding_receipt_fields(protocol_binding)
    for field in (
        "binding_status",
        "implementation_freeze_sha256",
        "implementation_freeze_file_sha256",
        "design_sha256",
        "design_file_sha256",
        "custody_sha256",
        "custody_file_sha256",
        "source_access_sha256",
        "source_access_file_sha256",
        "graph_core_sha256",
        "acquisition_runner_sha256",
        "evaluator_runner_sha256",
    ):
        if payload.get(field) != current_fields[field]:
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "prior-stage implementation chain drifted"
            )
    historical_head = payload.get("git_HEAD")
    if SHA1_RE.fullmatch(str(historical_head)) is None:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "prior-stage historical Git HEAD is malformed"
        )
    acquisition_fields = _acquisition_receipt_fields(acquisition)
    for field in (
        "acquisition_receipt_relative_path",
        "acquisition_receipt_sha256",
        "acquisition_receipt_file_sha256",
        "acquisition_receipt_git_blob_sha1",
    ):
        if payload.get(field) != acquisition_fields[field]:
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "prior-stage acquisition chain drifted"
            )
    if SHA1_RE.fullmatch(
        str(payload.get("acquisition_receipt_verified_at_git_HEAD"))
    ) is None:
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "prior-stage acquisition verification HEAD is malformed"
        )
    selected_recipe = payload.get("selected_recipe_id")
    selected_evaluator = payload.get("selected_evaluator_id")
    if (
        selected_recipe not in RECIPE_IDS
        or selected_recipe == R0
        or selected_evaluator not in EVALUATOR_IDS
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "prior-stage selected action binding is invalid"
        )
    if stage == "formation":
        if (
            payload.get("status") != "formation_complete_identifiable"
            or payload.get("A_hold_authorized") is not True
            or payload.get("F_search_labels_opened") is not False
            or payload.get("identifiable_transition") is not True
            or payload.get("runner_up_or_fallback_attempted") is not False
        ):
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "formation receipt does not authorize A_hold"
            )
    else:
        exact = payload.get("exact_test")
        exact_valid = False
        if isinstance(exact, Mapping):
            numerator = exact.get("p_value_numerator")
            denominator = exact.get("p_value_denominator")
            observed = exact.get("observed_net_U")
            exact_valid = (
                exact.get("promoted") is True
                and type(observed) is int
                and observed > 0
                and type(numerator) is int
                and type(denominator) is int
                and numerator >= 0
                and denominator > 0
                and Fraction(numerator, denominator) <= Fraction(1, 10)
            )
        if formation is None or (
            payload.get("status") != "promoted"
            or payload.get("M_search_authorized") is not True
            or not exact_valid
            or selected_recipe
            != formation.payload.get("selected_recipe_id")
            or selected_evaluator
            != formation.payload.get("selected_evaluator_id")
            or payload.get("formation_receipt_relative_path")
            != CANONICAL_STAGE_PATHS["formation"]["receipt"].as_posix()
            or payload.get("formation_receipt_sha256")
            != formation.receipt_sha256
            or payload.get("formation_receipt_file_sha256")
            != formation.file_sha256
            or payload.get("formation_receipt_git_blob_sha1")
            != formation.git_blob_sha1
            or SHA1_RE.fullmatch(
                str(payload.get("formation_receipt_verified_at_git_HEAD"))
            )
            is None
        ):
            raise EvidenceBenchGraphEvaluatorRunnerError(
                "A_hold receipt promotion or formation chain drifted"
            )
    return PriorStageReceiptBinding(
        stage=stage,
        receipt_sha256=declared,
        file_sha256=file_sha,
        git_blob_sha1=git_blob,
        verified_at_git_HEAD=str(protocol_binding["git_HEAD"]),
        payload=payload,
    )


def execute_formation_stage(
    *,
    project_root: Path,
    encoder: EncoderProtocol,
    runtime: OfficialRuntimeProtocol,
    protocol_binding: Mapping[str, Any] | None = None,
    progress: ProgressHook = _noop_progress,
) -> dict[str, Any]:
    root = _canonical_project_root(project_root)
    if protocol_binding is None:
        protocol_binding = verify_design_binding(root)
    acquisition = _load_canonical_acquisition_receipt(
        project_root=root, protocol_binding=protocol_binding
    )
    paths = _canonical_stage_absolutes(root, "formation")
    _preflight_canonical_stage_outputs(paths)
    marker_hash = consume_stage_marker(
        paths["root"] / "formation.attempt.marker",
        "formation",
        protocol_binding=protocol_binding,
    )
    try:
        a_view_relative, a_label_relative = CANONICAL_PRIVATE_PACKS["A_form"]
        f_view_relative = CANONICAL_PRIVATE_PACKS["F_search"][0]
        a_block = load_a_form_view(root / a_view_relative)
        _verify_loaded_pack_commitment(
            acquisition=acquisition,
            block="A_form",
            view=a_block,
            labels=None,
        )
        f_block = load_f_search_view(root / f_view_relative)
        _verify_loaded_pack_commitment(
            acquisition=acquisition,
            block="F_search",
            view=f_block,
            labels=None,
        )

        def load_bound_a_labels() -> LabelBlock:
            labels = load_a_form_labels(root / a_label_relative)
            _verify_loaded_pack_commitment(
                acquisition=acquisition,
                block="A_form",
                view=a_block,
                labels=labels,
            )
            return labels

        outcome = run_formation_wave(
            a_block,
            f_block,
            a_label_loader=load_bound_a_labels,
            encoder=encoder,
            runtime=runtime,
            work_root=paths["root"] / "formation.work",
            progress=progress,
        )
        receipt = formation_public_receipt(
            outcome,
            protocol_binding=protocol_binding,
            acquisition=acquisition,
        )
        _write_exclusive(paths["receipt"], receipt, 0o644)
        return receipt
    except Exception as exc:
        _persist_stage_failure(
            paths["failure"],
            "formation",
            exc,
            marker_hash,
            protocol_binding,
        )
        raise


def execute_a_hold_stage(
    *,
    project_root: Path,
    encoder: EncoderProtocol,
    runtime: OfficialRuntimeProtocol,
    protocol_binding: Mapping[str, Any] | None = None,
    progress: ProgressHook = _noop_progress,
) -> dict[str, Any]:
    root = _canonical_project_root(project_root)
    if protocol_binding is None:
        protocol_binding = verify_design_binding(root)
    acquisition = _load_canonical_acquisition_receipt(
        project_root=root, protocol_binding=protocol_binding
    )
    formation = _load_canonical_prior_stage_receipt(
        project_root=root,
        stage="formation",
        protocol_binding=protocol_binding,
        acquisition=acquisition,
    )
    selected_recipe = str(formation.payload["selected_recipe_id"])
    selected_evaluator = str(formation.payload["selected_evaluator_id"])
    paths = _canonical_stage_absolutes(root, "A_hold")
    _preflight_canonical_stage_outputs(paths)
    marker_hash = consume_stage_marker(
        paths["root"] / "A_hold.attempt.marker",
        "A_hold",
        protocol_binding=protocol_binding,
    )
    try:
        view_relative, label_relative = CANONICAL_PRIVATE_PACKS["A_hold"]
        block = load_a_hold_view(root / view_relative)
        _verify_loaded_pack_commitment(
            acquisition=acquisition,
            block="A_hold",
            view=block,
            labels=None,
        )

        def load_bound_a_hold_labels() -> LabelBlock:
            labels = load_a_hold_labels(root / label_relative)
            _verify_loaded_pack_commitment(
                acquisition=acquisition,
                block="A_hold",
                view=block,
                labels=labels,
            )
            return labels

        outcome = run_measurement_wave(
            block,
            selected_recipe_id=selected_recipe,
            selected_evaluator_id=selected_evaluator,
            label_loader=load_bound_a_hold_labels,
            encoder=encoder,
            runtime=runtime,
            work_root=paths["root"] / "A_hold.work",
            progress=progress,
        )
        receipt = measurement_public_receipt(
            outcome,
            protocol_binding=protocol_binding,
            acquisition=acquisition,
        )
        receipt.update(
            {
                "formation_receipt_relative_path": CANONICAL_STAGE_PATHS[
                    "formation"
                ]["receipt"].as_posix(),
                "formation_receipt_sha256": formation.receipt_sha256,
                "formation_receipt_file_sha256": formation.file_sha256,
                "formation_receipt_git_blob_sha1": formation.git_blob_sha1,
                "formation_receipt_verified_at_git_HEAD": (
                    formation.verified_at_git_HEAD
                ),
            }
        )
        body = dict(receipt)
        body.pop("receipt_sha256")
        receipt["receipt_sha256"] = _semantic_hash(body)
        _write_exclusive(paths["receipt"], receipt, 0o644)
        return receipt
    except Exception as exc:
        _persist_stage_failure(
            paths["failure"],
            "A_hold",
            exc,
            marker_hash,
            protocol_binding,
        )
        raise


def execute_m_search_stage(
    *,
    project_root: Path,
    encoder: EncoderProtocol,
    runtime: OfficialRuntimeProtocol,
    protocol_binding: Mapping[str, Any] | None = None,
    progress: ProgressHook = _noop_progress,
) -> dict[str, Any]:
    root = _canonical_project_root(project_root)
    if protocol_binding is None:
        protocol_binding = verify_design_binding(root)
    acquisition = _load_canonical_acquisition_receipt(
        project_root=root, protocol_binding=protocol_binding
    )
    formation = _load_canonical_prior_stage_receipt(
        project_root=root,
        stage="formation",
        protocol_binding=protocol_binding,
        acquisition=acquisition,
    )
    anchor = _load_canonical_prior_stage_receipt(
        project_root=root,
        stage="A_hold",
        protocol_binding=protocol_binding,
        acquisition=acquisition,
        formation=formation,
    )
    selected_recipe = str(anchor.payload["selected_recipe_id"])
    selected_evaluator = str(anchor.payload["selected_evaluator_id"])
    paths = _canonical_stage_absolutes(root, "M_search")
    _preflight_canonical_stage_outputs(paths)
    marker_hash = consume_stage_marker(
        paths["root"] / "M_search.attempt.marker",
        "M_search",
        protocol_binding=protocol_binding,
    )
    try:
        view_relative, label_relative = CANONICAL_PRIVATE_PACKS["M_search"]
        loaded_view: dict[str, LabelFreeBlock] = {}

        def load_bound_m_view() -> LabelFreeBlock:
            block = load_m_search_view(root / view_relative)
            _verify_loaded_pack_commitment(
                acquisition=acquisition,
                block="M_search",
                view=block,
                labels=None,
            )
            loaded_view["block"] = block
            return block

        def load_bound_m_labels() -> LabelBlock:
            block = loaded_view.get("block")
            if block is None:
                raise EvidenceBenchGraphEvaluatorRunnerError(
                    "M_search view was not opened before labels"
                )
            labels = load_m_search_labels(root / label_relative)
            _verify_loaded_pack_commitment(
                acquisition=acquisition,
                block="M_search",
                view=block,
                labels=labels,
            )
            return labels

        outcome = run_m_if_authorized(
            authorized=True,
            view_loader=load_bound_m_view,
            label_loader=load_bound_m_labels,
            selected_recipe_id=selected_recipe,
            selected_evaluator_id=selected_evaluator,
            encoder=encoder,
            runtime=runtime,
            work_root=paths["root"] / "M_search.work",
            progress=progress,
        )
        receipt = measurement_public_receipt(
            outcome,
            protocol_binding=protocol_binding,
            acquisition=acquisition,
        )
        receipt.update(
            {
                "A_hold_receipt_relative_path": CANONICAL_STAGE_PATHS[
                    "A_hold"
                ]["receipt"].as_posix(),
                "A_hold_receipt_sha256": anchor.receipt_sha256,
                "A_hold_receipt_file_sha256": anchor.file_sha256,
                "A_hold_receipt_git_blob_sha1": anchor.git_blob_sha1,
                "A_hold_receipt_verified_at_git_HEAD": (
                    anchor.verified_at_git_HEAD
                ),
                "formation_receipt_relative_path": CANONICAL_STAGE_PATHS[
                    "formation"
                ]["receipt"].as_posix(),
                "formation_receipt_sha256": formation.receipt_sha256,
                "formation_receipt_file_sha256": formation.file_sha256,
                "formation_receipt_git_blob_sha1": formation.git_blob_sha1,
                "formation_receipt_verified_at_git_HEAD": (
                    formation.verified_at_git_HEAD
                ),
            }
        )
        body = dict(receipt)
        body.pop("receipt_sha256")
        receipt["receipt_sha256"] = _semantic_hash(body)
        _write_exclusive(paths["receipt"], receipt, 0o644)
        return receipt
    except Exception as exc:
        _persist_stage_failure(
            paths["failure"],
            "M_search",
            exc,
            marker_hash,
            protocol_binding,
        )
        raise


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--runtime-python", required=True, type=Path)
    parser.add_argument("--local-llm-model", required=True, type=Path)
    parser.add_argument("--local-embedding-model", required=True, type=Path)
    parser.add_argument("--base-binding-receipt", required=True, type=Path)
    parser.add_argument("--attestation-receipt", required=True, type=Path)
    parser.add_argument("--minilm-manifest", required=True, type=Path)
    parser.add_argument("--minilm-model-root", required=True, type=Path)


def _prepare_resources(
    arguments: argparse.Namespace,
) -> tuple[OfflineMiniLMEncoder, PreparedFormalRuntimeV2]:
    # The prepared official adapter may launch its already-attested bwrap
    # subprocess.  This runner launches no source, label, or evaluator child.
    runtime = prepare_formal_runtime_v2(
        project_root=arguments.project_root,
        attestation_receipt_path=arguments.attestation_receipt,
        base_binding_receipt_path=arguments.base_binding_receipt,
        runtime_python=arguments.runtime_python,
        local_llm_model=arguments.local_llm_model,
        local_embedding_model=arguments.local_embedding_model,
    )
    encoder = OfflineMiniLMEncoder(
        asset_manifest_path=arguments.minilm_manifest,
        model_root=arguments.minilm_model_root,
        run_canary=True,
    )
    return encoder, runtime


def _safe_cli_result(receipt: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            {
                "receipt_sha256": receipt["receipt_sha256"],
                "stage": receipt["stage"],
                "status": receipt["status"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def _require_protocol_binding_unchanged(
    before_resource_preparation: Mapping[str, Any],
    after_resource_preparation: Mapping[str, Any],
) -> None:
    if _canonical_bytes(before_resource_preparation) != _canonical_bytes(
        after_resource_preparation
    ):
        raise EvidenceBenchGraphEvaluatorRunnerError(
            "formal protocol binding changed during resource preparation"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    formation = subparsers.add_parser("formation")
    _add_runtime_arguments(formation)

    a_hold = subparsers.add_parser("a-hold")
    _add_runtime_arguments(a_hold)

    m_search = subparsers.add_parser("m-search")
    _add_runtime_arguments(m_search)

    arguments = parser.parse_args(argv)
    # Bind loaded code before model/runtime construction, then require the
    # complete binding (including HEAD) to remain identical immediately before
    # the canonical stage entrypoint consumes its marker.
    initial_binding = verify_design_binding(arguments.project_root)
    encoder, runtime = _prepare_resources(arguments)
    fresh_binding = verify_design_binding(arguments.project_root)
    _require_protocol_binding_unchanged(initial_binding, fresh_binding)
    if arguments.command == "formation":
        receipt = execute_formation_stage(
            project_root=arguments.project_root,
            encoder=encoder,
            runtime=runtime,
            protocol_binding=fresh_binding,
        )
    elif arguments.command == "a-hold":
        receipt = execute_a_hold_stage(
            project_root=arguments.project_root,
            encoder=encoder,
            runtime=runtime,
            protocol_binding=fresh_binding,
        )
    else:
        receipt = execute_m_search_stage(
            project_root=arguments.project_root,
            encoder=encoder,
            runtime=runtime,
            protocol_binding=fresh_binding,
        )
    _safe_cli_result(receipt)
    return 0


__all__ = [
    "BLOCK_COUNT",
    "EVALUATOR_IDS",
    "EvidenceBenchGraphEvaluatorRunnerError",
    "EXPECTED_BINDING_INTERFACES",
    "FormationOutcome",
    "LABEL_FREE_ITEM_SCHEMA",
    "LABEL_FREE_SCHEMA",
    "LABEL_ITEM_SCHEMA",
    "LABEL_SCHEMA",
    "LabelBlock",
    "LabelFreeBlock",
    "LabelFreeItem",
    "LabelItem",
    "MeasurementOutcome",
    "NODE_COUNT",
    "OFFICIAL_CONCURRENCY_CAP",
    "PENDING",
    "IMPLEMENTATION_FREEZE_RELATIVE_PATH",
    "IMPLEMENTATION_FREEZE_SCHEMA",
    "PrivateNode",
    "RECIPE_IDS",
    "R0",
    "VERSION",
    "consume_stage_marker",
    "execute_a_hold_stage",
    "execute_formation_stage",
    "execute_m_search_stage",
    "formation_public_receipt",
    "load_a_form_labels",
    "load_a_form_view",
    "load_a_hold_labels",
    "load_a_hold_view",
    "load_f_search_view",
    "load_m_search_labels",
    "load_m_search_view",
    "main",
    "measurement_public_receipt",
    "precompute_local_block",
    "run_formation_wave",
    "run_m_if_authorized",
    "run_measurement_wave",
    "verify_design_binding",
]


if __name__ == "__main__":
    raise SystemExit(main())
