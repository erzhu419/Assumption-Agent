"""Frozen, synthetic-testable CUAD direct graph-evaluator stage runner.

The runner never knows how to locate or parse the CUAD source archive.  It
accepts only acquisition-produced, stage-specific private packs.  Labels have
separate loader types and are opened only after every required retrieval,
typed action, and official-runtime postflight has reached a terminal state.

Formal stage entrypoints execute in their invoking parent process.  The only
permitted child processes are those hidden behind the already-attested
official HippoRAG v2 adapter.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from assumption_agent.models import stable_hash
from replication_runtime.qasper_minilm_v1 import (
    OfflineMiniLMEncoder,
    quantized_cosine_similarity,
)

from .contractnli_typed_clause_graph_v1 import (
    AFormationSelection,
    ActionTrace,
    CoverageComponents,
    FSearchSelection,
    FormationItem,
    SourceSpan,
    TypedEdge,
    build_common_candidate_table,
    build_typed_clause_graph,
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


VERSION = "cuad_graph_evaluator_runner_v1"
DESIGN_RELATIVE_PATH = Path("manifests/cuad_graph_evaluator_design_v1.json")
DESIGN_SHA256 = "2a651230838f51ca615fbf93cfc902800f0d1debfb184b8f1b552d4fc6893a15"
DESIGN_FILE_SHA256 = "3c85a6949d18408013e2e8e9da0f140b16da434e63a7a053924532525163052c"
GRAPH_CORE_RELATIVE_PATH = Path(
    "assumption_agent/benchmarks/contractnli_typed_clause_graph_v1.py"
)
GRAPH_CORE_SHA256 = "7aef388172c08eecd227033111ce0e92845bca0b514a8bacbff205566963460c"
GRAPH_CORE_COMMIT = "4237cbd034b7edf4ec6fc34d6f1f9ea89cc89109"
MINILM_ASSET_SHA256 = "921d9b1945581130e03c53f448092c3de3b30714431c6cac9b3b32c2ec10abad"
MINILM_ASSET_FILE_SHA256 = (
    "62b85c7752f2e46932fb9fb13ae2f3aac9eb750a33c8f07102739040feb6cc75"
)

LABEL_FREE_SCHEMA = "cuad_direct_v1_label_free_block"
LABEL_FREE_ITEM_SCHEMA = "cuad_direct_v1_label_free_item"
LABEL_SCHEMA = "cuad_direct_v1_label_block"
LABEL_ITEM_SCHEMA = "cuad_direct_v1_label_item"
BLOCKS = ("A_form", "F_search", "A_hold", "M_search")
BLOCK_COUNT = 64
TOP_K = 5
MIN_NODES = 5
MAX_NODES = 128
MAX_NODE_CHARACTERS = 1200
MAX_PRIVATE_BYTES = 64 * 1024 * 1024
OFFICIAL_CONCURRENCY_CAP = 8
LOCAL_ITEM_CONCURRENCY_CAP = 64
RECIPE_IDS = tuple(recipe.recipe_id for recipe in recipe_registry())
EVALUATOR_IDS = tuple(evaluator.evaluator_id for evaluator in evaluator_registry())
R0 = "R0_HIPPO_TOP5"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

LABEL_FREE_BLOCK_KEYS = frozenset({"schema", "block", "count", "rows", "block_sha256"})
LABEL_FREE_ITEM_KEYS = frozenset(
    {
        "schema",
        "block",
        "ordinal",
        "item_commitment_sha256",
        "component_commitment_sha256",
        "question",
        "title",
        "nodes",
    }
)
NODE_KEYS = frozenset({"span_i", "start", "end", "identity_text"})
LABEL_BLOCK_KEYS = frozenset({"schema", "block", "count", "rows", "block_sha256"})
LABEL_ITEM_KEYS = frozenset(
    {
        "schema",
        "block",
        "ordinal",
        "item_commitment_sha256",
        "gold_node_indices",
    }
)


class CuadGraphEvaluatorRunnerError(RuntimeError):
    """Raised when a frozen runner, pack, stage, or receipt contract drifts."""


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

    def source_span(self) -> SourceSpan:
        return SourceSpan(self.span_i, self.start, self.end, self.identity_text)


@dataclass(frozen=True)
class LabelFreeItem:
    block: str
    ordinal: int
    item_commitment_sha256: str
    component_commitment_sha256: str
    question: str
    nodes: tuple[PrivateNode, ...]

    @property
    def spans(self) -> tuple[SourceSpan, ...]:
        return tuple(node.source_span() for node in self.nodes)

    @property
    def paragraphs(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "idx": node.span_i,
                "title": "CUAD_contract",
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
    gold_node_indices: tuple[int, ...]


@dataclass(frozen=True)
class LabelBlock:
    block: str
    block_sha256: str
    file_sha256: str
    rows: tuple[LabelItem, ...]


@dataclass(frozen=True)
class LocalTensor:
    raw_top5: tuple[int, int, int, int, int]
    query_span_similarities: tuple[int, ...]
    span_span_similarities: tuple[tuple[int, ...], ...]
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
    action_table_sha256: str
    runtime_binding_sha256: str


@dataclass(frozen=True)
class MeasurementOutcome:
    block: LabelFreeBlock
    labels: LabelBlock
    selected_recipe_id: str
    arm_aggregates: Mapping[str, Mapping[str, int]]
    delta_vector_sha256: str
    exact_test: Mapping[str, object]
    action_table_sha256: str
    runtime_binding_sha256: str


def _noop_progress(_event: str, _block: str | None, _ordinal: int | None) -> None:
    return None


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
        raise CuadGraphEvaluatorRunnerError(f"{field} must be lowercase sha256")
    return value


def _assert_no_symlink_components(path: Path, field: str) -> Path:
    absolute = path.expanduser().absolute()
    for candidate in (*reversed(absolute.parents), absolute):
        if candidate.is_symlink():
            raise CuadGraphEvaluatorRunnerError(f"{field} contains a symbolic link")
    return absolute


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
        raise CuadGraphEvaluatorRunnerError("value is not canonical JSON") from exc


def _block_hash(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("block_sha256", None)
    return _sha256_bytes(_canonical_bytes(body))


def verify_design_binding(project_root: Path) -> dict[str, Any]:
    root = _assert_no_symlink_components(project_root.resolve(strict=True), "project root")
    design_path = _assert_no_symlink_components(root / DESIGN_RELATIVE_PATH, "design path")
    core_path = _assert_no_symlink_components(root / GRAPH_CORE_RELATIVE_PATH, "graph core path")
    if not design_path.is_file() or _sha256_file(design_path) != DESIGN_FILE_SHA256:
        raise CuadGraphEvaluatorRunnerError("CUAD design file drifted")
    if not core_path.is_file() or _sha256_file(core_path) != GRAPH_CORE_SHA256:
        raise CuadGraphEvaluatorRunnerError("frozen typed graph core drifted")
    try:
        design = json.loads(design_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CuadGraphEvaluatorRunnerError("CUAD design is unreadable") from exc
    if not isinstance(design, dict):
        raise CuadGraphEvaluatorRunnerError("CUAD design root drifted")
    body = dict(design)
    declared = body.pop("design_sha256", None)
    if declared != DESIGN_SHA256 or _sha256_bytes(_canonical_bytes(body)) != declared:
        raise CuadGraphEvaluatorRunnerError("CUAD design self-hash drifted")
    binding = design.get("graph_core_binding")
    if not isinstance(binding, Mapping) or (
        binding.get("source_commit") != GRAPH_CORE_COMMIT
        or binding.get("source_file_sha256") != GRAPH_CORE_SHA256
        or binding.get("source_relative_path") != GRAPH_CORE_RELATIVE_PATH.as_posix()
    ):
        raise CuadGraphEvaluatorRunnerError("CUAD graph-core design binding drifted")
    return design


def _read_private_json(path: Path, field: str) -> tuple[dict[str, Any], str]:
    absolute = _assert_no_symlink_components(path, field)
    if not absolute.is_file():
        raise CuadGraphEvaluatorRunnerError(f"{field} is unavailable")
    info = absolute.stat()
    if stat.S_IMODE(info.st_mode) != 0o600 or not 1 <= info.st_size <= MAX_PRIVATE_BYTES:
        raise CuadGraphEvaluatorRunnerError(f"{field} mode or size is invalid")
    raw = absolute.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CuadGraphEvaluatorRunnerError(f"{field} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise CuadGraphEvaluatorRunnerError(f"{field} root must be an object")
    return payload, _sha256_bytes(raw)


def _parse_node(raw: object, expected_i: int, previous_end: int) -> PrivateNode:
    if not isinstance(raw, Mapping) or set(raw) != NODE_KEYS:
        raise CuadGraphEvaluatorRunnerError("private node schema drifted")
    span_i = raw.get("span_i")
    start = raw.get("start")
    end = raw.get("end")
    text = raw.get("identity_text")
    if (
        type(span_i) is not int
        or span_i != expected_i
        or type(start) is not int
        or type(end) is not int
        or start < previous_end
        or end <= start
        or not isinstance(text, str)
        or not text
        or "\x00" in text
        or len(text) > MAX_NODE_CHARACTERS
        or end - start != len(text)
    ):
        raise CuadGraphEvaluatorRunnerError("private node content is invalid")
    return PrivateNode(span_i, start, end, text)


def _parse_view_item(raw: object, block: str, expected_ordinal: int) -> LabelFreeItem:
    if not isinstance(raw, Mapping) or set(raw) != LABEL_FREE_ITEM_KEYS:
        raise CuadGraphEvaluatorRunnerError("label-free item schema drifted")
    if (
        raw.get("schema") != LABEL_FREE_ITEM_SCHEMA
        or raw.get("block") != block
        or type(raw.get("ordinal")) is not int
        or raw.get("ordinal") != expected_ordinal
        or raw.get("title") != "CUAD_contract"
    ):
        raise CuadGraphEvaluatorRunnerError("label-free item identity drifted")
    question = raw.get("question")
    if not isinstance(question, str) or not question.strip() or "\x00" in question:
        raise CuadGraphEvaluatorRunnerError("label-free question is invalid")
    item_commitment = _require_sha256(raw.get("item_commitment_sha256"), "item commitment")
    component_commitment = _require_sha256(
        raw.get("component_commitment_sha256"), "component commitment"
    )
    raw_nodes = raw.get("nodes")
    if not isinstance(raw_nodes, list) or not MIN_NODES <= len(raw_nodes) <= MAX_NODES:
        raise CuadGraphEvaluatorRunnerError("label-free node count is invalid")
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
        component_commitment_sha256=component_commitment,
        question=question,
        nodes=tuple(nodes),
    )


def _load_label_free_block(path: Path, expected_block: str) -> LabelFreeBlock:
    if expected_block not in BLOCKS:
        raise CuadGraphEvaluatorRunnerError("unknown label-free block")
    payload, file_hash = _read_private_json(path, f"{expected_block} label-free pack")
    if set(payload) != LABEL_FREE_BLOCK_KEYS or (
        payload.get("schema") != LABEL_FREE_SCHEMA
        or payload.get("block") != expected_block
        or payload.get("count") != BLOCK_COUNT
        or not isinstance(payload.get("rows"), list)
        or len(payload["rows"]) != BLOCK_COUNT
    ):
        raise CuadGraphEvaluatorRunnerError("label-free block schema drifted")
    declared = _require_sha256(payload.get("block_sha256"), "label-free block hash")
    if _block_hash(payload) != declared:
        raise CuadGraphEvaluatorRunnerError("label-free block self-hash drifted")
    rows = tuple(
        _parse_view_item(raw, expected_block, ordinal)
        for ordinal, raw in enumerate(payload["rows"])
    )
    if len({row.item_commitment_sha256 for row in rows}) != BLOCK_COUNT or len(
        {row.component_commitment_sha256 for row in rows}
    ) != BLOCK_COUNT:
        raise CuadGraphEvaluatorRunnerError("label-free block is not component unique")
    return LabelFreeBlock(expected_block, declared, file_hash, rows)


def _load_label_block(path: Path, expected_block: str) -> LabelBlock:
    if expected_block not in {"A_form", "A_hold", "M_search"}:
        raise CuadGraphEvaluatorRunnerError("labels are forbidden for this block")
    payload, file_hash = _read_private_json(path, f"{expected_block} label pack")
    if set(payload) != LABEL_BLOCK_KEYS or (
        payload.get("schema") != LABEL_SCHEMA
        or payload.get("block") != expected_block
        or payload.get("count") != BLOCK_COUNT
        or not isinstance(payload.get("rows"), list)
        or len(payload["rows"]) != BLOCK_COUNT
    ):
        raise CuadGraphEvaluatorRunnerError("label block schema drifted")
    declared = _require_sha256(payload.get("block_sha256"), "label block hash")
    if _block_hash(payload) != declared:
        raise CuadGraphEvaluatorRunnerError("label block self-hash drifted")
    rows: list[LabelItem] = []
    for ordinal, raw in enumerate(payload["rows"]):
        if not isinstance(raw, Mapping) or set(raw) != LABEL_ITEM_KEYS or (
            raw.get("schema") != LABEL_ITEM_SCHEMA
            or raw.get("block") != expected_block
            or raw.get("ordinal") != ordinal
        ):
            raise CuadGraphEvaluatorRunnerError("label item schema drifted")
        commitment = _require_sha256(raw.get("item_commitment_sha256"), "label item commitment")
        gold = raw.get("gold_node_indices")
        if (
            not isinstance(gold, list)
            or not 1 <= len(gold) <= TOP_K
            or any(type(value) is not int or value < 0 for value in gold)
            or gold != sorted(set(gold))
        ):
            raise CuadGraphEvaluatorRunnerError("gold-node envelope is invalid")
        rows.append(LabelItem(expected_block, ordinal, commitment, tuple(gold)))
    if len({row.item_commitment_sha256 for row in rows}) != BLOCK_COUNT:
        raise CuadGraphEvaluatorRunnerError("label block item commitments are not unique")
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


def _join_labels(view: LabelFreeBlock, labels: LabelBlock) -> tuple[tuple[LabelFreeItem, LabelItem], ...]:
    if view.block != labels.block or len(view.rows) != len(labels.rows):
        raise CuadGraphEvaluatorRunnerError("view and label blocks do not align")
    joined: list[tuple[LabelFreeItem, LabelItem]] = []
    for item, label in zip(view.rows, labels.rows):
        if (
            item.ordinal != label.ordinal
            or item.item_commitment_sha256 != label.item_commitment_sha256
            or any(index >= len(item.nodes) for index in label.gold_node_indices)
        ):
            raise CuadGraphEvaluatorRunnerError("view and label item commitments drifted")
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
    span_matrix = tuple(
        tuple(quantized_cosine_similarity(left, right) for right in node_embeddings)
        for left in node_embeddings
    )
    raw = tuple(
        sorted(range(len(item.nodes)), key=lambda index: (-query_similarities[index], index))[:TOP_K]
    )
    edges = build_typed_clause_graph(item.spans)
    return LocalTensor(
        raw_top5=raw,  # type: ignore[arg-type]
        query_span_similarities=query_similarities,
        span_span_similarities=span_matrix,
        typed_edges=edges,
    )


def precompute_local_block(
    block: LabelFreeBlock,
    encoder: EncoderProtocol,
    *,
    progress: ProgressHook = _noop_progress,
) -> tuple[LocalTensor, ...]:
    """Encode one 64-item block in one batch, then build 64 logical tensors."""

    flat_texts: list[str] = []
    slices: list[tuple[int, int]] = []
    for item in block.rows:
        start = len(flat_texts)
        flat_texts.append(item.question)
        flat_texts.extend(embedding_text(node.identity_text) for node in item.nodes)
        slices.append((start, len(flat_texts)))
    try:
        matrix = np.asarray(encoder.encode(tuple(flat_texts)), dtype=np.float32)
    except Exception as exc:
        raise CuadGraphEvaluatorRunnerError("offline MiniLM batch failed") from exc
    if matrix.ndim != 2 or matrix.shape != (len(flat_texts), 384) or not np.isfinite(matrix).all():
        raise CuadGraphEvaluatorRunnerError("offline MiniLM batch shape drifted")
    results: list[LocalTensor | None] = [None] * len(block.rows)
    with ThreadPoolExecutor(max_workers=min(LOCAL_ITEM_CONCURRENCY_CAP, len(block.rows))) as pool:
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
        for future in as_completed(futures):
            ordinal = futures[future]
            results[ordinal] = future.result()
            progress("local_tensor_terminal", block.block, ordinal)
    if any(result is None for result in results):
        raise CuadGraphEvaluatorRunnerError("local tensor wave did not close")
    return tuple(result for result in results if result is not None)


def _validated_official_top5(values: object, source_count: int) -> tuple[int, int, int, int, int]:
    if isinstance(values, (str, bytes)):
        raise CuadGraphEvaluatorRunnerError("official top5 is malformed")
    try:
        normalized = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise CuadGraphEvaluatorRunnerError("official top5 is malformed") from exc
    if (
        len(normalized) != TOP_K
        or len(set(normalized)) != TOP_K
        or any(type(value) is not int or not 0 <= value < source_count for value in normalized)
    ):
        raise CuadGraphEvaluatorRunnerError("official top5 violates the frozen contract")
    return normalized  # type: ignore[return-value]


def _execute_full_item(
    item: LabelFreeItem,
    local: LocalTensor,
    official_top5: tuple[int, int, int, int, int],
) -> FullItemActions:
    table = build_common_candidate_table(
        item.spans,
        local.typed_edges,
        official_top5,
        local.query_span_similarities,
    )
    traces = execute_all_recipes(official_top5, table, local.query_span_similarities)
    if tuple(trace.recipe_id for trace in traces) != RECIPE_IDS or len(
        {trace.common_scan_sha256 for trace in traces}
    ) != 1:
        raise CuadGraphEvaluatorRunnerError("nine-recipe common scan drifted")
    trace_map = {trace.recipe_id: trace for trace in traces}
    components = {
        trace.recipe_id: coverage_components(
            item.question,
            item.spans,
            trace.output_top5,
            official_top5,
            local.typed_edges,
            local.query_span_similarities,
            local.span_span_similarities,
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
    if any(len(rows) != len(EVALUATOR_IDS) for rows in evaluator_table.values()):
        raise CuadGraphEvaluatorRunnerError("sixteen-evaluator scan drifted")
    return FullItemActions(
        block=item.block,
        ordinal=item.ordinal,
        raw_top5=local.raw_top5,
        official_top5=official_top5,
        traces_by_recipe=trace_map,
        components_by_recipe=components,
        evaluator_score_table_sha256=stable_hash(evaluator_table),
    )


def _execute_measurement_item(
    item: LabelFreeItem,
    local: LocalTensor,
    official_top5: tuple[int, int, int, int, int],
    selected_recipe_id: str,
) -> MeasurementItemActions:
    table = build_common_candidate_table(
        item.spans,
        local.typed_edges,
        official_top5,
        local.query_span_similarities,
    )
    identity = execute_recipe(official_top5, table, local.query_span_similarities, R0)
    agent = execute_recipe(
        official_top5, table, local.query_span_similarities, selected_recipe_id
    )
    if identity.output_top5 != official_top5 or (
        identity.common_scan_sha256 != agent.common_scan_sha256
    ):
        raise CuadGraphEvaluatorRunnerError("paired R0/Agent common scan drifted")
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
    raw = json.dumps(safe, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if "/home/" in raw or "/tmp/" in raw or "\\" in raw:
        raise CuadGraphEvaluatorRunnerError("official safe binding leaks a host path")
    return stable_hash(safe)


def _run_official_action_wave(
    blocks: Sequence[LabelFreeBlock],
    tensors: Mapping[str, Sequence[LocalTensor]],
    runtime: OfficialRuntimeProtocol,
    work_root: Path,
    *,
    selected_recipe_id: str | None,
    progress: ProgressHook,
) -> tuple[dict[str, tuple[FullItemActions | MeasurementItemActions, ...]], str]:
    if work_root.exists():
        raise CuadGraphEvaluatorRunnerError("official work root already exists")
    work_root.mkdir(parents=True, mode=0o700)
    official_parent = work_root / "official"
    official_parent.mkdir(mode=0o700)
    ordered: dict[str, list[FullItemActions | MeasurementItemActions | None]] = {
        block.block: [None] * len(block.rows) for block in blocks
    }
    official_futures: dict[Future[tuple[int, ...]], tuple[LabelFreeItem, LocalTensor]] = {}
    local_futures: dict[
        Future[FullItemActions | MeasurementItemActions], tuple[str, int]
    ] = {}
    with ThreadPoolExecutor(max_workers=OFFICIAL_CONCURRENCY_CAP) as official_pool, ThreadPoolExecutor(
        max_workers=LOCAL_ITEM_CONCURRENCY_CAP
    ) as local_pool:
        for block in blocks:
            local_rows = tuple(tensors[block.block])
            if len(local_rows) != len(block.rows):
                raise CuadGraphEvaluatorRunnerError("local tensor block length drifted")
            for item, local in zip(block.rows, local_rows):
                item_work = official_parent / f"{block.block}_{item.ordinal:03d}"
                future = official_pool.submit(
                    runtime.retrieve,
                    question=item.question,
                    paragraphs=item.paragraphs,
                    work_root=item_work,
                )
                official_futures[future] = (item, local)
        try:
            for future in as_completed(official_futures):
                item, local = official_futures[future]
                official = _validated_official_top5(future.result(), len(item.nodes))
                progress("official_terminal", item.block, item.ordinal)
                if selected_recipe_id is None:
                    local_future: Future[FullItemActions | MeasurementItemActions] = local_pool.submit(
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
        raise CuadGraphEvaluatorRunnerError("action completion barrier did not close")
    try:
        fresh = dict(runtime.fresh_reverify())
    except Exception as exc:
        raise CuadGraphEvaluatorRunnerError("official postflight failed") from exc
    if stable_hash(fresh) != _runtime_binding_hash(runtime):
        raise CuadGraphEvaluatorRunnerError("official postflight binding drifted")
    progress("postflight_terminal", None, None)
    return (
        {
            block_id: tuple(row for row in rows if row is not None)
            for block_id, rows in ordered.items()
        },
        stable_hash(fresh),
    )


def _arm_aggregates(
    joined: Sequence[tuple[LabelFreeItem, LabelItem]],
    outputs: Mapping[str, Sequence[Sequence[int]]],
) -> tuple[dict[str, dict[str, int]], dict[str, tuple[int, ...]]]:
    aggregates: dict[str, dict[str, int]] = {}
    utilities: dict[str, tuple[int, ...]] = {}
    for arm, arm_outputs in outputs.items():
        if len(arm_outputs) != len(joined):
            raise CuadGraphEvaluatorRunnerError("arm output count drifted")
        hits_total = 0
        complete_total = 0
        utility_rows: list[int] = []
        support_total = 0
        for (item, label), top5 in zip(joined, arm_outputs):
            hits, complete, utility = item_utility(
                top5, label.gold_node_indices, source_count=len(item.nodes)
            )
            hits_total += hits
            complete_total += complete
            utility_rows.append(utility)
            support_total += len(label.gold_node_indices)
        aggregates[arm] = {
            "item_count": len(joined),
            "support_hit_count": hits_total,
            "support_total": support_total,
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
    """Precompute A/F label-free actions together; open only A labels afterward."""

    if a_block.block != "A_form" or f_block.block != "F_search":
        raise CuadGraphEvaluatorRunnerError("formation block identities drifted")
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
    if any(not isinstance(row, FullItemActions) for row in (*a_actions, *f_actions)):
        raise CuadGraphEvaluatorRunnerError("formation action type drifted")
    progress("labels_open", "A_form", None)
    a_labels = a_label_loader()
    joined = _join_labels(a_block, a_labels)
    formation_items: list[FormationItem] = []
    for (item, label), action in zip(joined, a_actions):
        assert isinstance(action, FullItemActions)
        utility_by_recipe: dict[str, int] = {}
        complete_by_recipe: dict[str, bool] = {}
        for recipe_id in RECIPE_IDS:
            _hits, complete, utility = item_utility(
                action.traces_by_recipe[recipe_id].output_top5,
                label.gold_node_indices,
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
        action.official_top5 for action in f_actions if isinstance(action, FullItemActions)
    )
    identifiable = has_identifiable_transition(
        f_selection.recipe_id, selected_f_outputs, official_f_outputs
    )
    a_outputs = {
        "canonical_RAW": tuple(
            action.raw_top5 for action in a_actions if isinstance(action, FullItemActions)
        ),
        "official_HippoRAG": tuple(
            action.official_top5 for action in a_actions if isinstance(action, FullItemActions)
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
                recipe_id: list(action.traces_by_recipe[recipe_id].output_top5)
                for recipe_id in RECIPE_IDS
            },
            "evaluator_score_table_sha256": action.evaluator_score_table_sha256,
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
        action_table_sha256=stable_hash(action_summary),
        runtime_binding_sha256=runtime_hash,
    )


def run_measurement_wave(
    block: LabelFreeBlock,
    *,
    selected_recipe_id: str,
    label_loader: Callable[[], LabelBlock],
    encoder: EncoderProtocol,
    runtime: OfficialRuntimeProtocol,
    work_root: Path,
    progress: ProgressHook = _noop_progress,
) -> MeasurementOutcome:
    if block.block not in {"A_hold", "M_search"}:
        raise CuadGraphEvaluatorRunnerError("measurement block identity drifted")
    if selected_recipe_id not in RECIPE_IDS or selected_recipe_id == R0:
        raise CuadGraphEvaluatorRunnerError("measurement Agent recipe is invalid")
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
        raise CuadGraphEvaluatorRunnerError("measurement action type drifted")
    progress("labels_open", block.block, None)
    labels = label_loader()
    joined = _join_labels(block, labels)
    outputs = {
        "canonical_RAW": tuple(
            row.raw_top5 for row in actions if isinstance(row, MeasurementItemActions)
        ),
        "official_HippoRAG": tuple(
            row.official_top5 for row in actions if isinstance(row, MeasurementItemActions)
        ),
        "Agent": tuple(
            row.agent_top5 for row in actions if isinstance(row, MeasurementItemActions)
        ),
    }
    aggregates, utilities = _arm_aggregates(joined, outputs)
    deltas = tuple(
        agent - official
        for agent, official in zip(utilities["Agent"], utilities["official_HippoRAG"])
    )
    exact = exact_magnitude_preserving_sign_flip(deltas)
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
        selected_recipe_id=selected_recipe_id,
        arm_aggregates=aggregates,
        delta_vector_sha256=stable_hash(list(deltas)),
        exact_test=exact,
        action_table_sha256=stable_hash(action_summary),
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
    progress: ProgressHook = _noop_progress,
) -> MeasurementOutcome:
    if authorized is not True:
        raise CuadGraphEvaluatorRunnerError("M_search is sealed because A_hold did not promote")
    block = view_loader()
    if block.block != "M_search":
        raise CuadGraphEvaluatorRunnerError("M_search loader returned the wrong block")
    return run_measurement_wave(
        block,
        selected_recipe_id=selected_recipe_id,
        label_loader=label_loader,
        encoder=encoder,
        runtime=runtime,
        work_root=work_root,
        progress=progress,
    )


def _receipt_with_hash(body: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt_sha256" in body:
        raise CuadGraphEvaluatorRunnerError("receipt body already contains a hash")
    return {**dict(body), "receipt_sha256": stable_hash(dict(body))}


def formation_public_receipt(outcome: FormationOutcome) -> dict[str, Any]:
    status = (
        "formation_complete_identifiable"
        if outcome.identifiable_transition
        else "terminal_unidentifiable_transition"
    )
    body: dict[str, Any] = {
        "schema": f"{VERSION}_formation_public_receipt",
        "version": VERSION,
        "stage": "formation",
        "status": status,
        "design_sha256": DESIGN_SHA256,
        "design_file_sha256": DESIGN_FILE_SHA256,
        "graph_core_sha256": GRAPH_CORE_SHA256,
        "minilm_asset_sha256": MINILM_ASSET_SHA256,
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
        "action_table_sha256": outcome.action_table_sha256,
        "runtime_binding_sha256": outcome.runtime_binding_sha256,
        "A_hold_authorized": outcome.identifiable_transition,
        "item_rows_persisted_publicly": False,
    }
    return _receipt_with_hash(body)


def measurement_public_receipt(outcome: MeasurementOutcome) -> dict[str, Any]:
    promoted = bool(outcome.exact_test.get("promoted"))
    if outcome.block.block == "M_search":
        status = "terminal_positive" if promoted else "terminal_negative"
    else:
        status = "promoted" if promoted else "valid_nonpromotion"
    body: dict[str, Any] = {
        "schema": f"{VERSION}_{outcome.block.block}_public_receipt",
        "version": VERSION,
        "stage": outcome.block.block,
        "status": status,
        "design_sha256": DESIGN_SHA256,
        "design_file_sha256": DESIGN_FILE_SHA256,
        "graph_core_sha256": GRAPH_CORE_SHA256,
        "minilm_asset_sha256": MINILM_ASSET_SHA256,
        "official_concurrency_cap": OFFICIAL_CONCURRENCY_CAP,
        "local_item_concurrency_cap": LOCAL_ITEM_CONCURRENCY_CAP,
        "item_count": len(outcome.block.rows),
        "view_file_sha256": outcome.block.file_sha256,
        "label_file_sha256": outcome.labels.file_sha256,
        "selected_recipe_id": outcome.selected_recipe_id,
        "arm_aggregates": dict(outcome.arm_aggregates),
        "delta_vector_sha256": outcome.delta_vector_sha256,
        "exact_test": dict(outcome.exact_test),
        "action_table_sha256": outcome.action_table_sha256,
        "runtime_binding_sha256": outcome.runtime_binding_sha256,
        "M_search_authorized": promoted if outcome.block.block == "A_hold" else False,
        "item_rows_persisted_publicly": False,
    }
    return _receipt_with_hash(body)


def _safe_failure_receipt(stage: str, failure_class: str, marker_sha256: str) -> dict[str, Any]:
    if stage not in {"formation", "A_hold", "M_search"}:
        raise CuadGraphEvaluatorRunnerError("failure stage is invalid")
    if failure_class not in {
        "private_pack_invalid",
        "offline_embedding_invalid",
        "official_runtime_invalid",
        "typed_action_invalid",
        "label_join_invalid",
        "receipt_persistence_invalid",
        "unexpected_internal_invalid",
    }:
        failure_class = "unexpected_internal_invalid"
    return _receipt_with_hash(
        {
            "schema": f"{VERSION}_stage_failure_receipt",
            "version": VERSION,
            "stage": stage,
            "status": "terminal_infrastructure_invalid_no_replay",
            "failure_class": failure_class,
            "marker_sha256": _require_sha256(marker_sha256, "failure marker hash"),
            "design_sha256": DESIGN_SHA256,
            "private_path_exception_message_or_item_persisted": False,
        }
    )


def _write_exclusive(path: Path, payload: Mapping[str, Any], mode: int) -> str:
    absolute = _assert_no_symlink_components(path, "output path")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_bytes(payload) + b"\n"
    descriptor = os.open(absolute, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
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
    parent_descriptor = os.open(absolute.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return _sha256_bytes(raw)


def consume_stage_marker(path: Path, stage: str) -> str:
    marker = {
        "schema": f"{VERSION}_one_shot_stage_marker",
        "version": VERSION,
        "stage": stage,
        "design_sha256": DESIGN_SHA256,
        "replay_allowed": False,
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
    if "label" in message or "gold" in message or "join" in message:
        return "label_join_invalid"
    if "recipe" in message or "action" in message or "coverage" in message:
        return "typed_action_invalid"
    return "unexpected_internal_invalid"


def _persist_stage_failure(path: Path, stage: str, exc: BaseException, marker_hash: str) -> None:
    receipt = _safe_failure_receipt(stage, _classify_failure(exc), marker_hash)
    _write_exclusive(path, receipt, 0o644)


def execute_formation_stage(
    *,
    project_root: Path,
    a_view_path: Path,
    a_label_path: Path,
    f_view_path: Path,
    stage_root: Path,
    receipt_path: Path,
    failure_receipt_path: Path,
    encoder: EncoderProtocol,
    runtime: OfficialRuntimeProtocol,
    progress: ProgressHook = _noop_progress,
) -> dict[str, Any]:
    verify_design_binding(project_root)
    marker_hash = consume_stage_marker(stage_root / "formation.attempt.marker", "formation")
    try:
        a_block = load_a_form_view(a_view_path)
        f_block = load_f_search_view(f_view_path)
        outcome = run_formation_wave(
            a_block,
            f_block,
            a_label_loader=lambda: load_a_form_labels(a_label_path),
            encoder=encoder,
            runtime=runtime,
            work_root=stage_root / "formation.work",
            progress=progress,
        )
        receipt = formation_public_receipt(outcome)
        _write_exclusive(receipt_path, receipt, 0o644)
        return receipt
    except Exception as exc:
        _persist_stage_failure(failure_receipt_path, "formation", exc, marker_hash)
        raise


def _load_public_receipt(path: Path, expected_schema: str) -> dict[str, Any]:
    absolute = _assert_no_symlink_components(path, "public receipt")
    if not absolute.is_file() or absolute.stat().st_size > 1024 * 1024:
        raise CuadGraphEvaluatorRunnerError("public receipt is unavailable")
    try:
        payload = json.loads(absolute.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CuadGraphEvaluatorRunnerError("public receipt is invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema") != expected_schema:
        raise CuadGraphEvaluatorRunnerError("public receipt schema drifted")
    body = dict(payload)
    declared = _require_sha256(body.pop("receipt_sha256", None), "public receipt hash")
    if stable_hash(body) != declared or payload.get("design_sha256") != DESIGN_SHA256:
        raise CuadGraphEvaluatorRunnerError("public receipt self-hash drifted")
    return payload


def execute_a_hold_stage(
    *,
    project_root: Path,
    formation_receipt_path: Path,
    view_path: Path,
    label_path: Path,
    stage_root: Path,
    receipt_path: Path,
    failure_receipt_path: Path,
    encoder: EncoderProtocol,
    runtime: OfficialRuntimeProtocol,
    progress: ProgressHook = _noop_progress,
) -> dict[str, Any]:
    verify_design_binding(project_root)
    formation = _load_public_receipt(
        formation_receipt_path, f"{VERSION}_formation_public_receipt"
    )
    if formation.get("status") != "formation_complete_identifiable" or (
        formation.get("A_hold_authorized") is not True
    ):
        raise CuadGraphEvaluatorRunnerError("A_hold is not authorized")
    selected = formation.get("selected_recipe_id")
    if not isinstance(selected, str):
        raise CuadGraphEvaluatorRunnerError("formation recipe binding is missing")
    marker_hash = consume_stage_marker(stage_root / "A_hold.attempt.marker", "A_hold")
    try:
        block = load_a_hold_view(view_path)
        outcome = run_measurement_wave(
            block,
            selected_recipe_id=selected,
            label_loader=lambda: load_a_hold_labels(label_path),
            encoder=encoder,
            runtime=runtime,
            work_root=stage_root / "A_hold.work",
            progress=progress,
        )
        receipt = measurement_public_receipt(outcome)
        receipt["formation_receipt_file_sha256"] = _sha256_file(formation_receipt_path)
        body = dict(receipt)
        body.pop("receipt_sha256")
        receipt["receipt_sha256"] = stable_hash(body)
        _write_exclusive(receipt_path, receipt, 0o644)
        return receipt
    except Exception as exc:
        _persist_stage_failure(failure_receipt_path, "A_hold", exc, marker_hash)
        raise


def execute_m_search_stage(
    *,
    project_root: Path,
    a_hold_receipt_path: Path,
    view_path: Path,
    label_path: Path,
    stage_root: Path,
    receipt_path: Path,
    failure_receipt_path: Path,
    encoder: EncoderProtocol,
    runtime: OfficialRuntimeProtocol,
    progress: ProgressHook = _noop_progress,
) -> dict[str, Any]:
    verify_design_binding(project_root)
    anchor = _load_public_receipt(
        a_hold_receipt_path, f"{VERSION}_A_hold_public_receipt"
    )
    if anchor.get("status") != "promoted" or anchor.get("M_search_authorized") is not True:
        raise CuadGraphEvaluatorRunnerError("M_search is not authorized")
    selected = anchor.get("selected_recipe_id")
    if not isinstance(selected, str):
        raise CuadGraphEvaluatorRunnerError("A_hold recipe binding is missing")
    marker_hash = consume_stage_marker(stage_root / "M_search.attempt.marker", "M_search")
    try:
        outcome = run_m_if_authorized(
            authorized=True,
            view_loader=lambda: load_m_search_view(view_path),
            label_loader=lambda: load_m_search_labels(label_path),
            selected_recipe_id=selected,
            encoder=encoder,
            runtime=runtime,
            work_root=stage_root / "M_search.work",
            progress=progress,
        )
        receipt = measurement_public_receipt(outcome)
        receipt["A_hold_receipt_file_sha256"] = _sha256_file(a_hold_receipt_path)
        body = dict(receipt)
        body.pop("receipt_sha256")
        receipt["receipt_sha256"] = stable_hash(body)
        _write_exclusive(receipt_path, receipt, 0o644)
        return receipt
    except Exception as exc:
        _persist_stage_failure(failure_receipt_path, "M_search", exc, marker_hash)
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
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--failure-receipt", required=True, type=Path)


def _prepare_resources(arguments: argparse.Namespace) -> tuple[OfflineMiniLMEncoder, PreparedFormalRuntimeV2]:
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
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    formation = subparsers.add_parser("formation")
    _add_runtime_arguments(formation)
    formation.add_argument("--a-view", required=True, type=Path)
    formation.add_argument("--a-labels", required=True, type=Path)
    formation.add_argument("--f-view", required=True, type=Path)

    a_hold = subparsers.add_parser("a-hold")
    _add_runtime_arguments(a_hold)
    a_hold.add_argument("--formation-receipt", required=True, type=Path)
    a_hold.add_argument("--view", required=True, type=Path)
    a_hold.add_argument("--labels", required=True, type=Path)

    m_search = subparsers.add_parser("m-search")
    _add_runtime_arguments(m_search)
    m_search.add_argument("--a-hold-receipt", required=True, type=Path)
    m_search.add_argument("--view", required=True, type=Path)
    m_search.add_argument("--labels", required=True, type=Path)

    arguments = parser.parse_args(argv)
    encoder, runtime = _prepare_resources(arguments)
    if arguments.command == "formation":
        receipt = execute_formation_stage(
            project_root=arguments.project_root,
            a_view_path=arguments.a_view,
            a_label_path=arguments.a_labels,
            f_view_path=arguments.f_view,
            stage_root=arguments.stage_root,
            receipt_path=arguments.receipt,
            failure_receipt_path=arguments.failure_receipt,
            encoder=encoder,
            runtime=runtime,
        )
    elif arguments.command == "a-hold":
        receipt = execute_a_hold_stage(
            project_root=arguments.project_root,
            formation_receipt_path=arguments.formation_receipt,
            view_path=arguments.view,
            label_path=arguments.labels,
            stage_root=arguments.stage_root,
            receipt_path=arguments.receipt,
            failure_receipt_path=arguments.failure_receipt,
            encoder=encoder,
            runtime=runtime,
        )
    else:
        receipt = execute_m_search_stage(
            project_root=arguments.project_root,
            a_hold_receipt_path=arguments.a_hold_receipt,
            view_path=arguments.view,
            label_path=arguments.labels,
            stage_root=arguments.stage_root,
            receipt_path=arguments.receipt,
            failure_receipt_path=arguments.failure_receipt,
            encoder=encoder,
            runtime=runtime,
        )
    _safe_cli_result(receipt)
    return 0


__all__ = [
    "BLOCK_COUNT",
    "CuadGraphEvaluatorRunnerError",
    "DESIGN_FILE_SHA256",
    "DESIGN_SHA256",
    "FormationOutcome",
    "LabelBlock",
    "LabelFreeBlock",
    "MeasurementOutcome",
    "OFFICIAL_CONCURRENCY_CAP",
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
