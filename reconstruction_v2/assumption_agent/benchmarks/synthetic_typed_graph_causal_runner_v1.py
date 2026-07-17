"""Offline staged runner for the frozen synthetic typed-graph causal study.

Only acquisition-produced action views enter retrieval.  Gold, family,
polarity, matching, and grammar item commitments live in a separate 0600 pack
that is opened after the complete action/postflight barrier.  F-search has no
label pack.  Official retrieval is invoked once per item with at most eight
live calls; all four graph interventions reuse that exact output.

The complete magnitude-sign enumeration is retained as the preregistered
protocol promotion heuristic and reference tail.  It is not reported as a
design-based randomization p-value or population inference.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from replication_runtime.qasper_minilm_v1 import (
    OfflineMiniLMEncoder,
    quantized_cosine_similarity,
)

from . import contractnli_typed_clause_graph_v1 as core
from . import synthetic_typed_graph_causal_grammar_v1 as grammar
from .musique_formal_runtime_binding_v2 import (
    PreparedFormalRuntimeV2,
    prepare_formal_runtime_v2,
)
from .synthetic_typed_graph_causal_acquisition_v1 import (
    ACQUISITION_RECEIPT_RELATIVE_PATH,
    DESIGN_FILE_SHA256,
    DESIGN_SHA256,
    GRAPH_CORE_SHA256,
    GRAMMAR_SHA256,
    LABEL_ITEM_SCHEMA,
    LABEL_SCHEMA,
    PRIVATE_MODE,
    PRIVATE_COHORT_RELATIVE_PATH,
    VIEW_ITEM_SCHEMA,
    VIEW_SCHEMA,
    PUBLIC_MODE,
    _committed_bytes,
    _git,
    _git_project_prefix,
    _assert_no_symlink_components,
    _write_json_exclusive,
    canonical_bytes,
    load_committed_acquisition_receipt,
    semantic_hash,
    verify_implementation_freeze,
)


VERSION = "synthetic_typed_graph_causal_runner_v1"
BLOCK_SIZE = 64
TOP_K = 5
OFFICIAL_CONCURRENCY_CAP = 8
LOCAL_CONCURRENCY_CAP = 64
R0 = "R0_HIPPO_TOP5"
RECIPE_IDS = tuple(recipe.recipe_id for recipe in core.recipe_registry())
EVALUATOR_IDS = tuple(evaluator.evaluator_id for evaluator in core.evaluator_registry())
REFERENCE_TAIL_INTERPRETATION = (
    "complete_magnitude_sign_enumeration_protocol_heuristic_not_"
    "design_based_randomization_or_population_p_value"
)
MINILM_ASSET_SHA256 = "921d9b1945581130e03c53f448092c3de3b30714431c6cac9b3b32c2ec10abad"

STAGE_ROOT_RELATIVE_PATH = Path(
    "artifacts/synthetic_typed_graph_causal_v1/runner"
)
FORMATION_RECEIPT_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_causal_formation_v1.json"
)
A_HOLD_RECEIPT_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_causal_A_hold_v1.json"
)
M_SEARCH_RECEIPT_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_causal_M_search_v1.json"
)
STAGE_RECEIPT_PATHS = {
    "formation": FORMATION_RECEIPT_RELATIVE_PATH,
    "A_hold": A_HOLD_RECEIPT_RELATIVE_PATH,
    "M_search": M_SEARCH_RECEIPT_RELATIVE_PATH,
}
MINILM_MANIFEST_RELATIVE_PATH = Path("manifests/qasper_minilm_runtime_asset_v1.json")
MINILM_MODEL_ROOT_RELATIVE_PATH = Path("artifacts/qasper_minilm_runtime_v1/model")
OFFICIAL_BASE_RECEIPT_RELATIVE_PATH = Path(
    "manifests/musique_official_hipporag_retrieve_only_binding_v1.json"
)
OFFICIAL_ATTESTATION_RELATIVE_PATH = Path(
    "manifests/musique_official_hipporag_runtime_attestation_v2.json"
)
_FORMAL_ENTRY_ACTIVE = False


class SyntheticCausalRunnerError(RuntimeError):
    """A private pack, runtime, action, label, or aggregate invariant failed."""


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


@dataclass(frozen=True)
class ViewNode:
    span_i: int
    start: int
    end: int
    identity_text: str

    def source_span(self) -> core.SourceSpan:
        return core.SourceSpan(self.span_i, self.start, self.end, self.identity_text)


@dataclass(frozen=True)
class ViewItem:
    block: str
    ordinal: int
    opaque_view_sha256: str
    question: str
    context: str
    nodes: tuple[ViewNode, ...]
    edges_by_mode: Mapping[str, tuple[core.TypedEdge, ...]]

    @property
    def spans(self) -> tuple[core.SourceSpan, ...]:
        return tuple(node.source_span() for node in self.nodes)

    @property
    def paragraphs(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "idx": node.span_i,
                "title": "synthetic_typed_graph_causal_v1",
                "paragraph_text": node.identity_text,
            }
            for node in self.nodes
        )


@dataclass(frozen=True)
class ViewBlock:
    block: str
    block_sha256: str
    file_sha256: str
    rows: tuple[ViewItem, ...]


@dataclass(frozen=True)
class LabelItem:
    block: str
    ordinal: int
    opaque_view_sha256: str
    item_commitment_sha256: str
    label_free_commitment_sha256: str
    matching_signature_sha256: str
    structural_draw_sha256: str
    family_slot: int
    family_id: str
    family_role: str
    template_split: str
    polarity: str
    negative_kind: str | None
    edge_family: str
    pair_key: str
    gold_node_indices: tuple[int, ...]
    permuted_from_opaque_view_sha256: str | None = None
    permuted_gold_node_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class LabelBlock:
    block: str
    block_sha256: str
    file_sha256: str
    rows: tuple[LabelItem, ...]


@dataclass(frozen=True)
class LocalTensor:
    raw_top5: tuple[int, int, int, int, int]
    query_similarities: tuple[int, ...]
    span_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class FormationActions:
    view: ViewItem
    raw_top5: tuple[int, int, int, int, int]
    official_top5: tuple[int, int, int, int, int]
    traces: Mapping[str, core.ActionTrace]
    components: Mapping[str, core.CoverageComponents]
    local_tensor_sha256: str


@dataclass(frozen=True)
class MeasurementActions:
    view: ViewItem
    raw_top5: tuple[int, int, int, int, int]
    official_top5: tuple[int, int, int, int, int]
    outputs_by_mode: Mapping[str, tuple[int, int, int, int, int]]
    permuted_evaluator_top5: tuple[int, int, int, int, int]
    fixed_e00_top5: tuple[int, int, int, int, int]
    scan_hashes_by_mode: Mapping[str, str]
    local_tensor_sha256: str


@dataclass(frozen=True)
class FormationOutcome:
    real_evaluator_id: str
    permuted_evaluator_id: str
    fixed_e00_evaluator_id: str
    real_recipe_id: str
    permuted_recipe_id: str
    fixed_e00_recipe_id: str
    identifiable_transition: bool
    evaluator_control_same_recipe: bool
    effective_same_gold_vector_count: int
    arm_aggregates: Mapping[str, Mapping[str, int]]
    action_table_sha256: str
    action_seal_sha256: str | None
    action_seal_file_sha256: str | None
    runtime_binding_sha256: str
    a_view_file_sha256: str
    a_label_file_sha256: str
    f_view_file_sha256: str


@dataclass(frozen=True)
class MeasurementOutcome:
    block: str
    selected_recipe_id: str
    permuted_recipe_id: str
    fixed_e00_recipe_id: str
    primary_reference_test: Mapping[str, object]
    mechanism_reference_tests: Mapping[str, Mapping[str, object]]
    evaluator_reference_tests: Mapping[str, Mapping[str, object]]
    aggregates: Mapping[str, Any]
    delta_hashes: Mapping[str, str]
    action_table_sha256: str
    action_seal_sha256: str | None
    action_seal_file_sha256: str | None
    runtime_binding_sha256: str
    view_file_sha256: str
    label_file_sha256: str


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_private(path: Path, field: str) -> tuple[dict[str, Any], str]:
    absolute = _assert_no_symlink_components(path, field)
    if not absolute.is_file() or absolute.is_symlink():
        raise SyntheticCausalRunnerError(f"{field} is unavailable")
    info = absolute.stat()
    if stat.S_IMODE(info.st_mode) != PRIVATE_MODE or not 1 <= info.st_size <= 32 * 1024 * 1024:
        raise SyntheticCausalRunnerError(f"{field} mode or size drifted")
    raw = absolute.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticCausalRunnerError(f"{field} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise SyntheticCausalRunnerError(f"{field} root is not an object")
    return payload, _sha256_bytes(raw)


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    body = dict(payload)
    declared = body.pop(field, None)
    if not isinstance(declared, str) or semantic_hash(body) != declared:
        raise SyntheticCausalRunnerError(f"{field} drifted")
    return declared


def _typed_edge(raw: object) -> core.TypedEdge:
    if (
        not isinstance(raw, list)
        or len(raw) != 3
        or raw[0] not in core.EDGE_FAMILIES
        or type(raw[1]) is not int
        or type(raw[2]) is not int
        or not 0 <= raw[1] < raw[2] < grammar.NODE_COUNT
    ):
        raise SyntheticCausalRunnerError("typed edge row drifted")
    return core.TypedEdge(core.EDGE_FAMILY_ORDER[str(raw[0])], raw[1], raw[2])


def _parse_view(raw: object, block: str, ordinal: int) -> ViewItem:
    if not isinstance(raw, Mapping):
        raise SyntheticCausalRunnerError("view item is not an object")
    allowed = {
        "schema", "block", "ordinal", "question", "context", "nodes",
        "edges_by_mode", "opaque_view_sha256",
    }
    if set(raw) != allowed or raw.get("schema") != VIEW_ITEM_SCHEMA:
        raise SyntheticCausalRunnerError("view item schema drifted")
    if raw.get("block") != block or raw.get("ordinal") != ordinal:
        raise SyntheticCausalRunnerError("view item coordinate drifted")
    _self_hash(raw, "opaque_view_sha256")
    question, context = raw.get("question"), raw.get("context")
    if not isinstance(question, str) or not question or not isinstance(context, str):
        raise SyntheticCausalRunnerError("view text drifted")
    node_rows = raw.get("nodes")
    if not isinstance(node_rows, list) or len(node_rows) != grammar.NODE_COUNT:
        raise SyntheticCausalRunnerError("view node count drifted")
    nodes: list[ViewNode] = []
    previous_end = 0
    for index, node in enumerate(node_rows):
        if not isinstance(node, Mapping) or set(node) != {
            "span_i", "start", "end", "identity_text"
        }:
            raise SyntheticCausalRunnerError("view node schema drifted")
        start, end, text = node.get("start"), node.get("end"), node.get("identity_text")
        if (
            node.get("span_i") != index
            or type(start) is not int
            or type(end) is not int
            or start < previous_end
            or end <= start
            or not isinstance(text, str)
            or context[start:end] != text
        ):
            raise SyntheticCausalRunnerError("view node content drifted")
        nodes.append(ViewNode(index, start, end, text))
        previous_end = end
    edge_rows = raw.get("edges_by_mode")
    if not isinstance(edge_rows, Mapping) or set(edge_rows) != set(grammar.ABLATION_MODES):
        raise SyntheticCausalRunnerError("graph mode registry drifted")
    edges: dict[str, tuple[core.TypedEdge, ...]] = {}
    for mode in grammar.ABLATION_MODES:
        rows = edge_rows[mode]
        if not isinstance(rows, list):
            raise SyntheticCausalRunnerError("graph mode edge table drifted")
        parsed = tuple(_typed_edge(row) for row in rows)
        if tuple(sorted(set(parsed))) != parsed:
            raise SyntheticCausalRunnerError("graph mode edges are not canonical")
        edges[mode] = parsed
    expected_full = core.build_typed_clause_graph(
        tuple(node.source_span() for node in nodes)
    )
    if edges[grammar.FULL_GRAPH] != expected_full:
        raise SyntheticCausalRunnerError("full graph is not the pinned core output")
    if edges[grammar.FULL_GRAPH] == edges[grammar.DROP_DESIGNATED]:
        raise SyntheticCausalRunnerError("drop intervention has no structural contrast")
    return ViewItem(
        block, ordinal, str(raw["opaque_view_sha256"]), question, context,
        tuple(nodes), edges,
    )


def load_view_block(path: Path, expected_block: str) -> ViewBlock:
    if expected_block not in grammar.BLOCK_ORDER:
        raise SyntheticCausalRunnerError("view block is unknown")
    payload, file_hash = _read_private(path, f"{expected_block} view pack")
    declared = _self_hash(payload, "block_sha256")
    rows = payload.get("rows")
    if (
        set(payload) != {"schema", "block", "count", "rows", "block_sha256"}
        or payload.get("schema") != VIEW_SCHEMA
        or payload.get("block") != expected_block
        or payload.get("count") != BLOCK_SIZE
        or not isinstance(rows, list)
        or len(rows) != BLOCK_SIZE
    ):
        raise SyntheticCausalRunnerError("view block schema drifted")
    parsed = tuple(_parse_view(row, expected_block, i) for i, row in enumerate(rows))
    if len({row.opaque_view_sha256 for row in parsed}) != BLOCK_SIZE:
        raise SyntheticCausalRunnerError("view identifiers overlap")
    return ViewBlock(expected_block, declared, file_hash, parsed)


def _parse_label(raw: object, block: str, ordinal: int) -> LabelItem:
    if not isinstance(raw, Mapping) or raw.get("schema") != LABEL_ITEM_SCHEMA:
        raise SyntheticCausalRunnerError("label item schema drifted")
    base = {
        "schema", "block", "ordinal", "opaque_view_sha256",
        "item_commitment_sha256", "label_free_commitment_sha256",
        "matching_signature_sha256", "structural_draw_sha256", "family_slot",
        "family_id", "family_role", "template_split", "polarity",
        "negative_kind", "edge_family", "pair_key", "gold_node_indices",
    }
    extra = {
        "permuted_from_item_commitment_sha256",
        "permuted_from_opaque_view_sha256",
        "permuted_gold_node_indices",
    }
    if set(raw) != (base | extra if block == "A_form" else base):
        raise SyntheticCausalRunnerError("label item field set drifted")
    gold = raw.get("gold_node_indices")
    if (
        raw.get("block") != block
        or raw.get("ordinal") != ordinal
        or not isinstance(gold, list)
        or not 1 <= len(gold) <= 3
        or gold != sorted(set(gold))
        or any(type(index) is not int or not 0 <= index < grammar.NODE_COUNT for index in gold)
        or raw.get("polarity") not in {grammar.POSITIVE, grammar.NEGATIVE}
        or raw.get("edge_family") not in grammar.EDGE_FAMILIES
        or raw.get("family_id") not in grammar.FAMILY_BY_ID
    ):
        raise SyntheticCausalRunnerError("label item content drifted")
    family = grammar.FAMILY_BY_ID[str(raw["family_id"])]
    for field, expected in {
        "family_role": family.family_role,
        "template_split": family.template_split,
        "polarity": family.polarity,
        "negative_kind": family.negative_kind,
        "edge_family": family.edge_family,
    }.items():
        if raw.get(field) != expected:
            raise SyntheticCausalRunnerError("label family registry drifted")
    permuted = raw.get("permuted_gold_node_indices")
    if block == "A_form" and (
        not isinstance(permuted, list)
        or len(permuted) != len(gold)
        or permuted != sorted(set(permuted))
        or any(type(index) is not int or not 0 <= index < grammar.NODE_COUNT for index in permuted)
    ):
        raise SyntheticCausalRunnerError("permuted A-form gold drifted")
    return LabelItem(
        block=block,
        ordinal=ordinal,
        opaque_view_sha256=str(raw["opaque_view_sha256"]),
        item_commitment_sha256=str(raw["item_commitment_sha256"]),
        label_free_commitment_sha256=str(raw["label_free_commitment_sha256"]),
        matching_signature_sha256=str(raw["matching_signature_sha256"]),
        structural_draw_sha256=str(raw["structural_draw_sha256"]),
        family_slot=int(raw["family_slot"]),
        family_id=str(raw["family_id"]),
        family_role=str(raw["family_role"]),
        template_split=str(raw["template_split"]),
        polarity=str(raw["polarity"]),
        negative_kind=raw["negative_kind"],
        edge_family=str(raw["edge_family"]),
        pair_key=str(raw["pair_key"]),
        gold_node_indices=tuple(gold),
        permuted_from_opaque_view_sha256=(
            str(raw["permuted_from_opaque_view_sha256"])
            if block == "A_form" else None
        ),
        permuted_gold_node_indices=tuple(permuted) if block == "A_form" else None,
    )


def load_label_block(path: Path, expected_block: str) -> LabelBlock:
    if expected_block not in {"A_form", "A_hold", "M_search"}:
        raise SyntheticCausalRunnerError("labels are forbidden for this block")
    payload, file_hash = _read_private(path, f"{expected_block} label pack")
    declared = _self_hash(payload, "block_sha256")
    rows = payload.get("rows")
    if (
        set(payload) != {"schema", "block", "count", "rows", "block_sha256"}
        or payload.get("schema") != LABEL_SCHEMA
        or payload.get("block") != expected_block
        or payload.get("count") != BLOCK_SIZE
        or not isinstance(rows, list)
        or len(rows) != BLOCK_SIZE
    ):
        raise SyntheticCausalRunnerError("label block schema drifted")
    parsed = tuple(_parse_label(row, expected_block, i) for i, row in enumerate(rows))
    if len({row.item_commitment_sha256 for row in parsed}) != BLOCK_SIZE:
        raise SyntheticCausalRunnerError("label item commitments overlap")
    if expected_block == "A_form":
        by_view = {row.opaque_view_sha256: row for row in parsed}
        sources = [row.permuted_from_opaque_view_sha256 for row in parsed]
        if len(set(sources)) != BLOCK_SIZE or set(sources) != set(by_view):
            raise SyntheticCausalRunnerError("A-form derangement is not bijective")
        for row in parsed:
            source = by_view[str(row.permuted_from_opaque_view_sha256)]
            if source is row or source.edge_family != row.edge_family or (
                len(source.gold_node_indices) != len(row.gold_node_indices)
            ) or row.permuted_gold_node_indices != source.gold_node_indices:
                raise SyntheticCausalRunnerError("A-form derangement stratum drifted")
    return LabelBlock(expected_block, declared, file_hash, parsed)


def load_canonical_view_block(project_root: Path, block: str) -> ViewBlock:
    receipt = load_committed_acquisition_receipt(project_root)
    pack = receipt["packs"].get(block)
    if not isinstance(pack, Mapping):
        raise SyntheticCausalRunnerError("canonical view receipt row is absent")
    path = project_root.resolve() / PRIVATE_COHORT_RELATIVE_PATH / (
        f"{block}.label_free.sealed.json"
    )
    loaded = load_view_block(path, block)
    if loaded.file_sha256 != pack.get("view_file_sha256") or (
        loaded.block_sha256 != pack.get("view_block_sha256")
    ):
        raise SyntheticCausalRunnerError("canonical view pack hash differs from receipt")
    return loaded


def load_canonical_label_block(project_root: Path, block: str) -> LabelBlock:
    if block == "F_search":
        raise SyntheticCausalRunnerError("F_search labels are forbidden")
    receipt = load_committed_acquisition_receipt(project_root)
    pack = receipt["packs"].get(block)
    if not isinstance(pack, Mapping):
        raise SyntheticCausalRunnerError("canonical label receipt row is absent")
    path = project_root.resolve() / PRIVATE_COHORT_RELATIVE_PATH / (
        f"{block}.labels.sealed.json"
    )
    loaded = load_label_block(path, block)
    if loaded.file_sha256 != pack.get("label_file_sha256") or (
        loaded.block_sha256 != pack.get("label_block_sha256")
    ):
        raise SyntheticCausalRunnerError("canonical label pack hash differs from receipt")
    return loaded


def _join(view: ViewBlock, labels: LabelBlock) -> tuple[tuple[ViewItem, LabelItem], ...]:
    if view.block != labels.block:
        raise SyntheticCausalRunnerError("view/label block mismatch")
    joined = tuple(zip(view.rows, labels.rows))
    if any(
        item.ordinal != label.ordinal
        or item.opaque_view_sha256 != label.opaque_view_sha256
        for item, label in joined
    ):
        raise SyntheticCausalRunnerError("view/label identity mismatch")
    return joined


def _quantized_vector(left: np.ndarray, rows: np.ndarray) -> tuple[int, ...]:
    return tuple(quantized_cosine_similarity(left, row) for row in rows)


def _tensor_from_embeddings(
    item: ViewItem, query: np.ndarray, node_rows: np.ndarray
) -> LocalTensor:
    query_similarities = _quantized_vector(query, node_rows)
    span_matrix = tuple(
        tuple(quantized_cosine_similarity(left, right) for right in node_rows)
        for left in node_rows
    )
    raw = tuple(
        sorted(
            range(grammar.NODE_COUNT),
            key=lambda index: (-query_similarities[index], index),
        )[:TOP_K]
    )
    return LocalTensor(raw, query_similarities, span_matrix)  # type: ignore[arg-type]


def precompute_local_blocks(
    blocks: Sequence[ViewBlock], encoder: EncoderProtocol
) -> dict[str, tuple[LocalTensor, ...]]:
    """Encode all active items in one offline batch and fan out 64 local workers."""

    texts: list[str] = []
    coordinates: list[tuple[str, int, int]] = []
    for block in blocks:
        for item in block.rows:
            start = len(texts)
            texts.append(item.question)
            texts.extend(core.embedding_text(node.identity_text) for node in item.nodes)
            coordinates.append((block.block, item.ordinal, start))
    try:
        matrix = np.asarray(encoder.encode(tuple(texts)), dtype=np.float32)
    except Exception as exc:
        raise SyntheticCausalRunnerError("offline MiniLM batch failed") from exc
    if matrix.ndim != 2 or matrix.shape[0] != len(texts) or not np.isfinite(matrix).all():
        raise SyntheticCausalRunnerError("offline MiniLM output drifted")
    output: dict[str, list[LocalTensor | None]] = {
        block.block: [None] * BLOCK_SIZE for block in blocks
    }
    futures: dict[Future[LocalTensor], tuple[str, int]] = {}
    with ThreadPoolExecutor(max_workers=LOCAL_CONCURRENCY_CAP) as pool:
        for block_id, ordinal, start in coordinates:
            futures[
                pool.submit(
                    _tensor_from_embeddings,
                    next(block.rows[ordinal] for block in blocks if block.block == block_id),
                    matrix[start],
                    matrix[start + 1 : start + 1 + grammar.NODE_COUNT],
                )
            ] = (block_id, ordinal)
        for future in as_completed(futures):
            block_id, ordinal = futures[future]
            output[block_id][ordinal] = future.result()
    if any(any(row is None for row in rows) for rows in output.values()):
        raise SyntheticCausalRunnerError("local tensor completion barrier failed")
    return {
        block: tuple(row for row in rows if row is not None)
        for block, rows in output.items()
    }


def _validated_official(value: Sequence[int]) -> tuple[int, int, int, int, int]:
    rows = tuple(value)
    if (
        len(rows) != TOP_K
        or len(set(rows)) != TOP_K
        or any(type(index) is not int or not 0 <= index < grammar.NODE_COUNT for index in rows)
    ):
        raise SyntheticCausalRunnerError("official Hippo top5 drifted")
    return rows  # type: ignore[return-value]


def _runtime_binding(runtime: OfficialRuntimeProtocol) -> str:
    safe = dict(runtime.safe_binding)
    serialized = json.dumps(safe, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if "/home/" in serialized or "/tmp/" in serialized or "\\" in serialized:
        raise SyntheticCausalRunnerError("official runtime safe binding leaks a host path")
    return semantic_hash(safe)


def _local_tensor_hash(tensor: LocalTensor) -> str:
    return semantic_hash(
        {
            "raw_top5": list(tensor.raw_top5),
            "query_similarities": list(tensor.query_similarities),
            "span_matrix": [list(row) for row in tensor.span_matrix],
        }
    )


def _execute_formation_item(
    item: ViewItem,
    tensor: LocalTensor,
    official: tuple[int, int, int, int, int],
) -> FormationActions:
    table = core.build_common_candidate_table(
        item.spans,
        item.edges_by_mode[grammar.FULL_GRAPH],
        official,
        tensor.query_similarities,
    )
    traces = core.execute_all_recipes(official, table, tensor.query_similarities)
    if tuple(trace.recipe_id for trace in traces) != RECIPE_IDS or len(
        {trace.common_scan_sha256 for trace in traces}
    ) != 1:
        raise SyntheticCausalRunnerError("formation common action scan drifted")
    components = {
        trace.recipe_id: core.coverage_components(
            item.question,
            item.spans,
            trace.output_top5,
            official,
            item.edges_by_mode[grammar.FULL_GRAPH],
            tensor.query_similarities,
            tensor.span_matrix,
        )
        for trace in traces
    }
    return FormationActions(
        item,
        tensor.raw_top5,
        official,
        {trace.recipe_id: trace for trace in traces},
        components,
        _local_tensor_hash(tensor),
    )


def _execute_measurement_item(
    item: ViewItem,
    tensor: LocalTensor,
    official: tuple[int, int, int, int, int],
    real_recipe_id: str,
    permuted_recipe_id: str,
    fixed_e00_recipe_id: str,
) -> MeasurementActions:
    outputs: dict[str, tuple[int, int, int, int, int]] = {}
    scans: dict[str, str] = {}
    for mode in grammar.ABLATION_MODES:
        table = core.build_common_candidate_table(
            item.spans, item.edges_by_mode[mode], official, tensor.query_similarities
        )
        identity = core.execute_recipe(official, table, tensor.query_similarities, R0)
        selected = core.execute_recipe(
            official, table, tensor.query_similarities, real_recipe_id
        )
        if identity.output_top5 != official or (
            identity.common_scan_sha256 != selected.common_scan_sha256
        ):
            raise SyntheticCausalRunnerError("measurement paired scan drifted")
        outputs[mode] = selected.output_top5
        scans[mode] = selected.common_scan_sha256
    full_table = core.build_common_candidate_table(
        item.spans,
        item.edges_by_mode[grammar.FULL_GRAPH],
        official,
        tensor.query_similarities,
    )
    permuted = core.execute_recipe(
        official, full_table, tensor.query_similarities, permuted_recipe_id
    )
    fixed = core.execute_recipe(
        official, full_table, tensor.query_similarities, fixed_e00_recipe_id
    )
    if permuted.common_scan_sha256 != scans[grammar.FULL_GRAPH] or (
        fixed.common_scan_sha256 != scans[grammar.FULL_GRAPH]
    ):
        raise SyntheticCausalRunnerError("evaluator control scan drifted")
    return MeasurementActions(
        item,
        tensor.raw_top5,
        official,
        outputs,
        permuted.output_top5,
        fixed.output_top5,
        scans,
        _local_tensor_hash(tensor),
    )


def _run_official_wave(
    blocks: Sequence[ViewBlock],
    tensors: Mapping[str, Sequence[LocalTensor]],
    runtime: OfficialRuntimeProtocol,
    work_root: Path,
    *,
    recipes: tuple[str, str, str] | None,
) -> tuple[dict[str, tuple[FormationActions | MeasurementActions, ...]], str]:
    if work_root.exists():
        raise SyntheticCausalRunnerError("official work root already exists")
    work_root.mkdir(parents=True, mode=0o700)
    ordered: dict[str, list[FormationActions | MeasurementActions | None]] = {
        block.block: [None] * BLOCK_SIZE for block in blocks
    }
    official_futures: dict[Future[tuple[int, ...]], tuple[ViewItem, LocalTensor]] = {}
    local_futures: dict[
        Future[FormationActions | MeasurementActions], tuple[str, int]
    ] = {}
    with ThreadPoolExecutor(max_workers=OFFICIAL_CONCURRENCY_CAP) as official_pool, ThreadPoolExecutor(
        max_workers=LOCAL_CONCURRENCY_CAP
    ) as local_pool:
        for block in blocks:
            for item, tensor in zip(block.rows, tensors[block.block]):
                future = official_pool.submit(
                    runtime.retrieve,
                    question=item.question,
                    paragraphs=item.paragraphs,
                    work_root=work_root / f"{block.block}_{item.ordinal:03d}",
                )
                official_futures[future] = (item, tensor)
        try:
            for future in as_completed(official_futures):
                item, tensor = official_futures[future]
                official = _validated_official(future.result())
                if recipes is None:
                    local_future = local_pool.submit(
                        _execute_formation_item, item, tensor, official
                    )
                else:
                    local_future = local_pool.submit(
                        _execute_measurement_item,
                        item,
                        tensor,
                        official,
                        recipes[0],
                        recipes[1],
                        recipes[2],
                    )
                local_futures[local_future] = (item.block, item.ordinal)
            for future in as_completed(local_futures):
                block, ordinal = local_futures[future]
                ordered[block][ordinal] = future.result()
        except Exception:
            for future in (*official_futures, *local_futures):
                future.cancel()
            raise
    if any(any(row is None for row in rows) for rows in ordered.values()):
        raise SyntheticCausalRunnerError("action completion barrier failed")
    try:
        postflight = dict(runtime.fresh_reverify())
    except Exception as exc:
        raise SyntheticCausalRunnerError("official runtime postflight failed") from exc
    if semantic_hash(postflight) != _runtime_binding(runtime):
        raise SyntheticCausalRunnerError("official runtime postflight binding drifted")
    return (
        {
            block: tuple(row for row in rows if row is not None)
            for block, rows in ordered.items()
        },
        semantic_hash(postflight),
    )


def _utility(top5: Sequence[int], gold: Sequence[int]) -> tuple[int, int, int]:
    return core.item_utility(top5, gold, source_count=grammar.NODE_COUNT)


def _arm_aggregate(
    joined: Sequence[tuple[ViewItem, LabelItem]],
    outputs: Sequence[Sequence[int]],
) -> tuple[dict[str, int], tuple[int, ...]]:
    if len(joined) != len(outputs):
        raise SyntheticCausalRunnerError("arm output length drifted")
    utilities: list[int] = []
    hits = complete = support = 0
    for (_view, label), top5 in zip(joined, outputs):
        item_hits, item_complete, utility = _utility(top5, label.gold_node_indices)
        hits += item_hits
        complete += item_complete
        support += len(label.gold_node_indices)
        utilities.append(utility)
    return (
        {
            "item_count": len(joined),
            "support_hit_count": hits,
            "support_total": support,
            "complete_count": complete,
            "total_U": sum(utilities),
        },
        tuple(utilities),
    )


def _formation_items(
    joined: Sequence[tuple[ViewItem, LabelItem]],
    actions: Sequence[FormationActions],
    *,
    permuted: bool,
) -> tuple[core.FormationItem, ...]:
    rows: list[core.FormationItem] = []
    for (view, label), action in zip(joined, actions):
        if action.view.opaque_view_sha256 != view.opaque_view_sha256:
            raise SyntheticCausalRunnerError("formation action/label join drifted")
        gold = (
            label.permuted_gold_node_indices
            if permuted else label.gold_node_indices
        )
        if gold is None:
            raise SyntheticCausalRunnerError("permuted formation label is absent")
        utilities: dict[str, int] = {}
        complete: dict[str, bool] = {}
        for recipe_id in RECIPE_IDS:
            _hits, is_complete, utility = _utility(
                action.traces[recipe_id].output_top5, gold
            )
            utilities[recipe_id] = utility
            complete[recipe_id] = bool(is_complete)
        rows.append(core.FormationItem(action.components, utilities, complete))
    return tuple(rows)


def _persist_action_seal(
    *,
    path: Path | None,
    stage: str,
    action_table_sha256: str,
    action_rows: Sequence[Mapping[str, Any]],
    runtime_binding_sha256: str,
    view_file_sha256s: Mapping[str, str],
    recipe_ids: Mapping[str, str] | None,
) -> tuple[str | None, str | None]:
    if path is None:
        if _FORMAL_ENTRY_ACTIVE:
            raise SyntheticCausalRunnerError("formal action seal path is absent")
        return None, None
    body = {
        "schema": f"{VERSION}_{stage}_label_free_action_seal",
        "version": VERSION,
        "stage": stage,
        "status": "all_required_label_free_actions_and_official_postflight_terminal",
        "action_table_sha256": action_table_sha256,
        "action_rows": list(action_rows),
        "minilm_asset_sha256": MINILM_ASSET_SHA256,
        "runtime_binding_sha256": runtime_binding_sha256,
        "view_file_sha256s": dict(sorted(view_file_sha256s.items())),
        "recipe_ids": dict(sorted(recipe_ids.items())) if recipe_ids is not None else None,
        "labels_opened_before_seal": False,
        "F_search_labels_created_or_opened": False,
    }
    if semantic_hash(list(action_rows)) != action_table_sha256:
        raise SyntheticCausalRunnerError("action seal preimage hash drifted")
    seal = _receipt(body)
    seal["action_seal_sha256"] = seal.pop("receipt_sha256")
    file_hash = _write_json_exclusive(path, seal, PRIVATE_MODE)
    return str(seal["action_seal_sha256"]), file_hash


def run_formation(
    a_view: ViewBlock,
    f_view: ViewBlock,
    *,
    a_label_loader: Callable[[], LabelBlock],
    encoder: EncoderProtocol,
    runtime: OfficialRuntimeProtocol,
    work_root: Path,
    action_seal_path: Path | None = None,
) -> FormationOutcome:
    if a_view.block != "A_form" or f_view.block != "F_search":
        raise SyntheticCausalRunnerError("formation blocks drifted")
    if len({
        row.opaque_view_sha256 for row in (*a_view.rows, *f_view.rows)
    }) != 2 * BLOCK_SIZE:
        raise SyntheticCausalRunnerError("A/F opaque view identities overlap")
    tensors = precompute_local_blocks((a_view, f_view), encoder)
    action_map, runtime_hash = _run_official_wave(
        (a_view, f_view), tensors, runtime, work_root, recipes=None
    )
    a_actions = tuple(action_map["A_form"])
    f_actions = tuple(action_map["F_search"])
    if any(not isinstance(row, FormationActions) for row in (*a_actions, *f_actions)):
        raise SyntheticCausalRunnerError("formation action type drifted")
    action_rows = [
        {
            "block": row.view.block,
            "ordinal": row.view.ordinal,
            "opaque_view_sha256": row.view.opaque_view_sha256,
            "local_tensor_sha256": row.local_tensor_sha256,
            "raw": list(row.raw_top5),
            "official": list(row.official_top5),
            "recipe_outputs": {
                recipe_id: list(row.traces[recipe_id].output_top5)
                for recipe_id in RECIPE_IDS
            },
            "common_scan_sha256": {
                recipe_id: row.traces[recipe_id].common_scan_sha256
                for recipe_id in RECIPE_IDS
            },
            "coverage_components": {
                recipe_id: row.components[recipe_id].as_mapping()
                for recipe_id in RECIPE_IDS
            },
        }
        for row in (*a_actions, *f_actions)
    ]
    action_table_hash = semantic_hash(action_rows)
    action_seal_hash, action_seal_file_hash = _persist_action_seal(
        path=action_seal_path,
        stage="formation",
        action_table_sha256=action_table_hash,
        action_rows=action_rows,
        runtime_binding_sha256=runtime_hash,
        view_file_sha256s={
            "A_form": a_view.file_sha256,
            "F_search": f_view.file_sha256,
        },
        recipe_ids=None,
    )
    # This is the first label access in the function, after every A/F action and
    # the official runtime postflight has reached a terminal state.
    a_labels = a_label_loader()
    joined = _join(a_view, a_labels)
    same_gold_vector_count = sum(
        label.permuted_gold_node_indices == label.gold_node_indices
        for _view, label in joined
    )
    real = core.select_a_evaluator(
        _formation_items(joined, a_actions, permuted=False)  # type: ignore[arg-type]
    )
    permuted = core.select_a_evaluator(
        _formation_items(joined, a_actions, permuted=True)  # type: ignore[arg-type]
    )
    component_tables = [
        row.components for row in f_actions if isinstance(row, FormationActions)
    ]
    real_f = core.select_f_recipe(component_tables, real.evaluator_id)
    perm_f = core.select_f_recipe(component_tables, permuted.evaluator_id)
    e00_f = core.select_f_recipe(component_tables, grammar.E00_CONTROL_EVALUATOR_ID)
    real_outputs = tuple(
        row.traces[real_f.recipe_id].output_top5
        for row in f_actions if isinstance(row, FormationActions)
    )
    official_outputs = tuple(
        row.official_top5 for row in f_actions if isinstance(row, FormationActions)
    )
    identifiable = core.has_identifiable_transition(
        real_f.recipe_id, real_outputs, official_outputs
    )
    a_output_map = {
        "canonical_RAW": tuple(row.raw_top5 for row in a_actions),
        "official_HippoRAG": tuple(row.official_top5 for row in a_actions),
        "Agent_real": tuple(row.traces[real_f.recipe_id].output_top5 for row in a_actions),
        "Agent_permuted_evaluator": tuple(
            row.traces[perm_f.recipe_id].output_top5 for row in a_actions
        ),
        "Agent_E00": tuple(row.traces[e00_f.recipe_id].output_top5 for row in a_actions),
    }
    arm_aggregates = {
        arm: _arm_aggregate(joined, outputs)[0]
        for arm, outputs in a_output_map.items()
    }
    return FormationOutcome(
        real.evaluator_id,
        permuted.evaluator_id,
        grammar.E00_CONTROL_EVALUATOR_ID,
        real_f.recipe_id,
        perm_f.recipe_id,
        e00_f.recipe_id,
        identifiable,
        real_f.recipe_id == perm_f.recipe_id,
        same_gold_vector_count,
        arm_aggregates,
        action_table_hash,
        action_seal_hash,
        action_seal_file_hash,
        runtime_hash,
        a_view.file_sha256,
        a_labels.file_sha256,
        f_view.file_sha256,
    )


def run_formal_formation(
    *,
    project_root: Path,
    encoder: EncoderProtocol,
    runtime: OfficialRuntimeProtocol,
    work_root: Path,
    action_seal_path: Path,
) -> FormationOutcome:
    """Canonical formation entrypoint bound to the committed acquisition receipt."""

    a_view = load_canonical_view_block(project_root, "A_form")
    f_view = load_canonical_view_block(project_root, "F_search")
    return run_formation(
        a_view,
        f_view,
        a_label_loader=lambda: load_canonical_label_block(project_root, "A_form"),
        encoder=encoder,
        runtime=runtime,
        work_root=work_root,
        action_seal_path=action_seal_path,
    )


def _reference_tail(deltas: Sequence[int]) -> dict[str, object]:
    raw = core.exact_magnitude_preserving_sign_flip(deltas)
    return {
        "statistic": "complete_one_sided_magnitude_sign_enumeration_v1",
        "interpretation": REFERENCE_TAIL_INTERPRETATION,
        "observed_net_U": raw["observed_net_U"],
        "nonzero_pair_count": raw["nonzero_pair_count"],
        "reference_tail_numerator": raw["p_value_numerator"],
        "reference_tail_denominator": raw["p_value_denominator"],
        "protocol_alpha_numerator": raw["alpha_numerator"],
        "protocol_alpha_denominator": raw["alpha_denominator"],
        "positive_observed_net": raw["positive_observed_net"],
        "reference_tail_at_or_below_protocol_alpha": raw[
            "exact_p_at_or_below_alpha"
        ],
        "positive_and_reference_tail_at_or_below_threshold": raw["promoted"],
        "design_based_randomization_p_value": False,
        "population_or_multi_seed_inference": False,
    }


def _stratum_summary(
    labels: Sequence[LabelItem], utilities: Sequence[int], field: str
) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for label, utility in zip(labels, utilities):
        grouped[str(getattr(label, field))].append(utility)
    return {
        key: {"item_count": len(values), "total_U": sum(values)}
        for key, values in sorted(grouped.items())
    }


def _measurement_aggregates(
    joined: Sequence[tuple[ViewItem, LabelItem]],
    outputs: Mapping[str, Sequence[Sequence[int]]],
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    labels = tuple(label for _view, label in joined)
    aggregates: dict[str, Any] = {}
    utilities: dict[str, tuple[int, ...]] = {}
    for arm, rows in outputs.items():
        global_summary, values = _arm_aggregate(joined, rows)
        utilities[arm] = values
        aggregates[arm] = {
            "global": global_summary,
            "by_edge_family": _stratum_summary(labels, values, "edge_family"),
            "by_family_id": _stratum_summary(labels, values, "family_id"),
            "by_family_role": _stratum_summary(labels, values, "family_role"),
            "by_polarity": _stratum_summary(labels, values, "polarity"),
            "by_template_split": _stratum_summary(labels, values, "template_split"),
        }
    return aggregates, utilities


def _matched_specificity_deltas(
    labels: Sequence[LabelItem],
    full: Sequence[int],
    ablated: Sequence[int],
) -> tuple[int, ...]:
    grouped: dict[str, list[tuple[LabelItem, int]]] = defaultdict(list)
    for label, full_u, ablated_u in zip(labels, full, ablated):
        grouped[label.pair_key].append((label, full_u - ablated_u))
    if len(grouped) != 32:
        raise SyntheticCausalRunnerError("matched pair count drifted")
    contrasts: list[int] = []
    for pair_key in sorted(grouped):
        pair = grouped[pair_key]
        if len(pair) != 2 or {row.polarity for row, _value in pair} != {
            grammar.POSITIVE,
            grammar.NEGATIVE,
        }:
            raise SyntheticCausalRunnerError("matched polarity pair drifted")
        values = {row.polarity: value for row, value in pair}
        positive = next(row for row, _value in pair if row.polarity == grammar.POSITIVE)
        negative = next(row for row, _value in pair if row.polarity == grammar.NEGATIVE)
        if (
            positive.edge_family != negative.edge_family
            or positive.family_slot != negative.family_slot
            or positive.matching_signature_sha256 != negative.matching_signature_sha256
            or positive.structural_draw_sha256 != negative.structural_draw_sha256
        ):
            raise SyntheticCausalRunnerError("matched structural signature drifted")
        contrasts.append(values[grammar.POSITIVE] - values[grammar.NEGATIVE])
    return tuple(contrasts)


def run_measurement(
    view: ViewBlock,
    *,
    real_recipe_id: str,
    permuted_recipe_id: str,
    fixed_e00_recipe_id: str,
    label_loader: Callable[[], LabelBlock],
    encoder: EncoderProtocol,
    runtime: OfficialRuntimeProtocol,
    work_root: Path,
    action_seal_path: Path | None = None,
) -> MeasurementOutcome:
    if view.block not in {"A_hold", "M_search"}:
        raise SyntheticCausalRunnerError("measurement block drifted")
    if real_recipe_id not in RECIPE_IDS or real_recipe_id == R0:
        raise SyntheticCausalRunnerError("real measurement recipe is not identifiable")
    if permuted_recipe_id not in RECIPE_IDS or fixed_e00_recipe_id not in RECIPE_IDS:
        raise SyntheticCausalRunnerError("evaluator-control recipe drifted")
    tensors = precompute_local_blocks((view,), encoder)
    action_map, runtime_hash = _run_official_wave(
        (view,),
        tensors,
        runtime,
        work_root,
        recipes=(real_recipe_id, permuted_recipe_id, fixed_e00_recipe_id),
    )
    actions = tuple(action_map[view.block])
    if any(not isinstance(row, MeasurementActions) for row in actions):
        raise SyntheticCausalRunnerError("measurement action type drifted")
    typed_actions = tuple(row for row in actions if isinstance(row, MeasurementActions))
    action_rows = [
        {
            "ordinal": row.view.ordinal,
            "opaque_view_sha256": row.view.opaque_view_sha256,
            "local_tensor_sha256": row.local_tensor_sha256,
            "raw": list(row.raw_top5),
            "official": list(row.official_top5),
            "modes": {
                mode: list(row.outputs_by_mode[mode])
                for mode in grammar.ABLATION_MODES
            },
            "permuted_evaluator": list(row.permuted_evaluator_top5),
            "E00": list(row.fixed_e00_top5),
            "scan_hashes": dict(row.scan_hashes_by_mode),
        }
        for row in typed_actions
    ]
    action_table_hash = semantic_hash(action_rows)
    action_seal_hash, action_seal_file_hash = _persist_action_seal(
        path=action_seal_path,
        stage=view.block,
        action_table_sha256=action_table_hash,
        action_rows=action_rows,
        runtime_binding_sha256=runtime_hash,
        view_file_sha256s={view.block: view.file_sha256},
        recipe_ids={
            "real_recipe_id": real_recipe_id,
            "permuted_recipe_id": permuted_recipe_id,
            "fixed_e00_recipe_id": fixed_e00_recipe_id,
        },
    )
    # First label access follows every retrieval, graph-only action, and the
    # official postflight.  M callers must separately enforce promotion.
    labels = label_loader()
    joined = _join(view, labels)
    outputs = {
        "canonical_RAW": tuple(row.raw_top5 for row in typed_actions),
        "official_HippoRAG": tuple(row.official_top5 for row in typed_actions),
        "Agent_full": tuple(
            row.outputs_by_mode[grammar.FULL_GRAPH] for row in typed_actions
        ),
        "Agent_drop_designated": tuple(
            row.outputs_by_mode[grammar.DROP_DESIGNATED] for row in typed_actions
        ),
        "Agent_wrong_type": tuple(
            row.outputs_by_mode[grammar.WRONG_TYPE] for row in typed_actions
        ),
        "Agent_endpoint_permuted": tuple(
            row.outputs_by_mode[grammar.ENDPOINT_PERMUTED] for row in typed_actions
        ),
        "Agent_permuted_evaluator": tuple(
            row.permuted_evaluator_top5 for row in typed_actions
        ),
        "Agent_E00": tuple(row.fixed_e00_top5 for row in typed_actions),
    }
    aggregates, utilities = _measurement_aggregates(joined, outputs)
    labels_only = tuple(label for _item, label in joined)
    primary_deltas = tuple(
        agent - official
        for agent, official in zip(
            utilities["Agent_full"], utilities["official_HippoRAG"]
        )
    )
    mechanism_deltas = {
        "full_minus_drop_designated_positive_minus_negative": _matched_specificity_deltas(
            labels_only,
            utilities["Agent_full"],
            utilities["Agent_drop_designated"],
        ),
        "full_minus_wrong_type_positive_minus_negative": _matched_specificity_deltas(
            labels_only,
            utilities["Agent_full"],
            utilities["Agent_wrong_type"],
        ),
        "full_minus_endpoint_permuted_positive_minus_negative": _matched_specificity_deltas(
            labels_only,
            utilities["Agent_full"],
            utilities["Agent_endpoint_permuted"],
        ),
    }
    evaluator_deltas = tuple(
        real - permuted
        for real, permuted in zip(
            utilities["Agent_full"], utilities["Agent_permuted_evaluator"]
        )
    )
    all_delta_rows: dict[str, tuple[int, ...]] = {
        "Agent_full_minus_official": primary_deltas,
        "real_evaluator_recipe_minus_permuted_evaluator_recipe": evaluator_deltas,
        **mechanism_deltas,
    }
    primary_reference = _reference_tail(primary_deltas)
    if view.block == "A_hold":
        primary_reference = {
            **primary_reference,
            "protocol_promoted": bool(
                primary_reference[
                    "positive_and_reference_tail_at_or_below_threshold"
                ]
            ),
            "sole_protocol_promotion_criterion": True,
        }
    return MeasurementOutcome(
        block=view.block,
        selected_recipe_id=real_recipe_id,
        permuted_recipe_id=permuted_recipe_id,
        fixed_e00_recipe_id=fixed_e00_recipe_id,
        primary_reference_test=primary_reference,
        mechanism_reference_tests={
            key: _reference_tail(value) for key, value in mechanism_deltas.items()
        },
        evaluator_reference_tests={
            "real_recipe_minus_permuted_recipe": _reference_tail(evaluator_deltas)
        },
        aggregates=aggregates,
        delta_hashes={key: semantic_hash(list(value)) for key, value in all_delta_rows.items()},
        action_table_sha256=action_table_hash,
        action_seal_sha256=action_seal_hash,
        action_seal_file_sha256=action_seal_file_hash,
        runtime_binding_sha256=runtime_hash,
        view_file_sha256=view.file_sha256,
        label_file_sha256=labels.file_sha256,
    )


def run_m_if_authorized(
    *,
    authorized: bool,
    view_loader: Callable[[], ViewBlock],
    label_loader: Callable[[], LabelBlock],
    real_recipe_id: str,
    permuted_recipe_id: str,
    fixed_e00_recipe_id: str,
    encoder: EncoderProtocol,
    runtime: OfficialRuntimeProtocol,
    work_root: Path,
) -> MeasurementOutcome:
    if authorized is not True:
        raise SyntheticCausalRunnerError("M_search remains sealed after nonpromotion")
    view = view_loader()
    if view.block != "M_search":
        raise SyntheticCausalRunnerError("M_search loader returned another block")
    return run_measurement(
        view,
        real_recipe_id=real_recipe_id,
        permuted_recipe_id=permuted_recipe_id,
        fixed_e00_recipe_id=fixed_e00_recipe_id,
        label_loader=label_loader,
        encoder=encoder,
        runtime=runtime,
        work_root=work_root,
    )


def _receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(body), "receipt_sha256": semantic_hash(dict(body))}


def formation_public_receipt(outcome: FormationOutcome) -> dict[str, Any]:
    return _receipt(
        {
            "schema": f"{VERSION}_formation_public_receipt",
            "version": VERSION,
            "stage": "formation",
            "status": (
                "formation_complete_identifiable"
                if outcome.identifiable_transition
                else "terminal_unidentifiable_transition"
            ),
            "design_sha256": DESIGN_SHA256,
            "design_file_sha256": DESIGN_FILE_SHA256,
            "grammar_sha256": GRAMMAR_SHA256,
            "graph_core_sha256": GRAPH_CORE_SHA256,
            "A_form_count": BLOCK_SIZE,
            "F_search_count": BLOCK_SIZE,
            "official_concurrency_cap": OFFICIAL_CONCURRENCY_CAP,
            "local_concurrency_cap": LOCAL_CONCURRENCY_CAP,
            "real_evaluator_id": outcome.real_evaluator_id,
            "permuted_evaluator_id": outcome.permuted_evaluator_id,
            "fixed_e00_evaluator_id": outcome.fixed_e00_evaluator_id,
            "real_recipe_id": outcome.real_recipe_id,
            "permuted_recipe_id": outcome.permuted_recipe_id,
            "fixed_e00_recipe_id": outcome.fixed_e00_recipe_id,
            "identifiable_transition": outcome.identifiable_transition,
            "evaluator_control_same_recipe": outcome.evaluator_control_same_recipe,
            "evaluator_derangement_effective_same_gold_vector_count": (
                outcome.effective_same_gold_vector_count
            ),
            "same_gold_vector_count_is_descriptive_not_a_gate_or_retry_trigger": True,
            "A_form_arm_aggregates": dict(outcome.arm_aggregates),
            "A_form_view_file_sha256": outcome.a_view_file_sha256,
            "A_form_label_file_sha256": outcome.a_label_file_sha256,
            "F_search_view_file_sha256": outcome.f_view_file_sha256,
            "F_search_labels_created_or_opened": False,
            "action_table_sha256": outcome.action_table_sha256,
            "action_seal_sha256": outcome.action_seal_sha256,
            "action_seal_file_sha256": outcome.action_seal_file_sha256,
            "runtime_binding_sha256": outcome.runtime_binding_sha256,
            "A_hold_authorized": outcome.identifiable_transition,
            "item_rows_or_item_commitments_persisted_publicly": False,
        }
    )


def measurement_public_receipt(outcome: MeasurementOutcome) -> dict[str, Any]:
    promoted = bool(outcome.primary_reference_test.get("protocol_promoted"))
    positive_net = bool(outcome.primary_reference_test.get("positive_observed_net"))
    return _receipt(
        {
            "schema": f"{VERSION}_{outcome.block}_public_receipt",
            "version": VERSION,
            "stage": outcome.block,
            "status": (
                "promoted" if outcome.block == "A_hold" and promoted
                else "valid_nonpromotion" if outcome.block == "A_hold"
                else "terminal_positive_net" if positive_net
                else "terminal_nonpositive_net"
            ),
            "design_sha256": DESIGN_SHA256,
            "design_file_sha256": DESIGN_FILE_SHA256,
            "grammar_sha256": GRAMMAR_SHA256,
            "graph_core_sha256": GRAPH_CORE_SHA256,
            "item_count": BLOCK_SIZE,
            "official_concurrency_cap": OFFICIAL_CONCURRENCY_CAP,
            "local_concurrency_cap": LOCAL_CONCURRENCY_CAP,
            "real_recipe_id": outcome.selected_recipe_id,
            "permuted_recipe_id": outcome.permuted_recipe_id,
            "fixed_e00_recipe_id": outcome.fixed_e00_recipe_id,
            "primary_reference_test": dict(outcome.primary_reference_test),
            "mechanism_reference_tests": {
                key: dict(value) for key, value in outcome.mechanism_reference_tests.items()
            },
            "evaluator_reference_tests": {
                key: dict(value) for key, value in outcome.evaluator_reference_tests.items()
            },
            "aggregate_only_arm_and_family_results": dict(outcome.aggregates),
            "delta_vector_sha256": dict(outcome.delta_hashes),
            "action_table_sha256": outcome.action_table_sha256,
            "action_seal_sha256": outcome.action_seal_sha256,
            "action_seal_file_sha256": outcome.action_seal_file_sha256,
            "runtime_binding_sha256": outcome.runtime_binding_sha256,
            "view_file_sha256": outcome.view_file_sha256,
            "label_file_sha256": outcome.label_file_sha256,
            "M_search_authorized": promoted if outcome.block == "A_hold" else False,
            "reference_tail_is_design_based_randomization_p_value": False,
            "item_rows_or_item_commitments_persisted_publicly": False,
        }
    )


def _load_committed_stage_receipt(
    project_root: Path, stage: str
) -> dict[str, Any]:
    if stage not in STAGE_RECEIPT_PATHS:
        raise SyntheticCausalRunnerError("stage receipt identity drifted")
    relative = STAGE_RECEIPT_PATHS[stage]
    path = _assert_no_symlink_components(
        project_root.resolve() / relative, f"committed {stage} receipt"
    )
    if not path.is_file() or path.is_symlink():
        raise SyntheticCausalRunnerError(f"committed {stage} receipt is absent")
    raw = path.read_bytes()
    try:
        committed = _committed_bytes(project_root.resolve(), relative)
    except Exception as exc:
        raise SyntheticCausalRunnerError(f"{stage} receipt is not committed") from exc
    if committed != raw:
        raise SyntheticCausalRunnerError(f"{stage} receipt is not current-HEAD committed")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticCausalRunnerError(f"{stage} receipt is invalid JSON") from exc
    expected_schema = f"{VERSION}_{stage}_public_receipt"
    if (
        not isinstance(payload, dict)
        or payload.get("stage") != stage
        or payload.get("schema") != expected_schema
    ):
        raise SyntheticCausalRunnerError(f"{stage} receipt schema drifted")
    body = dict(payload)
    declared = body.pop("receipt_sha256", None)
    if not isinstance(declared, str) or semantic_hash(body) != declared:
        raise SyntheticCausalRunnerError(f"{stage} receipt self-hash drifted")
    return payload


def _historical_bytes(root: Path, commit: str, relative: Path) -> bytes:
    prefix = _git_project_prefix(root)
    try:
        return _git(root, "show", f"{commit}:{prefix}{relative.as_posix()}")
    except Exception as exc:
        raise SyntheticCausalRunnerError("historical receipt binding is absent") from exc


def _validate_stage_marker_and_seal(
    *,
    root: Path,
    stage: str,
    receipt: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> None:
    paths = _canonical_stage_paths(root, stage)
    marker, marker_file_hash = _read_private(paths["marker"], f"{stage} marker")
    marker_body = dict(marker)
    marker_hash = marker_body.pop("marker_sha256", None)
    if (
        not isinstance(marker_hash, str)
        or semantic_hash(marker_body) != marker_hash
        or marker.get("schema") != f"{VERSION}_{stage}_attempt_marker"
        or marker.get("stage") != stage
        or marker.get("status") != "sole_stage_attempt_consumed"
        or marker.get("attempt_count") != 1
        or marker.get("implementation_freeze_sha256")
        != freeze.get("implementation_freeze_sha256")
        or marker.get("acquisition_receipt_sha256")
        != acquisition.get("receipt_sha256")
        or receipt.get("stage_attempt_marker_sha256") != marker_hash
        or receipt.get("stage_attempt_marker_file_sha256") != marker_file_hash
        or receipt.get("invocation_HEAD") != marker.get("actual_HEAD")
        or receipt.get("parent_receipt_sha256")
        != marker.get("parent_receipt_sha256")
    ):
        raise SyntheticCausalRunnerError(f"{stage} marker chain drifted")
    invocation = str(receipt.get("invocation_HEAD"))
    try:
        _git(root, "merge-base", "--is-ancestor", invocation, "HEAD")
    except Exception as exc:
        raise SyntheticCausalRunnerError(f"{stage} invocation HEAD is not an ancestor") from exc
    if _historical_bytes(root, invocation, ACQUISITION_RECEIPT_RELATIVE_PATH) != (
        root / ACQUISITION_RECEIPT_RELATIVE_PATH
    ).read_bytes():
        raise SyntheticCausalRunnerError(f"{stage} invocation acquisition receipt drifted")
    if stage != "formation":
        parent_stage = "formation" if stage == "A_hold" else "A_hold"
        parent_relative = STAGE_RECEIPT_PATHS[parent_stage]
        if _historical_bytes(root, invocation, parent_relative) != (
            root / parent_relative
        ).read_bytes():
            raise SyntheticCausalRunnerError(f"{stage} invocation parent receipt drifted")
    seal, seal_file_hash = _read_private(paths["seal"], f"{stage} action seal")
    seal_body = dict(seal)
    seal_hash = seal_body.pop("action_seal_sha256", None)
    if (
        not isinstance(seal_hash, str)
        or semantic_hash(seal_body) != seal_hash
        or seal.get("schema") != f"{VERSION}_{stage}_label_free_action_seal"
        or seal.get("stage") != stage
        or seal.get("status")
        != "all_required_label_free_actions_and_official_postflight_terminal"
        or seal.get("labels_opened_before_seal") is not False
        or seal.get("F_search_labels_created_or_opened") is not False
        or seal.get("minilm_asset_sha256") != MINILM_ASSET_SHA256
        or not isinstance(seal.get("action_rows"), list)
        or semantic_hash(seal.get("action_rows")) != seal.get("action_table_sha256")
        or seal.get("action_table_sha256") != receipt.get("action_table_sha256")
        or seal.get("runtime_binding_sha256") != receipt.get("runtime_binding_sha256")
        or receipt.get("action_seal_sha256") != seal_hash
        or receipt.get("action_seal_file_sha256") != seal_file_hash
    ):
        raise SyntheticCausalRunnerError(f"{stage} action seal chain drifted")
    expected_views = (
        {
            "A_form": receipt.get("A_form_view_file_sha256"),
            "F_search": receipt.get("F_search_view_file_sha256"),
        }
        if stage == "formation"
        else {stage: receipt.get("view_file_sha256")}
    )
    expected_recipes = (
        None
        if stage == "formation"
        else {
            "real_recipe_id": receipt.get("real_recipe_id"),
            "permuted_recipe_id": receipt.get("permuted_recipe_id"),
            "fixed_e00_recipe_id": receipt.get("fixed_e00_recipe_id"),
        }
    )
    if seal.get("view_file_sha256s") != expected_views or seal.get(
        "recipe_ids"
    ) != expected_recipes:
        raise SyntheticCausalRunnerError(f"{stage} action seal payload drifted")


def _load_validated_stage_receipt(
    *,
    root: Path,
    stage: str,
    acquisition: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _load_committed_stage_receipt(root, stage)
    acquisition_file_hash = _sha256_file(root / ACQUISITION_RECEIPT_RELATIVE_PATH)
    if (
        receipt.get("design_sha256") != DESIGN_SHA256
        or receipt.get("design_file_sha256") != DESIGN_FILE_SHA256
        or receipt.get("grammar_sha256") != GRAMMAR_SHA256
        or receipt.get("graph_core_sha256") != GRAPH_CORE_SHA256
        or receipt.get("implementation_freeze_sha256")
        != freeze.get("implementation_freeze_sha256")
        or receipt.get("acquisition_receipt_sha256")
        != acquisition.get("receipt_sha256")
        or receipt.get("acquisition_receipt_file_sha256") != acquisition_file_hash
    ):
        raise SyntheticCausalRunnerError(f"{stage} fixed receipt binding drifted")
    _validate_stage_marker_and_seal(
        root=root,
        stage=stage,
        receipt=receipt,
        acquisition=acquisition,
        freeze=freeze,
    )
    if stage == "formation":
        if receipt.get("parent_receipt_sha256") is not None or receipt.get(
            "parent_receipt_file_sha256"
        ) is not None:
            raise SyntheticCausalRunnerError("formation parent binding drifted")
        return receipt
    parent_stage = "formation" if stage == "A_hold" else "A_hold"
    parent = _load_validated_stage_receipt(
        root=root, stage=parent_stage, acquisition=acquisition, freeze=freeze
    )
    parent_path = root / STAGE_RECEIPT_PATHS[parent_stage]
    if receipt.get("parent_receipt_sha256") != parent.get("receipt_sha256") or (
        receipt.get("parent_receipt_file_sha256") != _sha256_file(parent_path)
    ):
        raise SyntheticCausalRunnerError(f"{stage} parent file binding drifted")
    for field in ("real_recipe_id", "permuted_recipe_id", "fixed_e00_recipe_id"):
        if receipt.get(field) != parent.get(field):
            raise SyntheticCausalRunnerError(f"{stage} recipe inheritance drifted")
    return receipt


def _canonical_stage_paths(project_root: Path, stage: str) -> dict[str, Path]:
    root = project_root.resolve()
    stage_root = root / STAGE_ROOT_RELATIVE_PATH / stage
    return {
        "root": stage_root,
        "marker": stage_root / f"{stage}.attempt.marker",
        "work": stage_root / f"{stage}.work",
        "seal": stage_root / f"{stage}.action.seal.json",
        "receipt": root / STAGE_RECEIPT_PATHS[stage],
        "failure": root
        / f"manifests/synthetic_typed_graph_causal_{stage}_failure_v1.json",
    }


def _consume_stage_marker(
    *,
    project_root: Path,
    stage: str,
    actual_head: str,
    implementation_freeze_sha256: str,
    acquisition_receipt_sha256: str,
    parent_receipt_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Path], str]:
    paths = _canonical_stage_paths(project_root, stage)
    if any(path.exists() or path.is_symlink() for key, path in paths.items() if key != "root"):
        raise SyntheticCausalRunnerError(f"canonical {stage} output already exists")
    marker = _receipt(
        {
            "schema": f"{VERSION}_{stage}_attempt_marker",
            "version": VERSION,
            "stage": stage,
            "status": "sole_stage_attempt_consumed",
            "actual_HEAD": actual_head,
            "implementation_freeze_sha256": implementation_freeze_sha256,
            "acquisition_receipt_sha256": acquisition_receipt_sha256,
            "parent_receipt_sha256": parent_receipt_sha256,
            "attempt_count": 1,
        }
    )
    marker["marker_sha256"] = marker.pop("receipt_sha256")
    marker_file_hash = _write_json_exclusive(paths["marker"], marker, PRIVATE_MODE)
    return marker, paths, marker_file_hash


def _persist_stage_failure(
    *,
    stage: str,
    path: Path,
    marker: Mapping[str, Any],
    marker_file_sha256: str,
    invocation_head: str,
    implementation_freeze_sha256: str,
    acquisition_receipt: Mapping[str, Any],
    acquisition_receipt_file_sha256: str,
    parent_receipt: Mapping[str, Any] | None,
    parent_receipt_file_sha256: str | None,
    action_seal_path: Path,
    exc: Exception,
) -> None:
    action_seal_sha256: str | None = None
    action_seal_file_sha256: str | None = None
    if action_seal_path.is_file() and not action_seal_path.is_symlink():
        seal, action_seal_file_sha256 = _read_private(
            action_seal_path, f"{stage} failed-stage action seal"
        )
        action_seal_sha256 = str(seal.get("action_seal_sha256"))
    failure = _receipt(
        {
            "schema": f"{VERSION}_stage_failure_receipt",
            "version": VERSION,
            "stage": stage,
            "status": "terminal_infrastructure_or_implementation_invalid_no_replay",
            "marker_sha256": marker["marker_sha256"],
            "stage_attempt_marker_file_sha256": marker_file_sha256,
            "invocation_HEAD": invocation_head,
            "implementation_freeze_sha256": implementation_freeze_sha256,
            "design_sha256": DESIGN_SHA256,
            "design_file_sha256": DESIGN_FILE_SHA256,
            "grammar_sha256": GRAMMAR_SHA256,
            "graph_core_sha256": GRAPH_CORE_SHA256,
            "acquisition_receipt_sha256": acquisition_receipt["receipt_sha256"],
            "acquisition_receipt_file_sha256": acquisition_receipt_file_sha256,
            "parent_receipt_sha256": (
                parent_receipt["receipt_sha256"] if parent_receipt is not None else None
            ),
            "parent_receipt_file_sha256": parent_receipt_file_sha256,
            "action_seal_sha256": action_seal_sha256,
            "action_seal_file_sha256": action_seal_file_sha256,
            "failure_class": type(exc).__name__,
            "exception_message_private_path_item_or_label_persisted_publicly": False,
            "item_rows_or_item_commitments_persisted_publicly": False,
            "retry_replay_replacement_or_backup_stage_authorized": False,
        }
    )
    if not path.exists():
        _write_json_exclusive(path, failure, PUBLIC_MODE)


def _chain_receipt(
    base: Mapping[str, Any],
    *,
    marker: Mapping[str, Any],
    marker_file_sha256: str,
    invocation_head: str,
    implementation_freeze_sha256: str,
    acquisition_receipt: Mapping[str, Any],
    acquisition_receipt_file_sha256: str,
    parent_receipt: Mapping[str, Any] | None,
    parent_receipt_file_sha256: str | None,
) -> dict[str, Any]:
    body = dict(base)
    body.pop("receipt_sha256", None)
    body.update(
        {
            "stage_attempt_marker_sha256": marker["marker_sha256"],
            "stage_attempt_marker_file_sha256": marker_file_sha256,
            "invocation_HEAD": invocation_head,
            "implementation_freeze_sha256": implementation_freeze_sha256,
            "acquisition_receipt_sha256": acquisition_receipt["receipt_sha256"],
            "acquisition_receipt_file_sha256": acquisition_receipt_file_sha256,
            "parent_receipt_sha256": (
                parent_receipt["receipt_sha256"] if parent_receipt is not None else None
            ),
            "parent_receipt_file_sha256": parent_receipt_file_sha256,
            "receipt_must_be_committed_before_next_stage": True,
        }
    )
    return _receipt(body)


def run_canonical_stage(
    *,
    project_root: Path,
    stage: str,
    encoder: EncoderProtocol,
    runtime: OfficialRuntimeProtocol,
) -> dict[str, Any]:
    """Consume exactly one canonical formation, A-hold, or family-out stage.

    Only public committed receipts and the implementation freeze are read
    before the marker.  Views and late labels are opened through canonical
    acquisition hashes after the marker.  Each returned receipt is persisted
    but must be committed by the caller before the next stage can start.
    """

    if _FORMAL_ENTRY_ACTIVE is not True:
        raise SyntheticCausalRunnerError(
            "canonical stage may only be consumed by the formal CLI entry"
        )
    if not isinstance(encoder, OfflineMiniLMEncoder) or not isinstance(
        runtime, PreparedFormalRuntimeV2
    ):
        raise SyntheticCausalRunnerError("canonical stage resources are not attested formal types")
    if stage not in {"formation", "A_hold", "M_search"}:
        raise SyntheticCausalRunnerError("canonical stage is unknown")
    root = project_root.resolve(strict=True)
    freeze, actual_head = verify_implementation_freeze(root)
    acquisition = load_committed_acquisition_receipt(root)
    acquisition_file_hash = _sha256_file(root / ACQUISITION_RECEIPT_RELATIVE_PATH)
    parent: dict[str, Any] | None = None
    parent_file_hash: str | None = None
    if stage == "A_hold":
        parent = _load_validated_stage_receipt(
            root=root, stage="formation", acquisition=acquisition, freeze=freeze
        )
        if (
            parent.get("status") != "formation_complete_identifiable"
            or parent.get("A_hold_authorized") is not True
            or parent.get("acquisition_receipt_sha256") != acquisition["receipt_sha256"]
        ):
            raise SyntheticCausalRunnerError("committed formation does not authorize A_hold")
        parent_file_hash = _sha256_file(root / FORMATION_RECEIPT_RELATIVE_PATH)
    elif stage == "M_search":
        formation = _load_validated_stage_receipt(
            root=root, stage="formation", acquisition=acquisition, freeze=freeze
        )
        parent = _load_validated_stage_receipt(
            root=root, stage="A_hold", acquisition=acquisition, freeze=freeze
        )
        recipe_fields = ("real_recipe_id", "permuted_recipe_id", "fixed_e00_recipe_id")
        if (
            parent.get("status") != "promoted"
            or parent.get("M_search_authorized") is not True
            or parent.get("acquisition_receipt_sha256") != acquisition["receipt_sha256"]
            or parent.get("parent_receipt_sha256") != formation["receipt_sha256"]
            or any(parent.get(field) != formation.get(field) for field in recipe_fields)
        ):
            raise SyntheticCausalRunnerError("committed A_hold does not authorize M_search")
        parent_file_hash = _sha256_file(root / A_HOLD_RECEIPT_RELATIVE_PATH)
    marker, paths, marker_file_hash = _consume_stage_marker(
        project_root=root,
        stage=stage,
        actual_head=actual_head,
        implementation_freeze_sha256=str(freeze["implementation_freeze_sha256"]),
        acquisition_receipt_sha256=str(acquisition["receipt_sha256"]),
        parent_receipt_sha256=(
            str(parent["receipt_sha256"]) if parent is not None else None
        ),
    )
    try:
        if stage == "formation":
            outcome = run_formal_formation(
                project_root=root,
                encoder=encoder,
                runtime=runtime,
                work_root=paths["work"],
                action_seal_path=paths["seal"],
            )
            base = formation_public_receipt(outcome)
        else:
            assert parent is not None
            block = stage
            view = load_canonical_view_block(root, block)
            outcome = run_measurement(
                view,
                real_recipe_id=str(parent["real_recipe_id"]),
                permuted_recipe_id=str(parent["permuted_recipe_id"]),
                fixed_e00_recipe_id=str(parent["fixed_e00_recipe_id"]),
                label_loader=lambda: load_canonical_label_block(root, block),
                encoder=encoder,
                runtime=runtime,
                work_root=paths["work"],
                action_seal_path=paths["seal"],
            )
            base = measurement_public_receipt(outcome)
        receipt = _chain_receipt(
            base,
            marker=marker,
            marker_file_sha256=marker_file_hash,
            invocation_head=actual_head,
            implementation_freeze_sha256=str(freeze["implementation_freeze_sha256"]),
            acquisition_receipt=acquisition,
            acquisition_receipt_file_sha256=acquisition_file_hash,
            parent_receipt=parent,
            parent_receipt_file_sha256=parent_file_hash,
        )
        _write_json_exclusive(paths["receipt"], receipt, PUBLIC_MODE)
        return receipt
    except Exception as exc:
        _persist_stage_failure(
            stage=stage,
            path=paths["failure"],
            marker=marker,
            marker_file_sha256=marker_file_hash,
            invocation_head=actual_head,
            implementation_freeze_sha256=str(freeze["implementation_freeze_sha256"]),
            acquisition_receipt=acquisition,
            acquisition_receipt_file_sha256=acquisition_file_hash,
            parent_receipt=parent,
            parent_receipt_file_sha256=parent_file_hash,
            action_seal_path=paths["seal"],
            exc=exc,
        )
        raise


def _prepare_formal_resources(
    *,
    project_root: Path,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
) -> tuple[OfflineMiniLMEncoder, PreparedFormalRuntimeV2]:
    root = project_root.resolve(strict=True)
    # These four repository assets are canonical and cannot be redirected by
    # CLI flags.  The three external runtime asset paths are verified against
    # the committed filesystem attestation by prepare_formal_runtime_v2.
    encoder = OfflineMiniLMEncoder(
        asset_manifest_path=root / MINILM_MANIFEST_RELATIVE_PATH,
        model_root=root / MINILM_MODEL_ROOT_RELATIVE_PATH,
        run_canary=True,
    )
    runtime = prepare_formal_runtime_v2(
        project_root=root,
        attestation_receipt_path=root / OFFICIAL_ATTESTATION_RELATIVE_PATH,
        base_binding_receipt_path=root / OFFICIAL_BASE_RECEIPT_RELATIVE_PATH,
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
    )
    return encoder, runtime


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("formation", "A_hold", "M_search"))
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--runtime-python", required=True, type=Path)
    parser.add_argument("--local-llm-model", required=True, type=Path)
    parser.add_argument("--local-embedding-model", required=True, type=Path)
    arguments = parser.parse_args(argv)
    encoder, runtime = _prepare_formal_resources(
        project_root=arguments.project_root,
        runtime_python=arguments.runtime_python,
        local_llm_model=arguments.local_llm_model,
        local_embedding_model=arguments.local_embedding_model,
    )
    global _FORMAL_ENTRY_ACTIVE
    if _FORMAL_ENTRY_ACTIVE:
        raise SyntheticCausalRunnerError("formal entry is already active")
    _FORMAL_ENTRY_ACTIVE = True
    try:
        receipt = run_canonical_stage(
            project_root=arguments.project_root,
            stage=arguments.stage,
            encoder=encoder,
            runtime=runtime,
        )
    finally:
        _FORMAL_ENTRY_ACTIVE = False
    print(
        json.dumps(
            {
                "stage": receipt["stage"],
                "status": receipt["status"],
                "receipt_sha256": receipt["receipt_sha256"],
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


__all__ = [
    "EncoderProtocol",
    "FormationOutcome",
    "LabelBlock",
    "LabelItem",
    "LOCAL_CONCURRENCY_CAP",
    "MeasurementOutcome",
    "OFFICIAL_CONCURRENCY_CAP",
    "OfficialRuntimeProtocol",
    "REFERENCE_TAIL_INTERPRETATION",
    "SyntheticCausalRunnerError",
    "VERSION",
    "ViewBlock",
    "ViewItem",
    "formation_public_receipt",
    "load_label_block",
    "load_canonical_label_block",
    "load_canonical_view_block",
    "load_view_block",
    "measurement_public_receipt",
    "precompute_local_blocks",
    "run_formation",
    "run_formal_formation",
    "run_m_if_authorized",
    "run_measurement",
]


if __name__ == "__main__":
    raise SystemExit(main())
