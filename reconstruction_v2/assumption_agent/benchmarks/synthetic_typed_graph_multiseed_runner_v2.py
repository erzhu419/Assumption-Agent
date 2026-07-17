"""Chunk-repaired public diagnostic and formal eight-seed replication runner.

The sole implementation change from the terminal-invalid v1 study is the
predeclared ordered MiniLM schedule: the 512 items are flattened in canonical
order and encoded in exactly two contiguous 8,448-text calls.  Both the public
integration diagnostic and the fresh formal study execute the same three-arm
1,536-action wave.  The diagnostic projects only label-free fields from the
already-public v1 terminal cohort and computes no evaluation quantity.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat
from statistics import median
import threading
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
from .synthetic_typed_graph_multiseed_acquisition_v2 import (
    ACQUISITION_RECEIPT_RELATIVE_PATH,
    ACTION_PACK_RELATIVE_PATH,
    DESIGN_FILE_SHA256,
    DESIGN_SHA256,
    LABEL_PACK_RELATIVE_PATH,
    PRIVATE_MODE,
    PUBLIC_MODE,
    _assert_no_symlink_components,
    _committed_bytes,
    _git,
    _write_json_exclusive,
    canonical_bytes,
    semantic_hash,
    verify_implementation_freeze,
)


VERSION = "synthetic_typed_graph_multiseed_runner_v2"
DESIGN_VERSION = "synthetic_typed_graph_multiseed_replication_v2"
BLOCK = "A_hold"
SEED_COUNT = 8
ITEMS_PER_SEED = 64
TOTAL_ITEMS = SEED_COUNT * ITEMS_PER_SEED
NODE_COUNT = 32
TEXTS_PER_ITEM = 1 + NODE_COUNT
CHUNK_ITEM_COUNT = 256
CHUNK_COUNT = 2
TEXTS_PER_CHUNK = CHUNK_ITEM_COUNT * TEXTS_PER_ITEM
TOTAL_TEXT_COUNT = TOTAL_ITEMS * TEXTS_PER_ITEM
FROZEN_MAXIMUM_TEXTS_PER_CALL = 16_384
TOP_K = 5
OFFICIAL_CONCURRENCY_CAP = 8
LOCAL_CONCURRENCY_CAP = 64
ACTION_WORK_UNITS = TOTAL_ITEMS * 3
RECIPE_ID = "R1_DEFINITION_1SWAP"
RAW_ARM = "RAW"
HIPPO_ARM = "official_HippoRAG"
HIPPORAG_ARM = HIPPO_ARM
AGENT_ARM = "Agent_R1"
ARM_IDS = (RAW_ARM, HIPPO_ARM, AGENT_ARM)

ACTION_PACK_SCHEMA = "synthetic_typed_graph_multiseed_action_pack_v2"
ACTION_ITEM_SCHEMA = "synthetic_typed_graph_multiseed_action_item_v2"
LABEL_PACK_SCHEMA = "synthetic_typed_graph_multiseed_label_pack_v2"
LABEL_ITEM_SCHEMA = "synthetic_typed_graph_multiseed_label_item_v2"
ACQUISITION_SCHEMA = "synthetic_typed_graph_multiseed_replication_acquisition_v2"
ACQUISITION_STATUS = (
    "formal_v2_multiseed_A_hold_cohort_acquired_private_labels_separated"
)
RESULT_SCHEMA = "synthetic_typed_graph_multiseed_replication_result_v2"
SUCCESS_RESULT_STATUS = "terminal_descriptive_eight_seed_replication_complete"
FAILURE_RESULT_STATUS = "terminal_infrastructure_or_implementation_invalid_no_replay"
DIAGNOSTIC_SCHEMA = (
    "synthetic_typed_graph_multiseed_replication_integration_diagnostic_v2"
)
DIAGNOSTIC_SUCCESS_STATUS = "integration_diagnostic_complete_no_scores_or_claims"
DIAGNOSTIC_FAILURE_STATUS = (
    "terminal_integration_diagnostic_invalid_fresh_formal_not_authorized"
)

ARTIFACT_ROOT_RELATIVE_PATH = Path(
    "artifacts/synthetic_typed_graph_multiseed_replication_v2"
)
DIAGNOSTIC_ROOT_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "integration_diagnostic"
DIAGNOSTIC_MARKER_RELATIVE_PATH = DIAGNOSTIC_ROOT_RELATIVE_PATH / "attempt.marker"
DIAGNOSTIC_WORK_RELATIVE_PATH = DIAGNOSTIC_ROOT_RELATIVE_PATH / "work"
DIAGNOSTIC_ACTION_SEAL_RELATIVE_PATH = (
    DIAGNOSTIC_ROOT_RELATIVE_PATH / "action_seal.json"
)
DIAGNOSTIC_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_integration_diagnostic_v2.json"
)
RUNNER_ROOT_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "runner"
FORMAL_MARKER_RELATIVE_PATH = RUNNER_ROOT_RELATIVE_PATH / "formal.attempt.marker"
FORMAL_WORK_RELATIVE_PATH = RUNNER_ROOT_RELATIVE_PATH / "formal.work"
FORMAL_ACTION_SEAL_RELATIVE_PATH = RUNNER_ROOT_RELATIVE_PATH / "action_seal.json"
RESULT_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_result_v2.json"
)
V1_PUBLICATION_RELATIVE_PATH = Path(
    "published/synthetic_typed_graph_multiseed_replication_v1/formal_seeds_and_cohort.json"
)
MINILM_MANIFEST_RELATIVE_PATH = Path("manifests/qasper_minilm_runtime_asset_v1.json")
MINILM_MODEL_ROOT_RELATIVE_PATH = Path("artifacts/qasper_minilm_runtime_v1/model")
OFFICIAL_BASE_RECEIPT_RELATIVE_PATH = Path(
    "manifests/musique_official_hipporag_retrieve_only_binding_v1.json"
)
OFFICIAL_ATTESTATION_RELATIVE_PATH = Path(
    "manifests/musique_official_hipporag_runtime_attestation_v2.json"
)
RUNNER_MODULE_RELATIVE_PATH = Path(
    "assumption_agent/benchmarks/synthetic_typed_graph_multiseed_runner_v2.py"
)
ACQUISITION_MODULE_RELATIVE_PATH = Path(
    "assumption_agent/benchmarks/synthetic_typed_graph_multiseed_acquisition_v2.py"
)
RUNNER_TEST_RELATIVE_PATH = Path(
    "tests/test_synthetic_typed_graph_multiseed_runner_v2.py"
)
ACQUISITION_TEST_RELATIVE_PATH = Path(
    "tests/test_synthetic_typed_graph_multiseed_acquisition_v2.py"
)

CHUNK_SCHEDULE: tuple[dict[str, int], ...] = (
    {
        "chunk_index": 0,
        "first_global_ordinal_inclusive": 0,
        "first_text_index_inclusive": 0,
        "item_count": 256,
        "last_global_ordinal_inclusive": 255,
        "last_text_index_exclusive": 8448,
        "text_count": 8448,
    },
    {
        "chunk_index": 1,
        "first_global_ordinal_inclusive": 256,
        "first_text_index_inclusive": 8448,
        "item_count": 256,
        "last_global_ordinal_inclusive": 511,
        "last_text_index_exclusive": 16896,
        "text_count": 8448,
    },
)
CHUNK_SCHEDULE_SHA256 = "faf5dd2b2a45b4a2b16b8913b5e38e930e99d7ccaa3ce35218d6de38a863a635"
V1_PUBLICATION_FILE_SHA256 = (
    "7ea28c298422191456ec976ddcc22450bd7021d4f14afb6bd283ae0f6d44b6e1"
)
V1_PUBLICATION_REPRODUCIBILITY_SHA256 = (
    "f54998cef3259ac196c7d4a767cc034df657f1e5221b6da6c3eb30b52d5ba13c"
)
V1_GENERATED_ITEM_COMMITMENT_SET_SHA256 = (
    "62f57fc07dfc95aafd1cb590787aa54a326ad2843711d397813a070398447bd6"
)
V1_PUBLICATION_SCHEMA = "synthetic_typed_graph_multiseed_terminal_reproducibility_v1"
V1_PUBLICATION_STATUS = "terminal_eight_seeds_and_full_compiled_cohort_published"

_DIAGNOSTIC_ENTRY_ACTIVE = False
_FORMAL_ENTRY_ACTIVE = False


class SyntheticTypedGraphMultiseedRunnerV2Error(RuntimeError):
    """A v2 projection, chunk, action, seal, or result invariant failed."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
class ActionNode:
    span_i: int
    start: int
    end: int
    identity_text: str

    def source_span(self) -> core.SourceSpan:
        return core.SourceSpan(self.span_i, self.start, self.end, self.identity_text)


@dataclass(frozen=True)
class ActionItem:
    global_ordinal: int
    seed_index: int
    seed_ordinal: int
    question: str
    context: str
    nodes: tuple[ActionNode, ...]
    designated_edges: tuple[core.TypedEdge, ...]
    full_edges: tuple[core.TypedEdge, ...]
    action_item_sha256: str

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
class ActionPack:
    pack_sha256: str
    file_sha256: str
    item_commitment_set_sha256: str
    rows: tuple[ActionItem, ...]


@dataclass(frozen=True)
class LabelItem:
    global_ordinal: int
    seed_index: int
    seed_ordinal: int
    action_item_sha256: str
    gold_node_indices: tuple[int, ...]
    family_id: str
    family_role: str
    polarity: str
    edge_family: str
    label_item_sha256: str


@dataclass(frozen=True)
class LabelPack:
    pack_sha256: str
    file_sha256: str
    item_commitment_set_sha256: str
    rows: tuple[LabelItem, ...]


@dataclass(frozen=True)
class MiniLMChunkAudit:
    chunk_schedule_sha256: str
    observed_input_row_counts: tuple[int, int]
    observed_output_row_counts: tuple[int, int]


@dataclass(frozen=True)
class LocalTensor:
    raw_top5: tuple[int, int, int, int, int]
    query_similarities: tuple[int, ...]
    tensor_sha256: str


@dataclass(frozen=True)
class ItemActions:
    global_ordinal: int
    action_item_sha256: str
    raw_top5: tuple[int, int, int, int, int]
    official_top5: tuple[int, int, int, int, int]
    agent_top5: tuple[int, int, int, int, int]
    common_scan_sha256: str
    local_tensor_sha256: str


@dataclass(frozen=True)
class ActionWaveOutcome:
    actions: tuple[ItemActions, ...]
    runtime_binding_sha256: str
    official_postflight_receipt_sha256: str
    official_peak_concurrency_count: int
    local_peak_concurrency_count: int


@dataclass(frozen=True)
class MultiseedOutcome:
    action_pack_file_sha256: str
    action_pack_sha256: str
    action_item_commitment_set_sha256: str
    label_pack_file_sha256: str
    label_pack_sha256: str
    label_item_commitment_set_sha256: str
    runtime_binding_sha256: str
    official_postflight_receipt_sha256: str
    chunk_audit: MiniLMChunkAudit
    action_table_sha256: str
    action_seal_sha256: str
    action_seal_file_sha256: str
    official_peak_concurrency_count: int
    local_peak_concurrency_count: int
    aggregates: Mapping[str, Any]
    cluster_differences: Mapping[str, Any]


class _ConcurrencyTracker:
    def __init__(self, cap: int) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._live = 0
        self._peak = 0

    def enter(self) -> None:
        with self._lock:
            self._live += 1
            if self._live > self._cap:
                raise SyntheticTypedGraphMultiseedRunnerV2Error(
                    "action concurrency exceeded its frozen cap"
                )
            self._peak = max(self._peak, self._live)

    def exit(self) -> None:
        with self._lock:
            self._live -= 1
            if self._live < 0:
                raise SyntheticTypedGraphMultiseedRunnerV2Error(
                    "action concurrency accounting drifted"
                )

    @property
    def peak(self) -> int:
        with self._lock:
            return self._peak


def _read_json(path: Path, *, private: bool, field: str) -> tuple[dict[str, Any], str]:
    absolute = _assert_no_symlink_components(path, field)
    if not absolute.is_file() or absolute.is_symlink():
        raise SyntheticTypedGraphMultiseedRunnerV2Error(f"{field} is unavailable")
    expected_mode = PRIVATE_MODE if private else PUBLIC_MODE
    info = absolute.stat()
    if stat.S_IMODE(info.st_mode) != expected_mode or not 1 <= info.st_size <= 128 * 1024 * 1024:
        raise SyntheticTypedGraphMultiseedRunnerV2Error(f"{field} mode or size drifted")
    raw = absolute.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticTypedGraphMultiseedRunnerV2Error(f"{field} is invalid JSON") from exc
    if not isinstance(payload, dict) or raw != canonical_bytes(payload) + b"\n":
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            f"{field} root or canonical encoding drifted"
        )
    return payload, _sha256_bytes(raw)


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    body = dict(payload)
    declared = body.pop(field, None)
    if not isinstance(declared, str) or semantic_hash(body) != declared:
        raise SyntheticTypedGraphMultiseedRunnerV2Error(f"{field} drifted")
    return declared


def _typed_edge(raw: object) -> core.TypedEdge:
    if not isinstance(raw, Mapping) or set(raw) != {
        "edge_family", "left_span_i", "right_span_i"
    }:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("designated edge schema drifted")
    family = raw.get("edge_family")
    left, right = raw.get("left_span_i"), raw.get("right_span_i")
    if (
        family not in core.EDGE_FAMILIES
        or type(left) is not int
        or type(right) is not int
        or not 0 <= left < right < NODE_COUNT
    ):
        raise SyntheticTypedGraphMultiseedRunnerV2Error("designated edge content drifted")
    return core.TypedEdge(core.EDGE_FAMILY_ORDER[str(family)], left, right)


def _parse_action_item(raw: object, ordinal: int) -> ActionItem:
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema", "global_ordinal", "seed_index", "seed_ordinal", "question",
        "context", "nodes", "designated_edges", "action_item_sha256",
    }:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("action item field set drifted")
    if raw.get("schema") != ACTION_ITEM_SCHEMA:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("action item schema drifted")
    expected_seed, expected_within = divmod(ordinal, ITEMS_PER_SEED)
    if (
        raw.get("global_ordinal") != ordinal
        or raw.get("seed_index") != expected_seed
        or raw.get("seed_ordinal") != expected_within
    ):
        raise SyntheticTypedGraphMultiseedRunnerV2Error("action item coordinate drifted")
    declared = _self_hash(raw, "action_item_sha256")
    question, context = raw.get("question"), raw.get("context")
    if not isinstance(question, str) or not question or not isinstance(context, str):
        raise SyntheticTypedGraphMultiseedRunnerV2Error("action item text drifted")
    node_rows = raw.get("nodes")
    if not isinstance(node_rows, list) or len(node_rows) != NODE_COUNT:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("action node count drifted")
    nodes: list[ActionNode] = []
    previous_end = 0
    for index, node in enumerate(node_rows):
        if not isinstance(node, Mapping) or set(node) != {
            "span_i", "start", "end", "identity_text"
        }:
            raise SyntheticTypedGraphMultiseedRunnerV2Error("action node schema drifted")
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
            raise SyntheticTypedGraphMultiseedRunnerV2Error("action node content drifted")
        nodes.append(ActionNode(index, start, end, text))
        previous_end = end
    edge_rows = raw.get("designated_edges")
    if not isinstance(edge_rows, list) or not edge_rows:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("designated edge table drifted")
    designated = tuple(_typed_edge(edge) for edge in edge_rows)
    if tuple(sorted(set(designated))) != designated:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("designated edges are not canonical")
    spans = tuple(node.source_span() for node in nodes)
    full = core.build_typed_clause_graph(spans)
    if not set(designated).issubset(full):
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "designated edges are absent from the frozen full graph"
        )
    return ActionItem(
        ordinal, expected_seed, expected_within, question, context, tuple(nodes),
        designated, full, declared,
    )


def load_action_pack(path: Path) -> ActionPack:
    payload, file_sha256 = _read_json(path, private=True, field="action pack")
    declared = _self_hash(payload, "pack_sha256")
    rows = payload.get("items")
    if (
        set(payload) != {
            "schema", "version", "block", "seed_count", "item_count_per_seed",
            "total_item_count", "labels_included", "items", "pack_sha256",
        }
        or payload.get("schema") != ACTION_PACK_SCHEMA
        or payload.get("version") != DESIGN_VERSION
        or payload.get("block") != BLOCK
        or payload.get("seed_count") != SEED_COUNT
        or payload.get("item_count_per_seed") != ITEMS_PER_SEED
        or payload.get("total_item_count") != TOTAL_ITEMS
        or payload.get("labels_included") is not False
        or not isinstance(rows, list)
        or len(rows) != TOTAL_ITEMS
    ):
        raise SyntheticTypedGraphMultiseedRunnerV2Error("action pack schema drifted")
    parsed = tuple(_parse_action_item(row, ordinal) for ordinal, row in enumerate(rows))
    hashes = [row.action_item_sha256 for row in parsed]
    if len(set(hashes)) != TOTAL_ITEMS:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("action item commitments overlap")
    return ActionPack(declared, file_sha256, semantic_hash(hashes), parsed)


def _expected_family_by_seed_ordinal() -> tuple[str, ...]:
    expected: list[str] = []
    for family_id, count in grammar.family_quota(BLOCK):
        expected.extend([family_id] * count)
    if len(expected) != ITEMS_PER_SEED:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("frozen family schedule drifted")
    return tuple(expected)


def _parse_label_item(raw: object, ordinal: int) -> LabelItem:
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema", "global_ordinal", "seed_index", "seed_ordinal",
        "action_item_sha256", "gold_node_indices", "family_id", "family_role",
        "polarity", "edge_family", "label_item_sha256",
    }:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("label item field set drifted")
    if raw.get("schema") != LABEL_ITEM_SCHEMA:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("label item schema drifted")
    expected_seed, expected_within = divmod(ordinal, ITEMS_PER_SEED)
    if (
        raw.get("global_ordinal") != ordinal
        or raw.get("seed_index") != expected_seed
        or raw.get("seed_ordinal") != expected_within
    ):
        raise SyntheticTypedGraphMultiseedRunnerV2Error("label item coordinate drifted")
    declared = _self_hash(raw, "label_item_sha256")
    gold = raw.get("gold_node_indices")
    family_id = raw.get("family_id")
    if (
        not _is_sha256(raw.get("action_item_sha256"))
        or not isinstance(gold, list)
        or not 1 <= len(gold) <= 3
        or gold != sorted(set(gold))
        or any(type(index) is not int or not 0 <= index < NODE_COUNT for index in gold)
        or family_id not in grammar.FAMILY_BY_ID
        or family_id != _expected_family_by_seed_ordinal()[expected_within]
    ):
        raise SyntheticTypedGraphMultiseedRunnerV2Error("label item content drifted")
    family = grammar.FAMILY_BY_ID[str(family_id)]
    if (
        raw.get("family_role") != family.family_role
        or raw.get("polarity") != family.polarity
        or raw.get("edge_family") != family.edge_family
    ):
        raise SyntheticTypedGraphMultiseedRunnerV2Error("label family registry drifted")
    return LabelItem(
        ordinal, expected_seed, expected_within, str(raw["action_item_sha256"]),
        tuple(gold), str(family_id), str(raw["family_role"]), str(raw["polarity"]),
        str(raw["edge_family"]), declared,
    )


def load_label_pack(path: Path) -> LabelPack:
    payload, file_sha256 = _read_json(path, private=True, field="late label pack")
    declared = _self_hash(payload, "pack_sha256")
    rows = payload.get("items")
    if (
        set(payload) != {
            "schema", "version", "block", "seed_count", "item_count_per_seed",
            "total_item_count", "items", "pack_sha256",
        }
        or payload.get("schema") != LABEL_PACK_SCHEMA
        or payload.get("version") != DESIGN_VERSION
        or payload.get("block") != BLOCK
        or payload.get("seed_count") != SEED_COUNT
        or payload.get("item_count_per_seed") != ITEMS_PER_SEED
        or payload.get("total_item_count") != TOTAL_ITEMS
        or not isinstance(rows, list)
        or len(rows) != TOTAL_ITEMS
    ):
        raise SyntheticTypedGraphMultiseedRunnerV2Error("late label pack schema drifted")
    parsed = tuple(_parse_label_item(row, ordinal) for ordinal, row in enumerate(rows))
    hashes = [row.label_item_sha256 for row in parsed]
    if len(set(hashes)) != TOTAL_ITEMS:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("label item commitments overlap")
    return LabelPack(declared, file_sha256, semantic_hash(hashes), parsed)


def project_public_v1_label_free_rows(
    rows: object,
    *,
    source_file_sha256: str,
) -> tuple[ActionPack, str]:
    """Project the exact public v1 rows without forwarding a label field.

    The returned digest over source ``label_free_commitment_sha256`` values
    proves which public rows were projected.  Actions receive only the parsed
    :class:`ActionItem` values, whose schema cannot represent any label.
    """

    if not isinstance(rows, list) or len(rows) != TOTAL_ITEMS:
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "public v1 diagnostic row count drifted"
        )
    projected_rows: list[dict[str, Any]] = []
    source_label_free_hashes: list[str] = []
    for ordinal, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise SyntheticTypedGraphMultiseedRunnerV2Error(
                "public v1 diagnostic row is not an object"
            )
        expected_seed, expected_within = divmod(ordinal, ITEMS_PER_SEED)
        if (
            source.get("global_ordinal") != ordinal
            or source.get("seed_index") != expected_seed
            or source.get("seed_ordinal") != expected_within
        ):
            raise SyntheticTypedGraphMultiseedRunnerV2Error(
                "public v1 diagnostic row coordinate drifted"
            )
        source_commitment = source.get("label_free_commitment_sha256")
        if not _is_sha256(source_commitment):
            raise SyntheticTypedGraphMultiseedRunnerV2Error(
                "public v1 label-free commitment drifted"
            )
        nodes = source.get("nodes")
        edges = source.get("designated_edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise SyntheticTypedGraphMultiseedRunnerV2Error(
                "public v1 label-free structure drifted"
            )
        # This literal projection is the complete semantic access boundary.
        # In particular, no get/index operation names a label or outcome field.
        body = {
            "schema": ACTION_ITEM_SCHEMA,
            "global_ordinal": ordinal,
            "seed_index": expected_seed,
            "seed_ordinal": expected_within,
            "question": source.get("question"),
            "context": source.get("context"),
            "nodes": [
                {
                    "span_i": node.get("span_i"),
                    "start": node.get("start"),
                    "end": node.get("end"),
                    "identity_text": node.get("identity_text"),
                }
                if isinstance(node, Mapping)
                else None
                for node in nodes
            ],
            "designated_edges": [
                {
                    "edge_family": edge.get("edge_family"),
                    "left_span_i": edge.get("left_span_i"),
                    "right_span_i": edge.get("right_span_i"),
                }
                if isinstance(edge, Mapping)
                else None
                for edge in edges
            ],
        }
        projected_rows.append(
            {**body, "action_item_sha256": semantic_hash(body)}
        )
        source_label_free_hashes.append(str(source_commitment))
    parsed = tuple(
        _parse_action_item(row, ordinal)
        for ordinal, row in enumerate(projected_rows)
    )
    action_hashes = [row.action_item_sha256 for row in parsed]
    if len(set(action_hashes)) != TOTAL_ITEMS:
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "projected public v1 action commitments overlap"
        )
    pack_body = {
        "schema": ACTION_PACK_SCHEMA,
        "version": DESIGN_VERSION,
        "block": BLOCK,
        "seed_count": SEED_COUNT,
        "item_count_per_seed": ITEMS_PER_SEED,
        "total_item_count": TOTAL_ITEMS,
        "labels_included": False,
        "items": projected_rows,
    }
    pack_sha256 = semantic_hash(pack_body)
    return (
        ActionPack(
            pack_sha256,
            source_file_sha256,
            semantic_hash(action_hashes),
            parsed,
        ),
        semantic_hash(source_label_free_hashes),
    )


def load_committed_v1_diagnostic_action_pack(
    project_root: Path,
) -> tuple[ActionPack, dict[str, str]]:
    root = project_root.resolve(strict=True)
    path = root / V1_PUBLICATION_RELATIVE_PATH
    payload, file_sha256 = _read_json(
        path, private=False, field="public v1 terminal publication"
    )
    try:
        committed = _committed_bytes(root, V1_PUBLICATION_RELATIVE_PATH)
    except Exception as exc:
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "public v1 terminal publication is not committed"
        ) from exc
    if committed != path.read_bytes() or file_sha256 != V1_PUBLICATION_FILE_SHA256:
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "public v1 terminal publication bytes drifted"
        )
    reproducibility = _self_hash(payload, "reproducibility_sha256")
    if (
        payload.get("schema") != V1_PUBLICATION_SCHEMA
        or payload.get("status") != V1_PUBLICATION_STATUS
        or payload.get("block") != BLOCK
        or payload.get("seed_count") != SEED_COUNT
        or payload.get("item_count_per_seed") != ITEMS_PER_SEED
        or payload.get("total_item_count") != TOTAL_ITEMS
        or payload.get("generated_item_commitment_set_sha256")
        != V1_GENERATED_ITEM_COMMITMENT_SET_SHA256
        or reproducibility != V1_PUBLICATION_REPRODUCIBILITY_SHA256
    ):
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "public v1 terminal publication contract drifted"
        )
    pack, source_projection_set = project_public_v1_label_free_rows(
        payload.get("items"), source_file_sha256=file_sha256
    )
    return pack, {
        "file_sha256": file_sha256,
        "reproducibility_sha256": reproducibility,
        "generated_item_commitment_set_sha256": (
            V1_GENERATED_ITEM_COMMITMENT_SET_SHA256
        ),
        "projected_action_pack_sha256": pack.pack_sha256,
        "projected_action_item_commitment_set_sha256": (
            pack.item_commitment_set_sha256
        ),
        "source_label_free_commitment_set_sha256": source_projection_set,
    }


def _validated_top5(
    value: Sequence[int], field: str
) -> tuple[int, int, int, int, int]:
    rows = tuple(value)
    if (
        len(rows) != TOP_K
        or len(set(rows)) != TOP_K
        or any(type(index) is not int or not 0 <= index < NODE_COUNT for index in rows)
    ):
        raise SyntheticTypedGraphMultiseedRunnerV2Error(f"{field} top5 drifted")
    return rows  # type: ignore[return-value]


def _ordered_embedding_texts(pack: ActionPack) -> tuple[str, ...]:
    if len(pack.rows) != TOTAL_ITEMS:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("action pack size drifted")
    texts: list[str] = []
    for ordinal, item in enumerate(pack.rows):
        if item.global_ordinal != ordinal or len(item.nodes) != NODE_COUNT:
            raise SyntheticTypedGraphMultiseedRunnerV2Error(
                "ordered embedding item coordinate drifted"
            )
        if len(texts) != ordinal * TEXTS_PER_ITEM:
            raise SyntheticTypedGraphMultiseedRunnerV2Error(
                "ordered embedding text offset drifted"
            )
        texts.append(item.question)
        texts.extend(core.embedding_text(node.identity_text) for node in item.nodes)
    if len(texts) != TOTAL_TEXT_COUNT:
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "ordered embedding text count drifted"
        )
    return tuple(texts)


def precompute_local_tensors(
    pack: ActionPack, encoder: EncoderProtocol
) -> tuple[tuple[LocalTensor, ...], MiniLMChunkAudit]:
    """Encode exactly two fixed contiguous chunks and preserve item offsets."""

    if semantic_hash(list(CHUNK_SCHEDULE)) != CHUNK_SCHEDULE_SHA256:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("chunk schedule hash drifted")
    if not (
        CHUNK_COUNT == 2
        and TEXTS_PER_CHUNK == 8448
        and TEXTS_PER_CHUNK <= FROZEN_MAXIMUM_TEXTS_PER_CALL
        and TOTAL_TEXT_COUNT == 16896
    ):
        raise SyntheticTypedGraphMultiseedRunnerV2Error("chunk constants drifted")
    texts = _ordered_embedding_texts(pack)
    matrices: list[np.ndarray] = []
    input_counts: list[int] = []
    output_counts: list[int] = []
    dimension: int | None = None
    for schedule in CHUNK_SCHEDULE:
        start = schedule["first_text_index_inclusive"]
        end = schedule["last_text_index_exclusive"]
        chunk = texts[start:end]
        if len(chunk) != TEXTS_PER_CHUNK:
            raise SyntheticTypedGraphMultiseedRunnerV2Error(
                "fixed MiniLM chunk input count drifted"
            )
        input_counts.append(len(chunk))
        try:
            matrix = np.asarray(encoder.encode(chunk), dtype=np.float32)
        except Exception as exc:
            raise SyntheticTypedGraphMultiseedRunnerV2Error(
                f"offline MiniLM fixed chunk {schedule['chunk_index']} failed"
            ) from exc
        if (
            matrix.ndim != 2
            or matrix.shape[0] != TEXTS_PER_CHUNK
            or not np.isfinite(matrix).all()
        ):
            raise SyntheticTypedGraphMultiseedRunnerV2Error(
                "offline MiniLM chunk output drifted"
            )
        if dimension is None:
            dimension = int(matrix.shape[1])
        if matrix.shape[1] != dimension or dimension <= 0:
            raise SyntheticTypedGraphMultiseedRunnerV2Error(
                "offline MiniLM chunk dimension drifted"
            )
        matrices.append(matrix)
        output_counts.append(int(matrix.shape[0]))
    if len(matrices) != CHUNK_COUNT:
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "offline MiniLM call count drifted"
        )
    matrix = np.concatenate(matrices, axis=0)
    if matrix.shape != (TOTAL_TEXT_COUNT, dimension):
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "ordered MiniLM concatenation drifted"
        )
    tensors: list[LocalTensor] = []
    for ordinal in range(TOTAL_ITEMS):
        start = ordinal * TEXTS_PER_ITEM
        query = matrix[start]
        nodes = matrix[start + 1 : start + TEXTS_PER_ITEM]
        similarities = tuple(
            quantized_cosine_similarity(query, node) for node in nodes
        )
        raw = _validated_top5(
            sorted(
                range(NODE_COUNT),
                key=lambda index: (-similarities[index], index),
            )[:TOP_K],
            RAW_ARM,
        )
        tensor_hash = semantic_hash(
            {"raw_top5": list(raw), "query_similarities": list(similarities)}
        )
        tensors.append(LocalTensor(raw, similarities, tensor_hash))
    audit = MiniLMChunkAudit(
        CHUNK_SCHEDULE_SHA256,
        tuple(input_counts),  # type: ignore[arg-type]
        tuple(output_counts),  # type: ignore[arg-type]
    )
    if (
        audit.observed_input_row_counts != (8448, 8448)
        or audit.observed_output_row_counts != (8448, 8448)
    ):
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "observed MiniLM chunk schedule drifted"
        )
    return tuple(tensors), audit


def _runtime_binding(runtime: OfficialRuntimeProtocol) -> str:
    safe = dict(runtime.safe_binding)
    encoded = json.dumps(safe, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if "/home/" in encoded or "/tmp/" in encoded or "\\" in encoded:
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "official runtime safe binding leaks a host path"
        )
    return semantic_hash(safe)


def _tracked_call(
    tracker: _ConcurrencyTracker,
    function: Callable[..., Any],
    *arguments: object,
) -> Any:
    tracker.enter()
    try:
        return function(*arguments)
    finally:
        tracker.exit()


def _official_action(
    runtime: OfficialRuntimeProtocol, item: ActionItem, work_root: Path
) -> tuple[int, int, int, int, int]:
    return _validated_top5(
        runtime.retrieve(
            question=item.question,
            paragraphs=item.paragraphs,
            work_root=work_root,
        ),
        HIPPO_ARM,
    )


def _raw_action(tensor: LocalTensor) -> tuple[int, int, int, int, int]:
    return _validated_top5(tensor.raw_top5, RAW_ARM)


def _agent_action(
    item: ActionItem,
    tensor: LocalTensor,
    official_future: Future[tuple[int, int, int, int, int]],
    submission_released: threading.Event,
    all_actions_submitted: threading.Event,
) -> tuple[tuple[int, int, int, int, int], str]:
    submission_released.wait()
    if not all_actions_submitted.is_set():
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "agent action released after an incomplete submission wave"
        )
    official = official_future.result()
    table = core.build_common_candidate_table(
        item.spans, item.full_edges, official, tensor.query_similarities
    )
    trace = core.execute_recipe(official, table, tensor.query_similarities, RECIPE_ID)
    if trace.recipe_id != RECIPE_ID:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("fixed R1 action drifted")
    return _validated_top5(trace.output_top5, AGENT_ARM), trace.common_scan_sha256


def _execute_all_actions(
    pack: ActionPack,
    tensors: Sequence[LocalTensor],
    runtime: OfficialRuntimeProtocol,
    work_root: Path,
) -> ActionWaveOutcome:
    """Submit all 1,536 futures before joining any one of them."""

    if work_root.exists() or work_root.is_symlink():
        raise SyntheticTypedGraphMultiseedRunnerV2Error("official work root already exists")
    work_root.mkdir(parents=True, mode=0o700)
    if len(tensors) != TOTAL_ITEMS:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("local tensor count drifted")
    preflight_binding = _runtime_binding(runtime)
    official_tracker = _ConcurrencyTracker(OFFICIAL_CONCURRENCY_CAP)
    local_tracker = _ConcurrencyTracker(LOCAL_CONCURRENCY_CAP)
    official_futures: list[Future[tuple[int, int, int, int, int]]] = []
    raw_futures: list[Future[tuple[int, int, int, int, int]]] = []
    agent_futures: list[Future[tuple[tuple[int, int, int, int, int], str]]] = []
    errors: list[BaseException] = []
    official_results: list[tuple[int, int, int, int, int] | None] = [None] * TOTAL_ITEMS
    raw_results: list[tuple[int, int, int, int, int] | None] = [None] * TOTAL_ITEMS
    agent_results: list[tuple[tuple[int, int, int, int, int], str] | None] = [None] * TOTAL_ITEMS
    submission_released = threading.Event()
    all_actions_submitted = threading.Event()
    with ThreadPoolExecutor(max_workers=OFFICIAL_CONCURRENCY_CAP) as official_pool, ThreadPoolExecutor(
        max_workers=LOCAL_CONCURRENCY_CAP
    ) as local_pool:
        try:
            for item in pack.rows:
                official_futures.append(
                    official_pool.submit(
                        _tracked_call,
                        official_tracker,
                        _official_action,
                        runtime,
                        item,
                        work_root / f"item_{item.global_ordinal:03d}",
                    )
                )
            for tensor in tensors:
                raw_futures.append(
                    local_pool.submit(_tracked_call, local_tracker, _raw_action, tensor)
                )
            for item, tensor, official_future in zip(pack.rows, tensors, official_futures):
                agent_futures.append(
                    local_pool.submit(
                        _tracked_call,
                        local_tracker,
                        _agent_action,
                        item,
                        tensor,
                        official_future,
                        submission_released,
                        all_actions_submitted,
                    )
                )
            if (
                len(official_futures) + len(raw_futures) + len(agent_futures)
                != ACTION_WORK_UNITS
            ):
                raise SyntheticTypedGraphMultiseedRunnerV2Error(
                    "action submission count drifted"
                )
            all_actions_submitted.set()
        finally:
            submission_released.set()
        for index, future in enumerate(official_futures):
            try:
                official_results[index] = future.result()
            except BaseException as exc:
                errors.append(exc)
        for index, future in enumerate(raw_futures):
            try:
                raw_results[index] = future.result()
            except BaseException as exc:
                errors.append(exc)
        for index, future in enumerate(agent_futures):
            try:
                agent_results[index] = future.result()
            except BaseException as exc:
                errors.append(exc)
    try:
        postflight = dict(runtime.fresh_reverify())
        postflight_receipt = semantic_hash(postflight)
        if postflight_receipt != preflight_binding:
            errors.append(
                SyntheticTypedGraphMultiseedRunnerV2Error(
                    "official runtime postflight binding drifted"
                )
            )
    except BaseException as exc:
        errors.append(exc)
        postflight_receipt = ""
    if errors:
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            f"label-free action or official postflight failed ({len(errors)} terminal errors)"
        ) from errors[0]
    if any(value is None for value in (*official_results, *raw_results, *agent_results)):
        raise SyntheticTypedGraphMultiseedRunnerV2Error("action completion barrier drifted")
    if not 1 <= official_tracker.peak <= OFFICIAL_CONCURRENCY_CAP:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("official peak concurrency drifted")
    if not 1 <= local_tracker.peak <= LOCAL_CONCURRENCY_CAP:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("local peak concurrency drifted")
    actions: list[ItemActions] = []
    for item, tensor, raw, official, agent in zip(
        pack.rows, tensors, raw_results, official_results, agent_results
    ):
        assert raw is not None and official is not None and agent is not None
        actions.append(
            ItemActions(
                item.global_ordinal,
                item.action_item_sha256,
                raw,
                official,
                agent[0],
                agent[1],
                tensor.tensor_sha256,
            )
        )
    return ActionWaveOutcome(
        tuple(actions),
        preflight_binding,
        postflight_receipt,
        official_tracker.peak,
        local_tracker.peak,
    )


def _action_rows(actions: Sequence[ItemActions]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "global_ordinal": row.global_ordinal,
            "action_item_sha256": row.action_item_sha256,
            "RAW_top5": list(row.raw_top5),
            "official_HippoRAG_top5": list(row.official_top5),
            "Agent_R1_top5": list(row.agent_top5),
            "common_scan_sha256": row.common_scan_sha256,
            "local_tensor_sha256": row.local_tensor_sha256,
        }
        for row in actions
    )


def _persist_action_seal(
    *,
    path: Path,
    pack: ActionPack,
    wave: ActionWaveOutcome,
    chunk_audit: MiniLMChunkAudit,
    purpose: str,
) -> tuple[str, str, str]:
    if purpose not in {"public_integration_diagnostic", "fresh_formal_replication"}:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("action seal purpose drifted")
    if len(wave.actions) != TOTAL_ITEMS:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("action seal row count drifted")
    action_table_sha256 = semantic_hash(
        [
            [
                list(row.raw_top5),
                list(row.official_top5),
                list(row.agent_top5),
            ]
            for row in wave.actions
        ]
    )
    if purpose == "public_integration_diagnostic":
        # The public diagnostic discards every ranked identity after forming
        # this single aggregate commitment.  Its private seal has the exact
        # narrow field set frozen by the v2 design: no rows, indices, per-row
        # hashes, similarities, or model output can survive in the artifact.
        diagnostic_body = {
            "schema": (
                "synthetic_typed_graph_multiseed_replication_"
                "integration_diagnostic_private_action_seal_v2"
            ),
            "total_action_count": ACTION_WORK_UNITS,
            "arm_terminal_counts": {
                RAW_ARM: TOTAL_ITEMS,
                HIPPO_ARM: TOTAL_ITEMS,
                AGENT_ARM: TOTAL_ITEMS,
            },
            "ordered_action_commitment_set_sha256": action_table_sha256,
            "official_peak_concurrency": wave.official_peak_concurrency_count,
            "local_peak_concurrency": wave.local_peak_concurrency_count,
            "postflight_receipt_sha256": (
                wave.official_postflight_receipt_sha256
            ),
            "action_rows_or_ranked_indices_persisted": False,
        }
        file_sha256 = _write_json_exclusive(path, diagnostic_body, PRIVATE_MODE)
        return semantic_hash(diagnostic_body), file_sha256, action_table_sha256
    rows = _action_rows(wave.actions)
    body = {
        "schema": f"{VERSION}_private_action_seal",
        "version": VERSION,
        "status": "all_1536_actions_joined_official_postflight_terminal",
        "purpose": purpose,
        "block": BLOCK,
        "recipe_id": RECIPE_ID,
        "item_count": TOTAL_ITEMS,
        "action_work_unit_count": ACTION_WORK_UNITS,
        "submitted_action_work_unit_count": ACTION_WORK_UNITS,
        "terminal_action_work_unit_count": ACTION_WORK_UNITS,
        "official_retrieve_action_count": TOTAL_ITEMS,
        "official_call_count": TOTAL_ITEMS,
        "RAW_action_count": TOTAL_ITEMS,
        "Agent_R1_action_count": TOTAL_ITEMS,
        "official_concurrency_cap": OFFICIAL_CONCURRENCY_CAP,
        "local_concurrency_cap": LOCAL_CONCURRENCY_CAP,
        "official_peak_concurrency_count": wave.official_peak_concurrency_count,
        "local_peak_concurrency_count": wave.local_peak_concurrency_count,
        "chunk_schedule_sha256": chunk_audit.chunk_schedule_sha256,
        "observed_encoder_input_row_counts": list(
            chunk_audit.observed_input_row_counts
        ),
        "observed_encoder_output_row_counts": list(
            chunk_audit.observed_output_row_counts
        ),
        "action_pack_file_sha256": pack.file_sha256,
        "action_pack_sha256": pack.pack_sha256,
        "action_item_commitment_set_sha256": pack.item_commitment_set_sha256,
        "runtime_binding_sha256": wave.runtime_binding_sha256,
        "official_postflight_receipt_sha256": (
            wave.official_postflight_receipt_sha256
        ),
        "action_table_sha256": action_table_sha256,
        "action_rows": list(rows),
        "labels_opened_before_action_seal": False,
        "labels_opened_before_seal": False,
        "scores_computed_before_action_seal": False,
    }
    seal = {**body, "action_seal_sha256": semantic_hash(body)}
    file_sha256 = _write_json_exclusive(path, seal, PRIVATE_MODE)
    return str(seal["action_seal_sha256"]), file_sha256, action_table_sha256


def _join(
    action_pack: ActionPack,
    label_pack: LabelPack,
    actions: Sequence[ItemActions],
) -> tuple[tuple[ActionItem, LabelItem, ItemActions], ...]:
    if len(actions) != TOTAL_ITEMS:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("action row count drifted")
    joined = tuple(zip(action_pack.rows, label_pack.rows, actions))
    if any(
        item.global_ordinal != label.global_ordinal
        or item.global_ordinal != action.global_ordinal
        or item.action_item_sha256 != label.action_item_sha256
        or item.action_item_sha256 != action.action_item_sha256
        for item, label, action in joined
    ):
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "action/label identity join drifted"
        )
    return joined


def _summary(
    rows: Sequence[tuple[ActionItem, LabelItem, ItemActions]], arm: str
) -> dict[str, int]:
    hits = complete = support = total_u = 0
    for _item, label, action in rows:
        output = (
            action.raw_top5
            if arm == RAW_ARM
            else action.official_top5
            if arm == HIPPO_ARM
            else action.agent_top5
        )
        item_hits, item_complete, utility = core.item_utility(
            output, label.gold_node_indices, source_count=NODE_COUNT
        )
        hits += item_hits
        complete += item_complete
        support += len(label.gold_node_indices)
        total_u += utility
    return {
        "item_count": len(rows),
        "support_hit_count": hits,
        "support_total": support,
        "complete_count": complete,
        "total_U": total_u,
    }


def _stratified_arm(
    joined: Sequence[tuple[ActionItem, LabelItem, ItemActions]], arm: str
) -> dict[str, Any]:
    by_seed: dict[str, list[tuple[ActionItem, LabelItem, ItemActions]]] = defaultdict(list)
    by_family: dict[str, list[tuple[ActionItem, LabelItem, ItemActions]]] = defaultdict(list)
    by_polarity: dict[str, list[tuple[ActionItem, LabelItem, ItemActions]]] = defaultdict(list)
    for row in joined:
        item, label, _action = row
        by_seed[f"seed_{item.seed_index:02d}"].append(row)
        by_family[label.family_id].append(row)
        by_polarity[label.polarity].append(row)
    return {
        "overall": _summary(joined, arm),
        "by_seed": {key: _summary(by_seed[key], arm) for key in sorted(by_seed)},
        "by_family": {key: _summary(by_family[key], arm) for key in sorted(by_family)},
        "by_polarity": {
            key: _summary(by_polarity[key], arm) for key in sorted(by_polarity)
        },
    }


def _difference_summary(
    left_seed: Mapping[str, Mapping[str, int]],
    right_seed: Mapping[str, Mapping[str, int]],
    comparison: str,
) -> dict[str, Any]:
    keys = tuple(f"seed_{index:02d}" for index in range(SEED_COUNT))
    deltas = tuple(left_seed[key]["total_U"] - right_seed[key]["total_U"] for key in keys)
    return {
        "comparison": comparison,
        "ordered_seed_deltas": list(deltas),
        "mean_delta": sum(deltas) / SEED_COUNT,
        "median_delta": float(median(deltas)),
        "minimum_delta": min(deltas),
        "maximum_delta": max(deltas),
        "range_delta": max(deltas) - min(deltas),
        "K_positive": sum(delta > 0 for delta in deltas),
    }


def _aggregate_arms(
    joined: Sequence[tuple[ActionItem, LabelItem, ItemActions]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(joined) != TOTAL_ITEMS:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("joined cohort size drifted")
    aggregates = {arm: _stratified_arm(joined, arm) for arm in ARM_IDS}
    differences = {
        "Agent_R1_minus_official_HippoRAG": _difference_summary(
            aggregates[AGENT_ARM]["by_seed"],
            aggregates[HIPPO_ARM]["by_seed"],
            "Agent_R1_minus_official_HippoRAG",
        ),
        "Agent_R1_minus_RAW": _difference_summary(
            aggregates[AGENT_ARM]["by_seed"],
            aggregates[RAW_ARM]["by_seed"],
            "Agent_R1_minus_RAW",
        ),
    }
    return aggregates, differences


def run_multiseed_replication(
    action_pack: ActionPack,
    *,
    label_loader: Callable[[], LabelPack],
    encoder: EncoderProtocol,
    runtime: OfficialRuntimeProtocol,
    work_root: Path,
    action_seal_path: Path,
) -> MultiseedOutcome:
    """Execute fixed actions, seal them, then open labels exactly once."""

    if action_seal_path.exists() or action_seal_path.is_symlink():
        raise SyntheticTypedGraphMultiseedRunnerV2Error("action seal already exists")
    tensors, chunk_audit = precompute_local_tensors(action_pack, encoder)
    wave = _execute_all_actions(action_pack, tensors, runtime, work_root)
    seal_sha256, seal_file_sha256, action_table_sha256 = _persist_action_seal(
        path=action_seal_path,
        pack=action_pack,
        wave=wave,
        chunk_audit=chunk_audit,
        purpose="fresh_formal_replication",
    )
    labels = label_loader()
    if not isinstance(labels, LabelPack):
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "late label loader returned wrong type"
        )
    joined = _join(action_pack, labels, wave.actions)
    aggregates, differences = _aggregate_arms(joined)
    return MultiseedOutcome(
        action_pack.file_sha256,
        action_pack.pack_sha256,
        action_pack.item_commitment_set_sha256,
        labels.file_sha256,
        labels.pack_sha256,
        labels.item_commitment_set_sha256,
        wave.runtime_binding_sha256,
        wave.official_postflight_receipt_sha256,
        chunk_audit,
        action_table_sha256,
        seal_sha256,
        seal_file_sha256,
        wave.official_peak_concurrency_count,
        wave.local_peak_concurrency_count,
        aggregates,
        differences,
    )


def multiseed_public_result(outcome: MultiseedOutcome) -> dict[str, Any]:
    body = {
        "schema": RESULT_SCHEMA,
        "version": DESIGN_VERSION,
        "status": SUCCESS_RESULT_STATUS,
        "design_sha256": DESIGN_SHA256,
        "design_file_sha256": DESIGN_FILE_SHA256,
        "block": BLOCK,
        "recipe_id": RECIPE_ID,
        "seed_count": SEED_COUNT,
        "item_count_per_seed": ITEMS_PER_SEED,
        "total_item_count": TOTAL_ITEMS,
        "arms": list(ARM_IDS),
        "action_work_unit_count": ACTION_WORK_UNITS,
        "official_retrieve_action_count": TOTAL_ITEMS,
        "official_concurrency_cap": OFFICIAL_CONCURRENCY_CAP,
        "local_concurrency_cap": LOCAL_CONCURRENCY_CAP,
        "official_peak_concurrency_count": outcome.official_peak_concurrency_count,
        "local_peak_concurrency_count": outcome.local_peak_concurrency_count,
        "chunk_schedule_sha256": outcome.chunk_audit.chunk_schedule_sha256,
        "observed_encoder_input_row_counts": list(
            outcome.chunk_audit.observed_input_row_counts
        ),
        "observed_encoder_output_row_counts": list(
            outcome.chunk_audit.observed_output_row_counts
        ),
        "action_pack_file_sha256": outcome.action_pack_file_sha256,
        "action_pack_sha256": outcome.action_pack_sha256,
        "action_item_commitment_set_sha256": outcome.action_item_commitment_set_sha256,
        "label_pack_file_sha256": outcome.label_pack_file_sha256,
        "label_pack_sha256": outcome.label_pack_sha256,
        "label_item_commitment_set_sha256": outcome.label_item_commitment_set_sha256,
        "runtime_binding_sha256": outcome.runtime_binding_sha256,
        "official_postflight_receipt_sha256": (
            outcome.official_postflight_receipt_sha256
        ),
        "action_table_sha256": outcome.action_table_sha256,
        "action_seal_sha256": outcome.action_seal_sha256,
        "action_seal_file_sha256": outcome.action_seal_file_sha256,
        "aggregates": dict(outcome.aggregates),
        "cluster_differences": dict(outcome.cluster_differences),
        "interpretation": "descriptive_fixed_cohort_replication_only",
        "seeds_or_item_rows_disclosed": False,
    }
    return {**body, "receipt_sha256": semantic_hash(body)}


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def _current_committed_code_bindings(project_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for relative in (
        ACQUISITION_MODULE_RELATIVE_PATH,
        RUNNER_MODULE_RELATIVE_PATH,
        ACQUISITION_TEST_RELATIVE_PATH,
        RUNNER_TEST_RELATIVE_PATH,
    ):
        path = _assert_no_symlink_components(
            project_root / relative, "diagnostic code binding"
        )
        if not path.is_file() or path.is_symlink():
            raise SyntheticTypedGraphMultiseedRunnerV2Error(
                "diagnostic code binding is unavailable"
            )
        raw = path.read_bytes()
        try:
            committed = _committed_bytes(project_root, relative)
        except Exception as exc:
            raise SyntheticTypedGraphMultiseedRunnerV2Error(
                "diagnostic code binding is not committed"
            ) from exc
        if raw != committed:
            raise SyntheticTypedGraphMultiseedRunnerV2Error(
                "diagnostic code binding differs from current HEAD"
            )
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "file_sha256": _sha256_bytes(committed),
                "git_blob_sha1": _git_blob_sha1(committed),
            }
        )
    return rows


def _verify_committed_design(project_root: Path) -> None:
    relative = Path("manifests/synthetic_typed_graph_multiseed_replication_design_v2.json")
    path = project_root / relative
    if (
        not path.is_file()
        or path.is_symlink()
        or _sha256_file(path) != DESIGN_FILE_SHA256
        or _committed_bytes(project_root, relative) != path.read_bytes()
    ):
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "v2 design is not exact current-HEAD committed bytes"
        )
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict) or _self_hash(payload, "design_sha256") != DESIGN_SHA256:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("v2 design hash drifted")
    schedule = payload.get("minilm_chunk_repair_contract")
    if (
        not isinstance(schedule, Mapping)
        or schedule.get("chunk_schedule_sha256") != CHUNK_SCHEDULE_SHA256
        or schedule.get("chunk_schedule") != list(CHUNK_SCHEDULE)
    ):
        raise SyntheticTypedGraphMultiseedRunnerV2Error("v2 design chunk schedule drifted")


def _diagnostic_forbidden_fresh_paths(project_root: Path) -> tuple[Path, ...]:
    return tuple(
        project_root / relative
        for relative in (
            ARTIFACT_ROOT_RELATIVE_PATH / "seed_generation.attempt.marker",
            ARTIFACT_ROOT_RELATIVE_PATH / "seed_batch.bin",
            ARTIFACT_ROOT_RELATIVE_PATH / "acquisition.attempt.marker",
            ACTION_PACK_RELATIVE_PATH,
            LABEL_PACK_RELATIVE_PATH,
            ARTIFACT_ROOT_RELATIVE_PATH / "full_compiled_cohort_pack.json",
            FORMAL_MARKER_RELATIVE_PATH,
            FORMAL_ACTION_SEAL_RELATIVE_PATH,
            RESULT_RELATIVE_PATH,
            Path(
                "published/synthetic_typed_graph_multiseed_replication_v2/"
                "formal_seeds_and_cohort.json"
            ),
        )
    )


def _consume_diagnostic_marker(
    *, project_root: Path, actual_head: str, bindings: Sequence[Mapping[str, str]]
) -> tuple[dict[str, Any], str]:
    marker_path = project_root / DIAGNOSTIC_MARKER_RELATIVE_PATH
    occupied = (
        marker_path,
        project_root / DIAGNOSTIC_WORK_RELATIVE_PATH,
        project_root / DIAGNOSTIC_ACTION_SEAL_RELATIVE_PATH,
        project_root / DIAGNOSTIC_RELATIVE_PATH,
    )
    if any(path.exists() or path.is_symlink() for path in occupied):
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "canonical integration diagnostic attempt already exists"
        )
    if any(
        path.exists() or path.is_symlink()
        for path in _diagnostic_forbidden_fresh_paths(project_root)
    ):
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "fresh formal seed or cohort exists before diagnostic success"
        )
    body = {
        "schema": f"{DIAGNOSTIC_SCHEMA}_attempt_marker",
        "version": DESIGN_VERSION,
        "status": "sole_public_label_free_integration_diagnostic_attempt_consumed",
        "actual_HEAD": actual_head,
        "design_sha256": DESIGN_SHA256,
        "bindings_sha256": semantic_hash(list(bindings)),
        "attempt_count": 1,
        "fresh_formal_seed_or_cohort_exists": False,
    }
    marker = {**body, "marker_sha256": semantic_hash(body)}
    marker_file_sha256 = _write_json_exclusive(marker_path, marker, PRIVATE_MODE)
    return marker, marker_file_sha256


def _diagnostic_success_receipt(
    *,
    actual_head: str,
    bindings: Sequence[Mapping[str, str]],
    source_binding: Mapping[str, str],
    chunk_audit: MiniLMChunkAudit,
    wave: ActionWaveOutcome,
    action_table_sha256: str,
    seal_sha256: str,
    seal_file_sha256: str,
    marker: Mapping[str, Any],
    marker_file_sha256: str,
) -> dict[str, Any]:
    body = {
        "schema": DIAGNOSTIC_SCHEMA,
        "version": DESIGN_VERSION,
        "status": DIAGNOSTIC_SUCCESS_STATUS,
        "invocation_HEAD": actual_head,
        "design_sha256": DESIGN_SHA256,
        "design_file_sha256": DESIGN_FILE_SHA256,
        "bindings": [dict(row) for row in bindings],
        "source_v1_publication": {
            "file_sha256": source_binding["file_sha256"],
            "reproducibility_sha256": source_binding["reproducibility_sha256"],
            "generated_item_commitment_set_sha256": source_binding[
                "generated_item_commitment_set_sha256"
            ],
            "projected_action_pack_sha256": source_binding[
                "projected_action_pack_sha256"
            ],
            "projected_action_item_commitment_set_sha256": source_binding[
                "projected_action_item_commitment_set_sha256"
            ],
            "source_label_free_commitment_set_sha256": source_binding[
                "source_label_free_commitment_set_sha256"
            ],
        },
        "chunk_schedule": {
            "chunk_count": CHUNK_COUNT,
            "texts_per_chunk": TEXTS_PER_CHUNK,
            "total_text_count": TOTAL_TEXT_COUNT,
            "chunk_schedule_sha256": chunk_audit.chunk_schedule_sha256,
        },
        "counts": {
            "item_count": TOTAL_ITEMS,
            "action_work_unit_count": ACTION_WORK_UNITS,
            "submitted_action_work_unit_count": ACTION_WORK_UNITS,
            "terminal_action_work_unit_count": ACTION_WORK_UNITS,
            "official_retrieve_action_count": TOTAL_ITEMS,
            "RAW_action_count": TOTAL_ITEMS,
            "Agent_R1_action_count": TOTAL_ITEMS,
        },
        "arms": list(ARM_IDS),
        "official_concurrency_cap": OFFICIAL_CONCURRENCY_CAP,
        "local_concurrency_cap": LOCAL_CONCURRENCY_CAP,
        "observed_encoder_input_row_counts": list(
            chunk_audit.observed_input_row_counts
        ),
        "observed_encoder_output_row_counts": list(
            chunk_audit.observed_output_row_counts
        ),
        "official_peak_concurrency_count": wave.official_peak_concurrency_count,
        "local_peak_concurrency_count": wave.local_peak_concurrency_count,
        "runtime_binding_sha256": wave.runtime_binding_sha256,
        "official_postflight_receipt_sha256": (
            wave.official_postflight_receipt_sha256
        ),
        "action_table_sha256": action_table_sha256,
        "action_seal_sha256": seal_sha256,
        "action_seal_file_sha256": seal_file_sha256,
        "diagnostic_attempt_marker_sha256": marker["marker_sha256"],
        "diagnostic_attempt_marker_file_sha256": marker_file_sha256,
        "labels_opened": False,
        "scores_computed": False,
        "estimands_computed": False,
        "claims_made": False,
        "network_calls": 0,
        "retrieval_actions_model_outputs_or_scores_disclosed": False,
        "action_rows_or_ranked_indices_persisted": False,
        "action_identity_or_quality_used_for_decision": False,
        "diagnostic_is_non_claim": True,
        "fresh_formal_seed_authorized": True,
    }
    return {**body, "diagnostic_sha256": semantic_hash(body)}


def _persist_diagnostic_failure(
    *,
    project_root: Path,
    actual_head: str,
    marker: Mapping[str, Any],
    marker_file_sha256: str,
    exc: BaseException,
) -> None:
    path = project_root / DIAGNOSTIC_RELATIVE_PATH
    if path.exists() or path.is_symlink():
        return
    body = {
        "schema": DIAGNOSTIC_SCHEMA,
        "version": DESIGN_VERSION,
        "status": DIAGNOSTIC_FAILURE_STATUS,
        "invocation_HEAD": actual_head,
        "design_sha256": DESIGN_SHA256,
        "design_file_sha256": DESIGN_FILE_SHA256,
        "diagnostic_attempt_marker_sha256": marker["marker_sha256"],
        "diagnostic_attempt_marker_file_sha256": marker_file_sha256,
        "failure_class": type(exc).__name__,
        "exception_message_or_action_content_persisted": False,
        "labels_opened": False,
        "scores_computed": False,
        "estimands_computed": False,
        "claims_made": False,
        "network_calls": 0,
        "fresh_formal_seed_authorized": False,
        "retry_replacement_or_backup_attempt_authorized": False,
    }
    failure = {**body, "diagnostic_sha256": semantic_hash(body)}
    _write_json_exclusive(path, failure, PUBLIC_MODE)


def run_canonical_integration_diagnostic(
    *,
    project_root: Path,
    encoder: EncoderProtocol,
    runtime: OfficialRuntimeProtocol,
) -> dict[str, Any]:
    """Consume the one public non-scoring integration diagnostic attempt."""

    if _DIAGNOSTIC_ENTRY_ACTIVE is not True:
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "canonical diagnostic may only be consumed by the diagnostic CLI"
        )
    if not isinstance(encoder, OfflineMiniLMEncoder) or not isinstance(
        runtime, PreparedFormalRuntimeV2
    ):
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "canonical diagnostic resources are not attested formal types"
        )
    root = project_root.resolve(strict=True)
    _verify_committed_design(root)
    actual_head = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    bindings = _current_committed_code_bindings(root)
    action_pack, source_binding = load_committed_v1_diagnostic_action_pack(root)
    marker, marker_file_sha256 = _consume_diagnostic_marker(
        project_root=root, actual_head=actual_head, bindings=bindings
    )
    try:
        tensors, chunk_audit = precompute_local_tensors(action_pack, encoder)
        wave = _execute_all_actions(
            action_pack,
            tensors,
            runtime,
            root / DIAGNOSTIC_WORK_RELATIVE_PATH,
        )
        seal_sha256, seal_file_sha256, action_table_sha256 = _persist_action_seal(
            path=root / DIAGNOSTIC_ACTION_SEAL_RELATIVE_PATH,
            pack=action_pack,
            wave=wave,
            chunk_audit=chunk_audit,
            purpose="public_integration_diagnostic",
        )
        # Action identities have now been reduced to one aggregate commitment;
        # no label loader or evaluation function is reachable in this branch.
        receipt = _diagnostic_success_receipt(
            actual_head=actual_head,
            bindings=bindings,
            source_binding=source_binding,
            chunk_audit=chunk_audit,
            wave=wave,
            action_table_sha256=action_table_sha256,
            seal_sha256=seal_sha256,
            seal_file_sha256=seal_file_sha256,
            marker=marker,
            marker_file_sha256=marker_file_sha256,
        )
        _write_json_exclusive(root / DIAGNOSTIC_RELATIVE_PATH, receipt, PUBLIC_MODE)
        return receipt
    except BaseException as exc:
        _persist_diagnostic_failure(
            project_root=root,
            actual_head=actual_head,
            marker=marker,
            marker_file_sha256=marker_file_sha256,
            exc=exc,
        )
        raise


def _load_committed_acquisition_metadata(
    project_root: Path, freeze: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    """Validate committed public metadata without opening a private pack."""

    path = project_root / ACQUISITION_RECEIPT_RELATIVE_PATH
    payload, file_sha256 = _read_json(
        path, private=False, field="committed v2 acquisition receipt"
    )
    try:
        committed = _committed_bytes(project_root, ACQUISITION_RECEIPT_RELATIVE_PATH)
    except Exception as exc:
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "v2 acquisition receipt is not committed"
        ) from exc
    if committed != path.read_bytes():
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "v2 acquisition receipt differs from current HEAD"
        )
    body = dict(payload)
    declared = body.pop("receipt_sha256", None)
    commitments = payload.get("commitments")
    if (
        payload.get("schema") != ACQUISITION_SCHEMA
        or payload.get("version") != DESIGN_VERSION
        or payload.get("status") != ACQUISITION_STATUS
        or payload.get("design_sha256") != DESIGN_SHA256
        or payload.get("design_file_sha256") != DESIGN_FILE_SHA256
        or payload.get("implementation_freeze_sha256")
        != freeze.get("implementation_freeze_sha256")
        or payload.get("block") != BLOCK
        or payload.get("seed_count") != SEED_COUNT
        or payload.get("item_count_per_seed") != ITEMS_PER_SEED
        or payload.get("total_item_count") != TOTAL_ITEMS
        or payload.get("grammar_generate_block_call_count") != SEED_COUNT
        or payload.get("new_original_and_v1_item_commitments_pairwise_disjoint")
        is not True
        or payload.get("fixed_recipe_id") != RECIPE_ID
        or payload.get("arms") != list(ARM_IDS)
        or not isinstance(commitments, Mapping)
        or set(commitments)
        != {
            "action_pack_file_sha256",
            "action_item_commitment_set_sha256",
            "label_pack_file_sha256",
            "label_item_commitment_set_sha256",
            "compiled_cohort_pack_file_sha256",
            "compiled_row_commitment_set_sha256",
        }
        or any(not _is_sha256(value) for value in commitments.values())
        or not _is_sha256(payload.get("generated_item_commitment_set_sha256"))
        or not isinstance(declared, str)
        or semantic_hash(body) != declared
    ):
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "committed v2 acquisition metadata drifted"
        )
    return payload, file_sha256


def _formal_paths(project_root: Path) -> dict[str, Path]:
    return {
        "marker": project_root / FORMAL_MARKER_RELATIVE_PATH,
        "work": project_root / FORMAL_WORK_RELATIVE_PATH,
        "seal": project_root / FORMAL_ACTION_SEAL_RELATIVE_PATH,
        "result": project_root / RESULT_RELATIVE_PATH,
    }


def _consume_formal_marker(
    *,
    project_root: Path,
    actual_head: str,
    freeze: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    acquisition_file_sha256: str,
) -> tuple[dict[str, Any], str, dict[str, Path]]:
    paths = _formal_paths(project_root)
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "canonical v2 formal runner attempt already exists"
        )
    body = {
        "schema": f"{VERSION}_formal_attempt_marker",
        "version": VERSION,
        "status": "sole_formal_replication_attempt_consumed",
        "actual_HEAD": actual_head,
        "design_sha256": DESIGN_SHA256,
        "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
        "acquisition_receipt_sha256": acquisition["receipt_sha256"],
        "acquisition_receipt_file_sha256": acquisition_file_sha256,
        "attempt_count": 1,
        "private_packs_opened_before_marker": False,
    }
    marker = {**body, "marker_sha256": semantic_hash(body)}
    marker_file_sha256 = _write_json_exclusive(
        paths["marker"], marker, PRIVATE_MODE
    )
    return marker, marker_file_sha256, paths


def _pack_matches_commitments(
    *,
    pack_file_sha256: str,
    item_set_sha256: str,
    commitments: Mapping[str, Any],
    prefix: str,
) -> None:
    if (
        pack_file_sha256 != commitments.get(f"{prefix}_pack_file_sha256")
        or item_set_sha256
        != commitments.get(f"{prefix}_item_commitment_set_sha256")
    ):
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            f"{prefix} pack differs from committed v2 acquisition"
        )


def _formal_result(
    outcome: MultiseedOutcome,
    *,
    marker: Mapping[str, Any],
    marker_file_sha256: str,
    actual_head: str,
    freeze: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    acquisition_file_sha256: str,
) -> dict[str, Any]:
    base = multiseed_public_result(outcome)
    base.pop("receipt_sha256")
    base.update(
        {
            "invocation_HEAD": actual_head,
            "implementation_freeze_sha256": freeze[
                "implementation_freeze_sha256"
            ],
            "acquisition_receipt_sha256": acquisition["receipt_sha256"],
            "acquisition_receipt_file_sha256": acquisition_file_sha256,
            "generated_item_commitment_set_sha256": acquisition[
                "generated_item_commitment_set_sha256"
            ],
            "formal_attempt_marker_sha256": marker["marker_sha256"],
            "formal_attempt_marker_file_sha256": marker_file_sha256,
            "result_must_be_committed_before_terminal_publication": True,
        }
    )
    return {**base, "receipt_sha256": semantic_hash(base)}


def _persist_formal_failure(
    *,
    path: Path,
    marker: Mapping[str, Any],
    marker_file_sha256: str,
    actual_head: str,
    freeze: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    acquisition_file_sha256: str,
    action_seal_path: Path,
    exc: BaseException,
) -> None:
    seal_file_sha256: str | None = None
    if action_seal_path.is_file() and not action_seal_path.is_symlink():
        seal_file_sha256 = _sha256_file(action_seal_path)
    body = {
        "schema": RESULT_SCHEMA,
        "version": DESIGN_VERSION,
        "status": FAILURE_RESULT_STATUS,
        "invocation_HEAD": actual_head,
        "design_sha256": DESIGN_SHA256,
        "design_file_sha256": DESIGN_FILE_SHA256,
        "implementation_freeze_sha256": freeze["implementation_freeze_sha256"],
        "acquisition_receipt_sha256": acquisition["receipt_sha256"],
        "acquisition_receipt_file_sha256": acquisition_file_sha256,
        "generated_item_commitment_set_sha256": acquisition[
            "generated_item_commitment_set_sha256"
        ],
        "formal_attempt_marker_sha256": marker["marker_sha256"],
        "formal_attempt_marker_file_sha256": marker_file_sha256,
        "action_seal_file_sha256": seal_file_sha256,
        "failure_class": type(exc).__name__,
        "retry_replacement_or_backup_attempt_authorized": False,
        "exception_message_seed_item_or_label_content_persisted_publicly": False,
        "result_must_be_committed_before_terminal_publication": True,
    }
    failure = {**body, "receipt_sha256": semantic_hash(body)}
    if not path.exists() and not path.is_symlink():
        _write_json_exclusive(path, failure, PUBLIC_MODE)


def run_canonical_multiseed(
    *,
    project_root: Path,
    encoder: EncoderProtocol,
    runtime: OfficialRuntimeProtocol,
) -> dict[str, Any]:
    """Consume the sole fresh v2 formal attempt and persist one terminal result."""

    if _FORMAL_ENTRY_ACTIVE is not True:
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "canonical v2 formal run may only be consumed by the formal CLI"
        )
    if not isinstance(encoder, OfflineMiniLMEncoder) or not isinstance(
        runtime, PreparedFormalRuntimeV2
    ):
        raise SyntheticTypedGraphMultiseedRunnerV2Error(
            "canonical v2 formal resources are not attested formal types"
        )
    root = project_root.resolve(strict=True)
    freeze, actual_head = verify_implementation_freeze(root)
    acquisition, acquisition_file_sha256 = _load_committed_acquisition_metadata(
        root, freeze
    )
    marker, marker_file_sha256, paths = _consume_formal_marker(
        project_root=root,
        actual_head=actual_head,
        freeze=freeze,
        acquisition=acquisition,
        acquisition_file_sha256=acquisition_file_sha256,
    )
    commitments = acquisition["commitments"]
    assert isinstance(commitments, Mapping)
    try:
        action_pack = load_action_pack(root / ACTION_PACK_RELATIVE_PATH)
        _pack_matches_commitments(
            pack_file_sha256=action_pack.file_sha256,
            item_set_sha256=action_pack.item_commitment_set_sha256,
            commitments=commitments,
            prefix="action",
        )
        label_open_count = 0

        def load_late_labels() -> LabelPack:
            nonlocal label_open_count
            label_open_count += 1
            if label_open_count != 1:
                raise SyntheticTypedGraphMultiseedRunnerV2Error(
                    "late label pack opening count drifted"
                )
            labels = load_label_pack(root / LABEL_PACK_RELATIVE_PATH)
            _pack_matches_commitments(
                pack_file_sha256=labels.file_sha256,
                item_set_sha256=labels.item_commitment_set_sha256,
                commitments=commitments,
                prefix="label",
            )
            return labels

        outcome = run_multiseed_replication(
            action_pack,
            label_loader=load_late_labels,
            encoder=encoder,
            runtime=runtime,
            work_root=paths["work"],
            action_seal_path=paths["seal"],
        )
        if label_open_count != 1:
            raise SyntheticTypedGraphMultiseedRunnerV2Error(
                "late label pack was not opened exactly once"
            )
        result = _formal_result(
            outcome,
            marker=marker,
            marker_file_sha256=marker_file_sha256,
            actual_head=actual_head,
            freeze=freeze,
            acquisition=acquisition,
            acquisition_file_sha256=acquisition_file_sha256,
        )
        _write_json_exclusive(paths["result"], result, PUBLIC_MODE)
        return result
    except BaseException as exc:
        _persist_formal_failure(
            path=paths["result"],
            marker=marker,
            marker_file_sha256=marker_file_sha256,
            actual_head=actual_head,
            freeze=freeze,
            acquisition=acquisition,
            acquisition_file_sha256=acquisition_file_sha256,
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
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run-integration-diagnostic", "run-formal"):
        child = subparsers.add_parser(command)
        child.add_argument("--project-root", required=True, type=Path)
        child.add_argument("--runtime-python", required=True, type=Path)
        child.add_argument("--local-llm-model", required=True, type=Path)
        child.add_argument("--local-embedding-model", required=True, type=Path)
    arguments = parser.parse_args(argv)
    encoder, runtime = _prepare_formal_resources(
        project_root=arguments.project_root,
        runtime_python=arguments.runtime_python,
        local_llm_model=arguments.local_llm_model,
        local_embedding_model=arguments.local_embedding_model,
    )
    global _DIAGNOSTIC_ENTRY_ACTIVE, _FORMAL_ENTRY_ACTIVE
    if _DIAGNOSTIC_ENTRY_ACTIVE or _FORMAL_ENTRY_ACTIVE:
        raise SyntheticTypedGraphMultiseedRunnerV2Error("canonical runner entry is active")
    if arguments.command == "run-integration-diagnostic":
        _DIAGNOSTIC_ENTRY_ACTIVE = True
        try:
            result = run_canonical_integration_diagnostic(
                project_root=arguments.project_root,
                encoder=encoder,
                runtime=runtime,
            )
        finally:
            _DIAGNOSTIC_ENTRY_ACTIVE = False
    else:
        _FORMAL_ENTRY_ACTIVE = True
        try:
            result = run_canonical_multiseed(
                project_root=arguments.project_root,
                encoder=encoder,
                runtime=runtime,
            )
        finally:
            _FORMAL_ENTRY_ACTIVE = False
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
