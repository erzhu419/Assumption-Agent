"""One-shot eight-seed replication runner for the frozen synthetic R1 arm.

The runner executes exactly three label-free actions for every one of 512
``A_hold`` items: the frozen offline MiniLM RAW top five, one shared official
HippoRAG top five, and ``R1_DEFINITION_1SWAP`` applied to that same official
output and the frozen full typed graph.  All 1,536 action futures are submitted
before any result is joined.  A fresh official-runtime postflight and a private
action seal precede the sole opening of the late-label pack.

The terminal report is descriptive.  It contains arm totals and the ordered
eight seed-cluster differences; it makes no decision or population-inference
claim.
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
from .synthetic_typed_graph_multiseed_acquisition_v1 import (
    ACQUISITION_RECEIPT_RELATIVE_PATH,
    ACTION_PACK_RELATIVE_PATH,
    DESIGN_FILE_SHA256,
    DESIGN_SHA256,
    IMPLEMENTATION_FREEZE_RELATIVE_PATH,
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


VERSION = "synthetic_typed_graph_multiseed_runner_v1"
DESIGN_VERSION = "synthetic_typed_graph_multiseed_replication_v1"
BLOCK = "A_hold"
SEED_COUNT = 8
ITEMS_PER_SEED = 64
TOTAL_ITEMS = SEED_COUNT * ITEMS_PER_SEED
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

ACTION_PACK_SCHEMA = "synthetic_typed_graph_multiseed_action_pack_v1"
ACTION_ITEM_SCHEMA = "synthetic_typed_graph_multiseed_action_item_v1"
LABEL_PACK_SCHEMA = "synthetic_typed_graph_multiseed_label_pack_v1"
LABEL_ITEM_SCHEMA = "synthetic_typed_graph_multiseed_label_item_v1"
FREEZE_SCHEMA = "synthetic_typed_graph_multiseed_replication_implementation_freeze_v1"
FREEZE_STATUS = "complete_preseed_implementation_frozen_must_commit_before_seed"
ACQUISITION_SCHEMA = "synthetic_typed_graph_multiseed_replication_acquisition_v1"
ACQUISITION_STATUS = (
    "formal_multiseed_A_hold_cohort_acquired_private_labels_separated"
)
RESULT_SCHEMA = "synthetic_typed_graph_multiseed_replication_result_v1"
SUCCESS_RESULT_STATUS = "terminal_descriptive_eight_seed_replication_complete"
FAILURE_RESULT_STATUS = (
    "terminal_infrastructure_or_implementation_invalid_no_replay"
)

ARTIFACT_ROOT_RELATIVE_PATH = Path(
    "artifacts/synthetic_typed_graph_multiseed_replication_v1"
)
RUNNER_ROOT_RELATIVE_PATH = ARTIFACT_ROOT_RELATIVE_PATH / "runner"
MARKER_RELATIVE_PATH = RUNNER_ROOT_RELATIVE_PATH / "formal.attempt.marker"
WORK_RELATIVE_PATH = RUNNER_ROOT_RELATIVE_PATH / "formal.work"
ACTION_SEAL_RELATIVE_PATH = RUNNER_ROOT_RELATIVE_PATH / "formal.action.seal.json"
FREEZE_RELATIVE_PATH = IMPLEMENTATION_FREEZE_RELATIVE_PATH
ACQUISITION_RELATIVE_PATH = ACQUISITION_RECEIPT_RELATIVE_PATH
RESULT_RELATIVE_PATH = Path(
    "manifests/synthetic_typed_graph_multiseed_replication_result_v1.json"
)
FAILURE_RELATIVE_PATH = RESULT_RELATIVE_PATH
MINILM_MANIFEST_RELATIVE_PATH = Path("manifests/qasper_minilm_runtime_asset_v1.json")
MINILM_MODEL_ROOT_RELATIVE_PATH = Path("artifacts/qasper_minilm_runtime_v1/model")
OFFICIAL_BASE_RECEIPT_RELATIVE_PATH = Path(
    "manifests/musique_official_hipporag_retrieve_only_binding_v1.json"
)
OFFICIAL_ATTESTATION_RELATIVE_PATH = Path(
    "manifests/musique_official_hipporag_runtime_attestation_v2.json"
)

_FORMAL_ENTRY_ACTIVE = False


class SyntheticTypedGraphMultiseedRunnerError(RuntimeError):
    """A pack, action, runtime, seal, or aggregate invariant failed."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
class MultiseedOutcome:
    action_pack_file_sha256: str
    action_pack_sha256: str
    action_item_commitment_set_sha256: str
    label_pack_file_sha256: str
    label_pack_sha256: str
    label_item_commitment_set_sha256: str
    runtime_binding_sha256: str
    action_table_sha256: str
    action_seal_sha256: str
    action_seal_file_sha256: str
    aggregates: Mapping[str, Any]
    cluster_differences: Mapping[str, Any]


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, private: bool, field: str) -> tuple[dict[str, Any], str]:
    absolute = _assert_no_symlink_components(path, field)
    if not absolute.is_file() or absolute.is_symlink():
        raise SyntheticTypedGraphMultiseedRunnerError(f"{field} is unavailable")
    info = absolute.stat()
    expected_mode = PRIVATE_MODE if private else PUBLIC_MODE
    if stat.S_IMODE(info.st_mode) != expected_mode or not 1 <= info.st_size <= 64 * 1024 * 1024:
        raise SyntheticTypedGraphMultiseedRunnerError(f"{field} mode or size drifted")
    raw = absolute.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyntheticTypedGraphMultiseedRunnerError(f"{field} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise SyntheticTypedGraphMultiseedRunnerError(f"{field} root is not an object")
    if raw != canonical_bytes(payload) + b"\n":
        raise SyntheticTypedGraphMultiseedRunnerError(f"{field} is not canonical JSON")
    return payload, _sha256_bytes(raw)


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    body = dict(payload)
    declared = body.pop(field, None)
    if not isinstance(declared, str) or semantic_hash(body) != declared:
        raise SyntheticTypedGraphMultiseedRunnerError(f"{field} drifted")
    return declared


def _typed_edge(raw: object) -> core.TypedEdge:
    if not isinstance(raw, Mapping) or set(raw) != {
        "edge_family", "left_span_i", "right_span_i"
    }:
        raise SyntheticTypedGraphMultiseedRunnerError("designated edge schema drifted")
    family = raw.get("edge_family")
    left, right = raw.get("left_span_i"), raw.get("right_span_i")
    if (
        family not in core.EDGE_FAMILIES
        or type(left) is not int
        or type(right) is not int
        or not 0 <= left < right < grammar.NODE_COUNT
    ):
        raise SyntheticTypedGraphMultiseedRunnerError("designated edge content drifted")
    return core.TypedEdge(core.EDGE_FAMILY_ORDER[str(family)], left, right)


def _parse_action_item(raw: object, ordinal: int) -> ActionItem:
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema", "global_ordinal", "seed_index", "seed_ordinal", "question",
        "context", "nodes", "designated_edges", "action_item_sha256",
    }:
        raise SyntheticTypedGraphMultiseedRunnerError("action item field set drifted")
    if raw.get("schema") != ACTION_ITEM_SCHEMA:
        raise SyntheticTypedGraphMultiseedRunnerError("action item schema drifted")
    expected_seed, expected_within = divmod(ordinal, ITEMS_PER_SEED)
    if (
        raw.get("global_ordinal") != ordinal
        or raw.get("seed_index") != expected_seed
        or raw.get("seed_ordinal") != expected_within
    ):
        raise SyntheticTypedGraphMultiseedRunnerError("action item coordinate drifted")
    declared = _self_hash(raw, "action_item_sha256")
    question, context = raw.get("question"), raw.get("context")
    if not isinstance(question, str) or not question or not isinstance(context, str):
        raise SyntheticTypedGraphMultiseedRunnerError("action item text drifted")
    node_rows = raw.get("nodes")
    if not isinstance(node_rows, list) or len(node_rows) != grammar.NODE_COUNT:
        raise SyntheticTypedGraphMultiseedRunnerError("action node count drifted")
    nodes: list[ActionNode] = []
    previous_end = 0
    for index, node in enumerate(node_rows):
        if not isinstance(node, Mapping) or set(node) != {
            "span_i", "start", "end", "identity_text"
        }:
            raise SyntheticTypedGraphMultiseedRunnerError("action node schema drifted")
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
            raise SyntheticTypedGraphMultiseedRunnerError("action node content drifted")
        nodes.append(ActionNode(index, start, end, text))
        previous_end = end
    edge_rows = raw.get("designated_edges")
    if not isinstance(edge_rows, list) or not edge_rows:
        raise SyntheticTypedGraphMultiseedRunnerError("designated edge table drifted")
    designated = tuple(_typed_edge(edge) for edge in edge_rows)
    if tuple(sorted(set(designated))) != designated:
        raise SyntheticTypedGraphMultiseedRunnerError("designated edges are not canonical")
    spans = tuple(node.source_span() for node in nodes)
    full = core.build_typed_clause_graph(spans)
    if not set(designated).issubset(full):
        raise SyntheticTypedGraphMultiseedRunnerError(
            "designated edges are absent from the frozen full graph"
        )
    return ActionItem(
        ordinal,
        expected_seed,
        expected_within,
        question,
        context,
        tuple(nodes),
        designated,
        full,
        declared,
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
        raise SyntheticTypedGraphMultiseedRunnerError("action pack schema drifted")
    parsed = tuple(_parse_action_item(row, ordinal) for ordinal, row in enumerate(rows))
    hashes = [row.action_item_sha256 for row in parsed]
    if len(set(hashes)) != TOTAL_ITEMS:
        raise SyntheticTypedGraphMultiseedRunnerError("action item commitments overlap")
    return ActionPack(declared, file_sha256, semantic_hash(hashes), parsed)


def _expected_family_by_seed_ordinal() -> tuple[str, ...]:
    expected: list[str] = []
    for family_id, count in grammar.family_quota(BLOCK):
        expected.extend([family_id] * count)
    if len(expected) != ITEMS_PER_SEED:
        raise SyntheticTypedGraphMultiseedRunnerError("frozen family schedule drifted")
    return tuple(expected)


def _parse_label_item(raw: object, ordinal: int) -> LabelItem:
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema", "global_ordinal", "seed_index", "seed_ordinal",
        "action_item_sha256", "gold_node_indices", "family_id", "family_role",
        "polarity", "edge_family", "label_item_sha256",
    }:
        raise SyntheticTypedGraphMultiseedRunnerError("label item field set drifted")
    if raw.get("schema") != LABEL_ITEM_SCHEMA:
        raise SyntheticTypedGraphMultiseedRunnerError("label item schema drifted")
    expected_seed, expected_within = divmod(ordinal, ITEMS_PER_SEED)
    if (
        raw.get("global_ordinal") != ordinal
        or raw.get("seed_index") != expected_seed
        or raw.get("seed_ordinal") != expected_within
    ):
        raise SyntheticTypedGraphMultiseedRunnerError("label item coordinate drifted")
    declared = _self_hash(raw, "label_item_sha256")
    gold = raw.get("gold_node_indices")
    family_id = raw.get("family_id")
    if (
        not _is_sha256(raw.get("action_item_sha256"))
        or not isinstance(gold, list)
        or not 1 <= len(gold) <= 3
        or gold != sorted(set(gold))
        or any(type(index) is not int or not 0 <= index < grammar.NODE_COUNT for index in gold)
        or family_id not in grammar.FAMILY_BY_ID
        or family_id != _expected_family_by_seed_ordinal()[expected_within]
    ):
        raise SyntheticTypedGraphMultiseedRunnerError("label item content drifted")
    family = grammar.FAMILY_BY_ID[str(family_id)]
    if (
        raw.get("family_role") != family.family_role
        or raw.get("polarity") != family.polarity
        or raw.get("edge_family") != family.edge_family
    ):
        raise SyntheticTypedGraphMultiseedRunnerError("label family registry drifted")
    return LabelItem(
        ordinal,
        expected_seed,
        expected_within,
        str(raw["action_item_sha256"]),
        tuple(gold),
        str(family_id),
        str(raw["family_role"]),
        str(raw["polarity"]),
        str(raw["edge_family"]),
        declared,
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
        raise SyntheticTypedGraphMultiseedRunnerError("late label pack schema drifted")
    parsed = tuple(_parse_label_item(row, ordinal) for ordinal, row in enumerate(rows))
    hashes = [row.label_item_sha256 for row in parsed]
    if len(set(hashes)) != TOTAL_ITEMS:
        raise SyntheticTypedGraphMultiseedRunnerError("label item commitments overlap")
    return LabelPack(declared, file_sha256, semantic_hash(hashes), parsed)


def _validated_top5(value: Sequence[int], field: str) -> tuple[int, int, int, int, int]:
    rows = tuple(value)
    if (
        len(rows) != TOP_K
        or len(set(rows)) != TOP_K
        or any(type(index) is not int or not 0 <= index < grammar.NODE_COUNT for index in rows)
    ):
        raise SyntheticTypedGraphMultiseedRunnerError(f"{field} top5 drifted")
    return rows  # type: ignore[return-value]


def precompute_local_tensors(
    pack: ActionPack, encoder: EncoderProtocol
) -> tuple[LocalTensor, ...]:
    """Compute the original frozen offline MiniLM RAW semantics in one batch."""

    if len(pack.rows) != TOTAL_ITEMS:
        raise SyntheticTypedGraphMultiseedRunnerError("action pack size drifted")
    texts: list[str] = []
    starts: list[int] = []
    for item in pack.rows:
        starts.append(len(texts))
        texts.append(item.question)
        texts.extend(core.embedding_text(node.identity_text) for node in item.nodes)
    try:
        matrix = np.asarray(encoder.encode(tuple(texts)), dtype=np.float32)
    except Exception as exc:
        raise SyntheticTypedGraphMultiseedRunnerError("offline MiniLM batch failed") from exc
    if matrix.ndim != 2 or matrix.shape[0] != len(texts) or not np.isfinite(matrix).all():
        raise SyntheticTypedGraphMultiseedRunnerError("offline MiniLM output drifted")
    tensors: list[LocalTensor] = []
    for start in starts:
        query = matrix[start]
        nodes = matrix[start + 1 : start + 1 + grammar.NODE_COUNT]
        similarities = tuple(
            quantized_cosine_similarity(query, node) for node in nodes
        )
        raw = _validated_top5(
            sorted(
                range(grammar.NODE_COUNT),
                key=lambda index: (-similarities[index], index),
            )[:TOP_K],
            "RAW",
        )
        tensor_hash = semantic_hash(
            {"raw_top5": list(raw), "query_similarities": list(similarities)}
        )
        tensors.append(LocalTensor(raw, similarities, tensor_hash))
    return tuple(tensors)


def _runtime_binding(runtime: OfficialRuntimeProtocol) -> str:
    safe = dict(runtime.safe_binding)
    encoded = json.dumps(safe, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if "/home/" in encoded or "/tmp/" in encoded or "\\" in encoded:
        raise SyntheticTypedGraphMultiseedRunnerError(
            "official runtime safe binding leaks a host path"
        )
    return semantic_hash(safe)


def _official_action(
    runtime: OfficialRuntimeProtocol, item: ActionItem, work_root: Path
) -> tuple[int, int, int, int, int]:
    return _validated_top5(
        runtime.retrieve(
            question=item.question,
            paragraphs=item.paragraphs,
            work_root=work_root,
        ),
        "official HippoRAG",
    )


def _raw_action(tensor: LocalTensor) -> tuple[int, int, int, int, int]:
    return _validated_top5(tensor.raw_top5, "RAW")


def _agent_action(
    item: ActionItem,
    tensor: LocalTensor,
    official_future: Future[tuple[int, int, int, int, int]],
    submission_released: threading.Event,
    all_actions_submitted: threading.Event,
) -> tuple[tuple[int, int, int, int, int], str]:
    submission_released.wait()
    if not all_actions_submitted.is_set():
        raise SyntheticTypedGraphMultiseedRunnerError(
            "agent action released after an incomplete submission wave"
        )
    official = official_future.result()
    table = core.build_common_candidate_table(
        item.spans, item.full_edges, official, tensor.query_similarities
    )
    trace = core.execute_recipe(official, table, tensor.query_similarities, RECIPE_ID)
    if trace.recipe_id != RECIPE_ID:
        raise SyntheticTypedGraphMultiseedRunnerError("fixed R1 action drifted")
    return _validated_top5(trace.output_top5, AGENT_ARM), trace.common_scan_sha256


def _execute_all_actions(
    pack: ActionPack,
    tensors: Sequence[LocalTensor],
    runtime: OfficialRuntimeProtocol,
    work_root: Path,
) -> tuple[tuple[ItemActions, ...], str]:
    """Eagerly submit and then join all 1,536 fixed action work units."""

    if work_root.exists() or work_root.is_symlink():
        raise SyntheticTypedGraphMultiseedRunnerError("official work root already exists")
    work_root.mkdir(parents=True, mode=0o700)
    if len(tensors) != TOTAL_ITEMS:
        raise SyntheticTypedGraphMultiseedRunnerError("local tensor count drifted")
    preflight_binding = _runtime_binding(runtime)
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
        # Submission is deliberately split into three complete waves.  No
        # Future is joined until all 1,536 work units have been accepted.
        try:
            for item in pack.rows:
                official_futures.append(
                    official_pool.submit(
                        _official_action,
                        runtime,
                        item,
                        work_root / f"item_{item.global_ordinal:03d}",
                    )
                )
            for tensor in tensors:
                raw_futures.append(local_pool.submit(_raw_action, tensor))
            for item, tensor, official_future in zip(pack.rows, tensors, official_futures):
                agent_futures.append(
                    local_pool.submit(
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
                raise SyntheticTypedGraphMultiseedRunnerError(
                    "action submission count drifted"
                )
            all_actions_submitted.set()
        finally:
            # Release already-started Agent workers even if a submit raises.
            # They fail closed without joining an official Future unless the
            # complete 1,536-work-unit wave was accepted.
            submission_released.set()
        for index, future in enumerate(official_futures):
            try:
                official_results[index] = future.result()
            except BaseException as exc:  # join every work unit before surfacing one error
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
        postflight_binding = semantic_hash(postflight)
        if postflight_binding != preflight_binding:
            errors.append(
                SyntheticTypedGraphMultiseedRunnerError(
                    "official runtime postflight binding drifted"
                )
            )
    except BaseException as exc:
        errors.append(exc)
        postflight_binding = ""
    if errors:
        raise SyntheticTypedGraphMultiseedRunnerError(
            f"label-free action or official postflight failed ({len(errors)} terminal errors)"
        ) from errors[0]
    if any(value is None for value in (*official_results, *raw_results, *agent_results)):
        raise SyntheticTypedGraphMultiseedRunnerError("action completion barrier drifted")
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
    return tuple(actions), postflight_binding


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
    actions: Sequence[ItemActions],
    runtime_binding_sha256: str,
) -> tuple[str, str, str]:
    rows = _action_rows(actions)
    action_table_sha256 = semantic_hash(rows)
    body = {
        "schema": f"{VERSION}_private_action_seal",
        "version": VERSION,
        "status": "all_1536_actions_joined_official_postflight_terminal",
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
        "action_pack_file_sha256": pack.file_sha256,
        "action_pack_sha256": pack.pack_sha256,
        "action_item_commitment_set_sha256": pack.item_commitment_set_sha256,
        "runtime_binding_sha256": runtime_binding_sha256,
        "action_table_sha256": action_table_sha256,
        "action_rows": list(rows),
        "labels_opened_before_action_seal": False,
        "labels_opened_before_seal": False,
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
        raise SyntheticTypedGraphMultiseedRunnerError("action row count drifted")
    joined = tuple(zip(action_pack.rows, label_pack.rows, actions))
    if any(
        item.global_ordinal != label.global_ordinal
        or item.global_ordinal != action.global_ordinal
        or item.action_item_sha256 != label.action_item_sha256
        or item.action_item_sha256 != action.action_item_sha256
        for item, label, action in joined
    ):
        raise SyntheticTypedGraphMultiseedRunnerError("action/label identity join drifted")
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
            output, label.gold_node_indices, source_count=grammar.NODE_COUNT
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
        raise SyntheticTypedGraphMultiseedRunnerError("joined cohort size drifted")
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
    """Run the fixed comparison; invoke ``label_loader`` once after the seal."""

    if action_seal_path.exists() or action_seal_path.is_symlink():
        raise SyntheticTypedGraphMultiseedRunnerError("action seal already exists")
    tensors = precompute_local_tensors(action_pack, encoder)
    actions, runtime_binding = _execute_all_actions(
        action_pack, tensors, runtime, work_root
    )
    seal_sha256, seal_file_sha256, action_table_sha256 = _persist_action_seal(
        path=action_seal_path,
        pack=action_pack,
        actions=actions,
        runtime_binding_sha256=runtime_binding,
    )
    # This is the sole late-label opening point.  Nothing label-derived is
    # available to any action or to the official postflight/seal barrier.
    labels = label_loader()
    if not isinstance(labels, LabelPack):
        raise SyntheticTypedGraphMultiseedRunnerError("late label loader returned wrong type")
    joined = _join(action_pack, labels, actions)
    aggregates, differences = _aggregate_arms(joined)
    return MultiseedOutcome(
        action_pack.file_sha256,
        action_pack.pack_sha256,
        action_pack.item_commitment_set_sha256,
        labels.file_sha256,
        labels.pack_sha256,
        labels.item_commitment_set_sha256,
        runtime_binding,
        action_table_sha256,
        seal_sha256,
        seal_file_sha256,
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
        "action_pack_file_sha256": outcome.action_pack_file_sha256,
        "action_pack_sha256": outcome.action_pack_sha256,
        "action_item_commitment_set_sha256": outcome.action_item_commitment_set_sha256,
        "label_pack_file_sha256": outcome.label_pack_file_sha256,
        "label_pack_sha256": outcome.label_pack_sha256,
        "label_item_commitment_set_sha256": outcome.label_item_commitment_set_sha256,
        "runtime_binding_sha256": outcome.runtime_binding_sha256,
        "action_table_sha256": outcome.action_table_sha256,
        "action_seal_sha256": outcome.action_seal_sha256,
        "action_seal_file_sha256": outcome.action_seal_file_sha256,
        "aggregates": dict(outcome.aggregates),
        "cluster_differences": dict(outcome.cluster_differences),
        "interpretation": "descriptive_fixed_cohort_replication_only",
        "seeds_or_item_rows_disclosed": False,
    }
    return {**body, "receipt_sha256": semantic_hash(body)}


def _load_committed_acquisition_metadata(
    root: Path, freeze: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    """Validate only committed public metadata, never either private pack."""

    path = root / ACQUISITION_RELATIVE_PATH
    payload, file_sha256 = _read_json(
        path, private=False, field="committed acquisition receipt"
    )
    try:
        committed = _committed_bytes(root, ACQUISITION_RELATIVE_PATH)
    except Exception as exc:
        raise SyntheticTypedGraphMultiseedRunnerError(
            "acquisition receipt is not current-HEAD committed"
        ) from exc
    if committed != path.read_bytes():
        raise SyntheticTypedGraphMultiseedRunnerError(
            "acquisition receipt differs from current HEAD"
        )
    body = dict(payload)
    declared = body.pop("receipt_sha256", None)
    commitments = payload.get("commitments")
    if (
        payload.get("schema") != ACQUISITION_SCHEMA
        or payload.get("status") != ACQUISITION_STATUS
        or payload.get("design_sha256") != DESIGN_SHA256
        or payload.get("implementation_freeze_sha256")
        != freeze.get("implementation_freeze_sha256")
        or payload.get("block") != BLOCK
        or payload.get("seed_count") != SEED_COUNT
        or payload.get("item_count_per_seed") != ITEMS_PER_SEED
        or payload.get("total_item_count") != TOTAL_ITEMS
        or payload.get("fixed_recipe_id") != RECIPE_ID
        or payload.get("arms") != list(ARM_IDS)
        or not isinstance(commitments, Mapping)
        or set(commitments) != {
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
        raise SyntheticTypedGraphMultiseedRunnerError(
            "committed acquisition metadata drifted"
        )
    return payload, file_sha256


def _canonical_formal_paths(root: Path) -> dict[str, Path]:
    return {
        "marker": root / MARKER_RELATIVE_PATH,
        "work": root / WORK_RELATIVE_PATH,
        "seal": root / ACTION_SEAL_RELATIVE_PATH,
        "result": root / RESULT_RELATIVE_PATH,
        "failure": root / FAILURE_RELATIVE_PATH,
    }


def _consume_formal_marker(
    *,
    root: Path,
    actual_head: str,
    freeze: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    acquisition_file_sha256: str,
) -> tuple[dict[str, Any], str, dict[str, Path]]:
    paths = _canonical_formal_paths(root)
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise SyntheticTypedGraphMultiseedRunnerError(
            "canonical multiseed runner attempt already exists"
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
        raise SyntheticTypedGraphMultiseedRunnerError(
            f"{prefix} pack differs from committed acquisition"
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
    """Consume the sole canonical replication attempt and persist its result."""

    if _FORMAL_ENTRY_ACTIVE is not True:
        raise SyntheticTypedGraphMultiseedRunnerError(
            "canonical replication may only be consumed by the formal CLI"
        )
    if not isinstance(encoder, OfflineMiniLMEncoder) or not isinstance(
        runtime, PreparedFormalRuntimeV2
    ):
        raise SyntheticTypedGraphMultiseedRunnerError(
            "canonical replication resources are not attested formal types"
        )
    root = project_root.resolve(strict=True)
    freeze, actual_head = verify_implementation_freeze(root)
    # Public current-HEAD metadata is the only acquisition material read before
    # the durable attempt marker.  This helper deliberately cannot open packs.
    acquisition, acquisition_file_sha256 = _load_committed_acquisition_metadata(
        root, freeze
    )
    marker, marker_file_sha256, paths = _consume_formal_marker(
        root=root,
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
        label_load_count = 0

        def load_late_labels() -> LabelPack:
            nonlocal label_load_count
            label_load_count += 1
            if label_load_count != 1:
                raise SyntheticTypedGraphMultiseedRunnerError(
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
        if label_load_count != 1:
            raise SyntheticTypedGraphMultiseedRunnerError(
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
            path=paths["failure"],
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
        raise SyntheticTypedGraphMultiseedRunnerError("formal entry is already active")
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
