"""End-to-end non-claim TRAIN runtime integration for FiQA P10.

The stage embeds the filtered FiQA corpus, runs the frozen offline typed-query
generator, expands the pool with deterministic bridge queries, and executes
P10, RAW, and candidate-restricted graph-bearing HippoRAG.  TRAIN labels are
opened only after every action is sealed.  DEV and TEST qrels are never read.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from reconstruction_v2.assumption_agent.benchmarks import (
    bright_bridge_expansion_core_v1 as bridge,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    bright_reasoning_retrieval_core_v1 as core,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    bright_reasoning_retrieval_study_v1 as bright_runtime,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    fiqa_bridge_expansion_train_integration_v1 as integration_v1,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    fiqa_bridge_expansion_train_integration_v2 as integration_v2,
)
from reconstruction_v2.replication_runtime.bridge_expanded_cross_encoder_v1 import (
    contract as cross_contract,
)
from reconstruction_v2.replication_runtime.bridge_expanded_cross_encoder_v1 import (
    worker as cross_worker,
)
from reconstruction_v2.replication_runtime.bright_minilm_v1.encoder import (
    float32_matrix_sha256,
    quantized_scores,
)
from reconstruction_v2.replication_runtime.bright_official_hipporag_v1 import (
    contract as hippo_contract,
)
from reconstruction_v2.replication_runtime.bright_query_generator_v1 import (
    contract as qwen_contract,
)


SCHEMA = "fiqa_bridge_expansion_train_runtime_result_v1"
ATTEMPT_SCHEMA = "fiqa_bridge_expansion_train_runtime_attempt_v1"
ACTION_SCHEMA = "fiqa_bridge_expansion_train_runtime_actions_v1"
INTENT_SCHEMA = "fiqa_bridge_expansion_train_runtime_intents_v1"
FREEZE_SCHEMA = "fiqa_bridge_expansion_train_runtime_implementation_freeze_v1"
ITEM_COUNT = 12
HIPPORAG_CONCURRENCY = 12
EXTERNAL_PROCESS_CONCURRENCY = 13

FREEZE_RELATIVE = Path(
    "manifests/fiqa_bridge_expansion_train_runtime_implementation_freeze_v1.json"
)
RESULT_RELATIVE = Path("manifests/fiqa_bridge_expansion_train_runtime_result_v1.json")
INTEGRATION_RESULT_RELATIVE = Path(
    "manifests/fiqa_bridge_expansion_train_integration_result_v2.json"
)
RUNTIME_QUALIFICATION_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_runtime_qualification_v1.json"
)
CROSS_ASSET_RELATIVE = Path("manifests/bright_cross_encoder_runtime_asset_v1.json")
RUN_ROOT_RELATIVE = Path("artifacts/fiqa_bridge_expansion_train_runtime_v1")
TRAIN_SOURCE_ROOT_RELATIVE = Path(
    "artifacts/fiqa_bridge_expansion_train_integration_v2"
)
CROSS_MODEL_RELATIVE = Path("artifacts/bright_cross_encoder_runtime_v1/model")

INTEGRATION_RESULT_FILE_SHA256 = (
    "ff24838e9a238c606462b7142cf29571435a63226a8559a02cedd5bdf7c30890"
)
INTEGRATION_RESULT_SELF_SHA256 = (
    "c194ed16cd83e89b01a1058dbde5f77a4139671893a5170332959953badeb032"
)
RUNTIME_QUALIFICATION_FILE_SHA256 = (
    "630d47f5f1d9bdab7d456ad437dec3e39d45378672ffffa3eee61b633e72708e"
)
RUNTIME_QUALIFICATION_SELF_SHA256 = (
    "80f4e846f3a1ad9ad2c1bd84d9df02aebd386074da628b06def9516422a98d18"
)
CROSS_ASSET_FILE_SHA256 = (
    "cbb90ef21571b94e41e3fcb501228dbd130edcba87dcd40f95f15d7e805c133c"
)
CROSS_ASSET_SELF_SHA256 = (
    "56c550fd1224096dad64ebf7ed5ae8552d55ee8a1216376f39b1dc11be32ff43"
)

REQUIRED_IMPLEMENTATION_RELATIVES = (
    Path("assumption_agent/benchmarks/fiqa_bridge_expansion_train_runtime_v1.py"),
    Path("tests/test_fiqa_bridge_expansion_train_runtime_v1.py"),
    Path("replication_runtime/bridge_expanded_cross_encoder_v1/__init__.py"),
    Path("replication_runtime/bridge_expanded_cross_encoder_v1/contract.py"),
    Path("replication_runtime/bridge_expanded_cross_encoder_v1/worker.py"),
    Path("tests/test_bridge_expanded_cross_encoder_v1.py"),
)


class FiqaTrainRuntimeError(RuntimeError):
    """The frozen FiQA TRAIN runtime integration failed closed."""


class OneShotRefusal(FiqaTrainRuntimeError):
    """The TRAIN runtime integration attempt or result already exists."""


@dataclass(frozen=True)
class ViewItem:
    ordinal: int
    item_key: str
    query: str
    excluded_ids: tuple[str, ...]


@dataclass(frozen=True)
class Corpus:
    ids: tuple[str, ...]
    contents: tuple[str, ...]
    embeddings: np.ndarray


@dataclass(frozen=True)
class LocalPlan:
    item: ViewItem
    base_pool: tuple[int, ...]
    raw_rows: tuple[int, ...]
    original_scores: np.ndarray
    relation_scores: np.ndarray
    mechanism_scores: np.ndarray
    relation_query: str
    mechanism_query: str
    seed_rows: tuple[int, ...]
    anchors: tuple[bridge.BridgeAnchor, ...]
    bridge_queries: tuple[bridge.BridgeQuery, ...]
    excluded_rows: tuple[int, ...]


@dataclass(frozen=True)
class ExpandedPlan:
    local: LocalPlan
    expanded: bridge.ExpandedCandidatePool


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FiqaTrainRuntimeError(f"{name} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FiqaTrainRuntimeError(f"{name} is invalid") from exc
    if not isinstance(value, Mapping):
        raise FiqaTrainRuntimeError(f"{name} root drifted")
    return value


def _verify_self(value: Mapping[str, Any], field: str, expected: str) -> None:
    body = dict(value)
    declared = body.pop(field, None)
    if (
        declared != expected
        or hashlib.sha256(integration_v1.canonical_json(body)).hexdigest() != expected
    ):
        raise FiqaTrainRuntimeError(f"{field} drifted")


def _load_bound_json(
    base: Path,
    relative: Path,
    *,
    file_sha256: str,
    self_field: str,
    self_sha256: str,
) -> Mapping[str, Any]:
    path = base / relative
    if integration_v1.file_sha256(path) != file_sha256:
        raise FiqaTrainRuntimeError(f"{relative.name} file binding drifted")
    value = _read_json(path, relative.name)
    _verify_self(value, self_field, self_sha256)
    return value


def _verify_freeze(base: Path) -> Mapping[str, Any]:
    value = _read_json(base / FREEZE_RELATIVE, "TRAIN runtime freeze")
    if value.get("schema") != FREEZE_SCHEMA:
        raise FiqaTrainRuntimeError("TRAIN runtime freeze schema drifted")
    declared = value.get("self_sha256")
    if not isinstance(declared, str):
        raise FiqaTrainRuntimeError("TRAIN runtime freeze identity is absent")
    _verify_self(value, "self_sha256", declared)
    rows = value.get("implementation_bindings")
    if not isinstance(rows, list):
        raise FiqaTrainRuntimeError("TRAIN runtime freeze bindings drifted")
    observed = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise FiqaTrainRuntimeError("TRAIN runtime freeze row drifted")
        relative = row.get("relative_path")
        digest = row.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise FiqaTrainRuntimeError("TRAIN runtime freeze value drifted")
        observed[relative] = digest
    expected_paths = {path.as_posix() for path in REQUIRED_IMPLEMENTATION_RELATIVES}
    if set(observed) != expected_paths:
        raise FiqaTrainRuntimeError("TRAIN runtime freeze file set drifted")
    for relative, expected in observed.items():
        if integration_v1.file_sha256(base / relative) != expected:
            raise FiqaTrainRuntimeError("TRAIN runtime implementation drifted")
    if value.get("integration_result_self_sha256") != INTEGRATION_RESULT_SELF_SHA256:
        raise FiqaTrainRuntimeError("TRAIN runtime integration binding drifted")
    return value


def _load_preconditions(project_root: Path) -> dict[str, Mapping[str, Any]]:
    base = project_root / "reconstruction_v2"
    for relative in integration_v1.MANIFEST_BINDINGS:
        integration_v1._load_manifest(project_root, relative)
    integration_v2._load_failure_v1(project_root)
    integration = _load_bound_json(
        base,
        INTEGRATION_RESULT_RELATIVE,
        file_sha256=INTEGRATION_RESULT_FILE_SHA256,
        self_field="integration_sha256",
        self_sha256=INTEGRATION_RESULT_SELF_SHA256,
    )
    runtime = _load_bound_json(
        base,
        RUNTIME_QUALIFICATION_RELATIVE,
        file_sha256=RUNTIME_QUALIFICATION_FILE_SHA256,
        self_field="self_sha256",
        self_sha256=RUNTIME_QUALIFICATION_SELF_SHA256,
    )
    cross_asset = _load_bound_json(
        base,
        CROSS_ASSET_RELATIVE,
        file_sha256=CROSS_ASSET_FILE_SHA256,
        self_field="asset_sha256",
        self_sha256=CROSS_ASSET_SELF_SHA256,
    )
    freeze = _verify_freeze(base)
    return {
        "cross_asset": cross_asset,
        "freeze": freeze,
        "integration": integration,
        "runtime": runtime,
    }


def _parse_canonical_jsonl(path: Path, name: str) -> list[Mapping[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise FiqaTrainRuntimeError(f"{name} is unavailable")
    rows: list[Mapping[str, Any]] = []
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FiqaTrainRuntimeError(f"{name} read failed") from exc
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise FiqaTrainRuntimeError(f"{name} is not canonical JSONL")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FiqaTrainRuntimeError(f"{name} JSONL is invalid") from exc
        if not isinstance(value, Mapping) or integration_v1.canonical_json(value) + b"\n" != line:
            raise FiqaTrainRuntimeError(f"{name} row drifted")
        rows.append(value)
    return rows


def load_filtered_corpus(base: Path, integration: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    binding = integration.get("filtered_corpus_binding")
    if not isinstance(binding, Mapping):
        raise FiqaTrainRuntimeError("filtered corpus binding drifted")
    relative = binding.get("relative_path")
    if not isinstance(relative, str):
        raise FiqaTrainRuntimeError("filtered corpus path drifted")
    path = base / relative
    if (
        path.stat().st_size != binding.get("size_bytes")
        or integration_v1.file_sha256(path) != binding.get("sha256")
    ):
        raise FiqaTrainRuntimeError("filtered corpus file drifted")
    rows = _parse_canonical_jsonl(path, "filtered corpus")
    ids: list[str] = []
    contents: list[str] = []
    for row in rows:
        if set(row) != {"_id", "title", "text"}:
            raise FiqaTrainRuntimeError("filtered corpus row shape drifted")
        identifier = integration_v1._required_text(row.get("_id"), "document ID")
        title = integration_v1._required_text(
            row.get("title"), "document title", allow_empty=True
        )
        body = integration_v1._required_text(
            row.get("text"), "document text", allow_empty=True
        )
        content = "\n".join(value.strip() for value in (title, body) if value.strip())
        if not content:
            raise FiqaTrainRuntimeError("filtered corpus contains an empty document")
        ids.append(identifier)
        contents.append(content[: bright_runtime.DOCUMENT_TEXT_CHARACTERS])
    aggregates = integration.get("source_aggregates")
    if (
        not isinstance(aggregates, Mapping)
        or len(ids) != aggregates.get("usable_corpus_document_count")
        or len(set(ids)) != len(ids)
    ):
        raise FiqaTrainRuntimeError("filtered corpus identity drifted")
    return tuple(ids), tuple(contents)


def load_train_pack(base: Path, integration: Mapping[str, Any]) -> tuple[tuple[ViewItem, ...], Mapping[str, tuple[str, ...]]]:
    binding = integration.get("TRAIN_diagnostic_pack")
    if not isinstance(binding, Mapping) or binding.get("item_count") != ITEM_COUNT:
        raise FiqaTrainRuntimeError("TRAIN diagnostic pack binding drifted")
    root = base / TRAIN_SOURCE_ROOT_RELATIVE
    view_path = root / "train_integration.view.jsonl"
    label_path = root / "train_integration.labels.jsonl"
    if (
        view_path.stat().st_size != binding.get("view_file_size_bytes")
        or label_path.stat().st_size != binding.get("label_file_size_bytes")
        or integration_v1.file_sha256(view_path) != binding.get("view_file_sha256")
        or integration_v1.file_sha256(label_path) != binding.get("label_file_sha256")
    ):
        raise FiqaTrainRuntimeError("TRAIN diagnostic pack files drifted")
    view_rows = _parse_canonical_jsonl(view_path, "TRAIN view pack")
    label_rows = _parse_canonical_jsonl(label_path, "TRAIN label pack")
    if len(view_rows) != ITEM_COUNT or len(label_rows) != ITEM_COUNT:
        raise FiqaTrainRuntimeError("TRAIN pack row count drifted")
    items: list[ViewItem] = []
    labels: dict[str, tuple[str, ...]] = {}
    for ordinal, (view, label) in enumerate(zip(view_rows, label_rows)):
        if set(view) != {
            "excluded_document_ids",
            "family",
            "item_key",
            "query",
            "source_query_id",
        } or set(label) != {"family", "gold_document_ids", "item_key"}:
            raise FiqaTrainRuntimeError("TRAIN pack row shape drifted")
        item_key = integration_v1._required_text(view.get("item_key"), "item key")
        if (
            view.get("family") != "FIQA"
            or label.get("family") != "FIQA"
            or label.get("item_key") != item_key
            or item_key in labels
        ):
            raise FiqaTrainRuntimeError("TRAIN pack identity drifted")
        excluded_raw = view.get("excluded_document_ids")
        gold_raw = label.get("gold_document_ids")
        if not isinstance(excluded_raw, list) or not isinstance(gold_raw, list) or not gold_raw:
            raise FiqaTrainRuntimeError("TRAIN pack list field drifted")
        excluded = tuple(
            integration_v1._required_text(value, "excluded document ID")
            for value in excluded_raw
        )
        gold = tuple(
            integration_v1._required_text(value, "gold document ID") for value in gold_raw
        )
        if len(set(excluded)) != len(excluded) or len(set(gold)) != len(gold):
            raise FiqaTrainRuntimeError("TRAIN pack IDs are duplicated")
        items.append(
            ViewItem(
                ordinal=ordinal,
                item_key=item_key,
                query=integration_v1._required_text(view.get("query"), "query"),
                excluded_ids=excluded,
            )
        )
        labels[item_key] = gold
    return tuple(items), labels


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def build_local_plan(
    *,
    item: ViewItem,
    document_ids: Sequence[str],
    document_contents: Sequence[str],
    query_score_vectors: Sequence[np.ndarray],
    expansions: Sequence[str],
) -> LocalPlan:
    if len(query_score_vectors) != 5 or len(expansions) != 4:
        raise FiqaTrainRuntimeError("typed query count drifted")
    if tuple(qwen_contract.EXPANSION_KEYS) != (
        "entity_query",
        "relation_query",
        "mechanism_query",
        "constraint_query",
    ):
        raise FiqaTrainRuntimeError("typed query registry drifted")
    id_to_row = {identifier: index for index, identifier in enumerate(document_ids)}
    excluded_rows = tuple(
        id_to_row[identifier] for identifier in item.excluded_ids if identifier in id_to_row
    )
    local = core.build_local_retrieval(
        query_score_vectors,
        excluded_rows=excluded_rows,
    )
    relation = np.asarray(query_score_vectors[2])
    mechanism = np.asarray(query_score_vectors[3])
    seeds = bridge.select_seed_rows(
        local.candidate_rows,
        [int(relation[row]) for row in local.candidate_rows],
        [int(mechanism[row]) for row in local.candidate_rows],
    )
    anchors = bridge.extract_bridge_anchors(
        original_query=item.query,
        relation_query=expansions[1],
        mechanism_query=expansions[2],
        seed_rows=seeds,
        documents_by_row={row: document_contents[row] for row in seeds},
    )
    bridge_queries = bridge.build_bridge_queries(
        relation_query=expansions[1],
        mechanism_query=expansions[2],
        anchors=anchors,
    )
    return LocalPlan(
        item=item,
        base_pool=local.candidate_rows,
        raw_rows=local.raw_rows,
        original_scores=np.asarray(query_score_vectors[0]),
        relation_scores=relation,
        mechanism_scores=mechanism,
        relation_query=expansions[1],
        mechanism_query=expansions[2],
        seed_rows=seeds,
        anchors=anchors,
        bridge_queries=bridge_queries,
        excluded_rows=excluded_rows,
    )


def expand_plan(local: LocalPlan, bridge_score_vectors: Sequence[np.ndarray]) -> ExpandedPlan:
    if len(bridge_score_vectors) != len(local.bridge_queries):
        raise FiqaTrainRuntimeError("bridge score count drifted")
    expanded = bridge.expand_candidate_pool(
        base_pool=local.base_pool,
        bridge_score_vectors=bridge_score_vectors,
        excluded_rows=local.excluded_rows,
    )
    return ExpandedPlan(local=local, expanded=expanded)


def build_cross_input(
    plans: Sequence[ExpandedPlan],
    contents: Sequence[str],
) -> dict[str, Any]:
    rows = []
    for ordinal, plan in enumerate(plans):
        if plan.local.item.ordinal != ordinal:
            raise FiqaTrainRuntimeError("expanded plan ordinal drifted")
        rows.append(
            {
                "documents": [
                    {"content": contents[row], "ordinal": document_ordinal}
                    for document_ordinal, row in enumerate(plan.expanded.expanded_pool)
                ],
                "mechanism_query": plan.local.mechanism_query,
                "ordinal": ordinal,
                "relation_query": plan.local.relation_query,
            }
        )
    return cross_contract.input_payload(rows)


def _prepare_hipporag_inputs(
    *,
    root: Path,
    plans: Sequence[ExpandedPlan],
    contents: Sequence[str],
) -> tuple[Path, ...]:
    hippo_root = root / "hipporag"
    hippo_root.mkdir(mode=0o700)
    roots: list[Path] = []
    for plan in plans:
        item_root = hippo_root / f"item_{plan.local.item.ordinal:03d}"
        item_root.mkdir(mode=0o700)
        for name in ("home", "hf", "tmp"):
            (item_root / name).mkdir(mode=0o700)
        payload = {
            "documents": [
                {"content": contents[row], "ordinal": ordinal}
                for ordinal, row in enumerate(plan.local.base_pool)
            ],
            "query": plan.local.item.query,
            "schema": hippo_contract.INPUT_SCHEMA,
        }
        hippo_contract.validate_input(payload["query"], payload["documents"])
        bright_runtime._write_json(item_root / "input.json", payload)
        roots.append(item_root)
    return tuple(roots)


def _git_head(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _paired(left: Sequence[int], right: Sequence[int]) -> dict[str, int]:
    if len(left) != len(right):
        raise FiqaTrainRuntimeError("paired score count drifted")
    deltas = [int(a) - int(b) for a, b in zip(left, right)]
    return {
        "gain": sum(value > 0 for value in deltas),
        "harm": sum(value < 0 for value in deltas),
        "net_integer_ndcg": sum(deltas),
        "tie": sum(value == 0 for value in deltas),
    }


def run_formal(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    result_path = base / RESULT_RELATIVE
    if result_path.exists():
        raise OneShotRefusal("TRAIN runtime result already exists")
    preconditions = _load_preconditions(project_root)
    integration = preconditions["integration"]
    items, labels = load_train_pack(base, integration)
    ids, contents = load_filtered_corpus(base, integration)

    root = base / RUN_ROOT_RELATIVE
    try:
        root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise OneShotRefusal("TRAIN runtime root already exists") from exc
    marker = {
        "implementation_freeze_self_sha256": preconditions["freeze"]["self_sha256"],
        "integration_result_self_sha256": INTEGRATION_RESULT_SELF_SHA256,
        "schema": ATTEMPT_SCHEMA,
    }
    marker_path = root / "attempt.marker"
    bright_runtime._write_json(marker_path, marker)

    encoder = bright_runtime._new_minilm(base)
    corpus_embeddings = bright_runtime._encode_chunks(encoder, contents)
    if corpus_embeddings.shape != (len(ids), 384):
        raise FiqaTrainRuntimeError("corpus embedding shape drifted")
    ids_pack = integration_v1.self_hashed(
        {"document_ids": list(ids), "schema": "fiqa_bridge_expansion_corpus_ids_v1"},
        "pack_sha256",
    )
    ids_path = root / "corpus.ids.json"
    embeddings_path = root / "corpus.embeddings.npy"
    bright_runtime._write_json(ids_path, ids_pack)
    bright_runtime._save_npy_exclusive(embeddings_path, corpus_embeddings)
    del encoder
    _release_cuda()

    bright_items = tuple(
        bright_runtime.ViewItem(
            ordinal=item.ordinal,
            family="FIQA",
            commitment=item.item_key,
            query=item.query,
            excluded_ids=item.excluded_ids,
        )
        for item in items
    )
    qwen_output, qwen_receipt = bright_runtime._run_qwen(base, root, bright_items)
    qwen_rows = qwen_output.get("items")
    if (
        not isinstance(qwen_rows, list)
        or len(qwen_rows) != ITEM_COUNT
        or not all(row.get("generation_valid") is True for row in qwen_rows)
    ):
        raise FiqaTrainRuntimeError("TRAIN typed query generation was not fully valid")

    flattened_queries: list[str] = []
    query_slices: list[tuple[int, int]] = []
    for item, row in zip(items, qwen_rows):
        expansions = row.get("expansions")
        if not isinstance(expansions, list) or len(expansions) != 4:
            raise FiqaTrainRuntimeError("TRAIN expansion row drifted")
        start = len(flattened_queries)
        flattened_queries.extend([item.query, *expansions])
        query_slices.append((start, len(flattened_queries)))
    encoder = bright_runtime._new_minilm(base)
    query_embeddings = bright_runtime._encode_chunks(encoder, flattened_queries)
    local_plans: list[LocalPlan] = []
    for item, row, (start, end) in zip(items, qwen_rows, query_slices):
        score_vectors = [
            quantized_scores(corpus_embeddings, query_embeddings[index])
            for index in range(start, end)
        ]
        local_plans.append(
            build_local_plan(
                item=item,
                document_ids=ids,
                document_contents=contents,
                query_score_vectors=score_vectors,
                expansions=row["expansions"],
            )
        )
    flattened_bridges = [
        query.text for plan in local_plans for query in plan.bridge_queries
    ]
    bridge_embeddings = (
        bright_runtime._encode_chunks(encoder, flattened_bridges)
        if flattened_bridges
        else np.empty((0, 384), dtype=np.float32)
    )
    expanded_plans: list[ExpandedPlan] = []
    offset = 0
    for plan in local_plans:
        count = len(plan.bridge_queries)
        vectors = [
            quantized_scores(corpus_embeddings, bridge_embeddings[index])
            for index in range(offset, offset + count)
        ]
        expanded_plans.append(expand_plan(plan, vectors))
        offset += count
    if offset != len(flattened_bridges):
        raise FiqaTrainRuntimeError("bridge embedding accounting drifted")
    query_embedding_path = root / "typed_query.embeddings.npy"
    bridge_embedding_path = root / "bridge_query.embeddings.npy"
    bright_runtime._save_npy_exclusive(query_embedding_path, query_embeddings)
    bright_runtime._save_npy_exclusive(bridge_embedding_path, bridge_embeddings)

    cross_payload = build_cross_input(expanded_plans, contents)
    cross_input_path = root / "cross_encoder.input.json"
    cross_output_path = root / "cross_encoder.output.json"
    bright_runtime._write_exclusive(
        cross_input_path,
        cross_contract.canonical_json_bytes(cross_payload),
        mode=0o600,
    )
    hippo_roots = _prepare_hipporag_inputs(
        root=root,
        plans=expanded_plans,
        contents=contents,
    )
    intents = integration_v1.self_hashed(
        {
            "cross_encoder_input_file_sha256": integration_v1.file_sha256(
                cross_input_path
            ),
            "items": [
                {
                    "base_pool": list(plan.local.base_pool),
                    "expanded_pool": list(plan.expanded.expanded_pool),
                    "item_key": plan.local.item.item_key,
                    "ordinal": plan.local.item.ordinal,
                }
                for plan in expanded_plans
            ],
            "schema": INTENT_SCHEMA,
        },
        "pack_sha256",
    )
    intent_path = root / "action.intents.json"
    bright_runtime._write_json(intent_path, intents)
    del encoder, query_embeddings, bridge_embeddings
    _release_cuda()

    semaphore = threading.Semaphore(HIPPORAG_CONCURRENCY)
    counter = bright_runtime._ConcurrencyCounter()
    cross_future: Future[Any] | None = None
    hippo_futures: dict[Future[Any], int] = {}
    completed_hippo: dict[int, Mapping[str, Any]] = {}
    environment_updates = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    previous_environment = {key: os.environ.get(key) for key in environment_updates}
    os.environ.update(environment_updates)
    try:
        with ThreadPoolExecutor(max_workers=EXTERNAL_PROCESS_CONCURRENCY) as executor:
            cross_future = executor.submit(
                cross_worker.run,
                input_path=cross_input_path,
                output_path=cross_output_path,
                model_root=base / CROSS_MODEL_RELATIVE,
            )
            for index, (plan, item_root) in enumerate(zip(expanded_plans, hippo_roots)):
                future = executor.submit(
                    bright_runtime._run_hipporag_item,
                    project_root=base,
                    item_root=item_root,
                    candidate_rows=plan.local.base_pool,
                    semaphore=semaphore,
                    counter=counter,
                )
                hippo_futures[future] = index
            for future in as_completed([cross_future, *hippo_futures]):
                if future is cross_future:
                    future.result()
                else:
                    completed_hippo[hippo_futures[future]] = future.result()
    finally:
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if (
        counter.current != 0
        or counter.peak > HIPPORAG_CONCURRENCY
        or set(completed_hippo) != set(range(ITEM_COUNT))
        or not cross_output_path.is_file()
    ):
        raise FiqaTrainRuntimeError("external action completion drifted")
    cross_output = cross_contract.parse_output(cross_output_path.read_bytes())

    action_rows: list[dict[str, Any]] = []
    for plan, cross_row in zip(expanded_plans, cross_output["items"]):
        if cross_row["document_count"] != len(plan.expanded.expanded_pool):
            raise FiqaTrainRuntimeError("expanded cross-encoder row drifted")
        p10 = bridge.rank_p10(
            expanded=plan.expanded,
            original_scores=plan.local.original_scores,
            relation_scores=plan.local.relation_scores,
            mechanism_scores=plan.local.mechanism_scores,
            cross_encoder_relation_scores=cross_row["relation_scores_quantized"],
            cross_encoder_mechanism_scores=cross_row["mechanism_scores_quantized"],
        )
        hippo = dict(completed_hippo[plan.local.item.ordinal])
        action_rows.append(
            {
                "bridge_anchor_count": len(plan.local.anchors),
                "bridge_query_count": len(plan.local.bridge_queries),
                "candidate_expansion": dict(
                    bridge.candidate_expansion_diagnostics(
                        base_pool=plan.local.base_pool,
                        expanded_pool=plan.expanded.expanded_pool,
                        p10_rows=p10.rows,
                    )
                ),
                "HippoRAG": {
                    **hippo,
                    "document_ids": [ids[row] for row in hippo["top_rows"]],
                },
                "item_key": plan.local.item.item_key,
                "ordinal": plan.local.item.ordinal,
                "P10_document_ids": [ids[row] for row in p10.rows],
                "P10_rows": list(p10.rows),
                "RAW_document_ids": [ids[row] for row in plan.local.raw_rows],
                "RAW_rows": list(plan.local.raw_rows),
            }
        )
    actions = integration_v1.self_hashed(
        {
            "active_Agent": "P10_TYPED_BRIDGE_EXPAND_CE_RRF",
            "item_count": ITEM_COUNT,
            "items": action_rows,
            "schema": ACTION_SCHEMA,
        },
        "pack_sha256",
    )
    action_path = root / "three_arm.actions.json"
    bright_runtime._write_json(action_path, actions)

    id_to_row = {identifier: index for index, identifier in enumerate(ids)}
    arm_scores: dict[str, list[int]] = {"P10": [], "RAW": [], "HippoRAG": []}
    diagnostics: list[Mapping[str, int]] = []
    for plan, action in zip(expanded_plans, action_rows):
        gold_ids = labels[plan.local.item.item_key]
        if not set(gold_ids) <= set(id_to_row):
            raise FiqaTrainRuntimeError("TRAIN gold references filtered corpus absence")
        arm_scores["P10"].append(core.integer_ndcg_at_10(action["P10_document_ids"], gold_ids))
        arm_scores["RAW"].append(core.integer_ndcg_at_10(action["RAW_document_ids"], gold_ids))
        arm_scores["HippoRAG"].append(
            core.integer_ndcg_at_10(action["HippoRAG"]["document_ids"], gold_ids)
        )
        diagnostics.append(
            bridge.candidate_expansion_diagnostics(
                base_pool=plan.local.base_pool,
                expanded_pool=plan.expanded.expanded_pool,
                p10_rows=action["P10_rows"],
                gold_rows=[id_to_row[value] for value in gold_ids],
            )
        )
    aggregates = {
        arm: {
            "mean_ndcg_at_10": sum(values) / (ITEM_COUNT * core.UTILITY_SCALE),
            "sum_integer_ndcg": sum(values),
        }
        for arm, values in arm_scores.items()
    }
    result = integration_v1.self_hashed(
        {
            "aggregates": aggregates,
            "candidate_expansion_aggregates": {
                key: sum(int(row[key]) for row in diagnostics)
                for key in diagnostics[0]
            },
            "claim_boundary": {
                "claim_eligible": False,
                "DEV_qrel_member_open_count": 0,
                "external_network_call_count": 0,
                "labels_opened_after_all_action_seal": True,
                "online_evaluator_call_count": 0,
                "TEST_qrel_member_open_count": 0,
            },
            "execution": {
                "cross_encoder_pair_count": sum(
                    2 * len(plan.expanded.expanded_pool) for plan in expanded_plans
                ),
                "external_process_concurrency_cap": EXTERNAL_PROCESS_CONCURRENCY,
                "HippoRAG_graph_edge_count_min": min(
                    row["HippoRAG"]["graph_edge_count"] for row in action_rows
                ),
                "HippoRAG_graph_node_count_min": min(
                    row["HippoRAG"]["graph_node_count"] for row in action_rows
                ),
                "HippoRAG_peak_process_concurrency": counter.peak,
                "HippoRAG_terminal_count": len(completed_hippo),
                "qwen_network_audit": qwen_receipt["network_audit"],
                "valid_generation_count": qwen_receipt["valid_generation_count"],
            },
            "formal_binding": {
                "action_file_sha256": integration_v1.file_sha256(action_path),
                "action_pack_sha256": actions["pack_sha256"],
                "attempt_marker_sha256": integration_v1.file_sha256(marker_path),
                "corpus_embedding_file_sha256": integration_v1.file_sha256(
                    embeddings_path
                ),
                "corpus_embedding_float32_bytes_sha256": float32_matrix_sha256(
                    corpus_embeddings
                ),
                "formal_implementation_commit": _git_head(project_root),
                "implementation_freeze_self_sha256": preconditions["freeze"][
                    "self_sha256"
                ],
                "integration_result_self_sha256": INTEGRATION_RESULT_SELF_SHA256,
                "intent_pack_sha256": intents["pack_sha256"],
            },
            "item_count": ITEM_COUNT,
            "paired_descriptives": {
                "P10_minus_HippoRAG": _paired(
                    arm_scores["P10"], arm_scores["HippoRAG"]
                ),
                "P10_minus_RAW": _paired(arm_scores["P10"], arm_scores["RAW"]),
            },
            "schema": SCHEMA,
            "status": "TRAIN_end_to_end_runtime_integration_complete_nonclaim_DEV_and_TEST_unopened",
        },
        "result_sha256",
    )
    bright_runtime._write_json(result_path, result, mode=0o644)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_formal(arguments.project_root)
    print(
        integration_v1.canonical_json(
            {
                "result_sha256": result["result_sha256"],
                "schema": SCHEMA,
                "status": result["status"],
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
