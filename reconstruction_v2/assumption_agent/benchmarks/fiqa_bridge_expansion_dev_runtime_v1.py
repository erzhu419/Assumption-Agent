"""One-shot offline FiQA DEV runtime for frozen P10, RAW, and HippoRAG.

The runtime reuses the hash-bound corpus tensor, forms all 48 label-free
actions with the frozen P10 implementation, seals the joined three-arm action
pack, and only then opens the separately acquired DEV label pack.  It never
opens TEST qrels and performs no online evaluation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import gc
import json
import os
from pathlib import Path
import subprocess
import threading
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from reconstruction_v2.assumption_agent.benchmarks import (
    fiqa_bridge_expansion_dev_acquisition_v1 as acquisition,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    fiqa_bridge_expansion_train_runtime_v1 as train_v1,
)


SCHEMA = "fiqa_bridge_expansion_dev_runtime_result_v1"
ATTEMPT_SCHEMA = "fiqa_bridge_expansion_dev_runtime_attempt_v1"
INTENT_SCHEMA = "fiqa_bridge_expansion_dev_runtime_intents_v1"
ACTION_SCHEMA = "fiqa_bridge_expansion_dev_runtime_actions_v1"
FREEZE_SCHEMA = "fiqa_bridge_expansion_dev_runtime_implementation_freeze_v1"
ITEM_COUNT = 48
HIPPORAG_CONCURRENCY = 12
EXTERNAL_PROCESS_CONCURRENCY = 13

FREEZE_RELATIVE = Path(
    "manifests/fiqa_bridge_expansion_dev_runtime_implementation_freeze_v1.json"
)
RESULT_RELATIVE = Path("manifests/fiqa_bridge_expansion_dev_runtime_result_v1.json")
IMPLEMENTATION_RELATIVE = Path(
    "assumption_agent/benchmarks/fiqa_bridge_expansion_dev_runtime_v1.py"
)
TEST_RELATIVE = Path("tests/test_fiqa_bridge_expansion_dev_runtime_v1.py")
RUN_ROOT_RELATIVE = Path("artifacts/fiqa_bridge_expansion_dev_runtime_v1")

ACQUISITION_RESULT_RELATIVE = acquisition.RESULT_RELATIVE
ACQUISITION_RESULT_FILE_SHA256 = (
    "624b694f17652bb8b24f932ae82c977f94f7ebf5578d24ec19ab2a700b33a572"
)
ACQUISITION_RESULT_SELF_SHA256 = (
    "a67b090e8062677182148b51095dc23a12f193ae66624e5611f905119b9f7e14"
)
ACQUISITION_ROOT_RELATIVE = acquisition.RUN_ROOT_RELATIVE

CORPUS_IDS_RELATIVE = Path(
    "artifacts/fiqa_bridge_expansion_train_runtime_v1/corpus.ids.json"
)
CORPUS_IDS_FILE_SHA256 = (
    "a3b9b5dc2fc4a89f94e79ed69a374e99acadc992d7092cebb2ac24cfc1b816e6"
)
CORPUS_IDS_PACK_SHA256 = (
    "3dfaba21b3c3708c3be79091267fcb9dc21d5393e9274b87af0993061e6fe907"
)
CORPUS_EMBEDDINGS_RELATIVE = Path(
    "artifacts/fiqa_bridge_expansion_train_runtime_v1/corpus.embeddings.npy"
)
CORPUS_EMBEDDINGS_FILE_SHA256 = (
    "806b4ac10a4579afbc7b881f3b59e003a17ccca2d21929903364671b478963e3"
)
CORPUS_EMBEDDINGS_FLOAT32_SHA256 = (
    "021031a45276f853288628540a896d7ac5a7e6909ea7245b6f42a679fb8617b0"
)


class FiqaDevRuntimeError(RuntimeError):
    """The frozen prospective DEV runtime failed closed."""


class OneShotRefusal(FiqaDevRuntimeError):
    """The formal DEV runtime root or result is already consumed."""


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FiqaDevRuntimeError(f"{name} is invalid") from exc
    if not isinstance(value, Mapping):
        raise FiqaDevRuntimeError(f"{name} root drifted")
    return value


def _verify_self(
    value: Mapping[str, Any],
    field: str,
    expected: str,
) -> None:
    try:
        acquisition.verify_self_hash(value, field, expected)
    except acquisition.FiqaDevAcquisitionError as exc:
        raise FiqaDevRuntimeError(str(exc)) from exc


def _load_acquisition(base: Path) -> Mapping[str, Any]:
    path = base / ACQUISITION_RESULT_RELATIVE
    if (
        not path.is_file()
        or path.is_symlink()
        or train_v1.integration_v1.file_sha256(path)
        != ACQUISITION_RESULT_FILE_SHA256
    ):
        raise FiqaDevRuntimeError("DEV acquisition result file drifted")
    value = _read_json(path, "DEV acquisition result")
    _verify_self(value, "acquisition_sha256", ACQUISITION_RESULT_SELF_SHA256)
    boundary = value.get("claim_boundary")
    if (
        value.get("schema") != acquisition.SCHEMA
        or value.get("status")
        != "one_shot_FiQA_DEV_C_confirm_acquired_labels_separated_no_action_TEST_unopened"
        or not isinstance(boundary, Mapping)
        or boundary.get("TEST_qrel_member_open_count") != 0
    ):
        raise FiqaDevRuntimeError("DEV acquisition completion drifted")
    return value


def _verify_freeze(base: Path) -> Mapping[str, Any]:
    path = base / FREEZE_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise FiqaDevRuntimeError("DEV runtime freeze is unavailable")
    value = _read_json(path, "DEV runtime freeze")
    if value.get("schema") != FREEZE_SCHEMA:
        raise FiqaDevRuntimeError("DEV runtime freeze schema drifted")
    declared = value.get("self_sha256")
    if not isinstance(declared, str):
        raise FiqaDevRuntimeError("DEV runtime freeze identity is absent")
    _verify_self(value, "self_sha256", declared)
    rows = value.get("implementation_bindings")
    if not isinstance(rows, list):
        raise FiqaDevRuntimeError("DEV runtime freeze bindings drifted")
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in rows
        if isinstance(row, Mapping)
    }
    expected_paths = {IMPLEMENTATION_RELATIVE.as_posix(), TEST_RELATIVE.as_posix()}
    if set(observed) != expected_paths:
        raise FiqaDevRuntimeError("DEV runtime freeze file set drifted")
    for relative, digest in observed.items():
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or train_v1.integration_v1.file_sha256(base / relative) != digest
        ):
            raise FiqaDevRuntimeError("DEV runtime implementation drifted")
    if (
        value.get("DEV_acquisition_result_self_sha256")
        != ACQUISITION_RESULT_SELF_SHA256
        or value.get("P10_core_implementation_freeze_self_sha256")
        != "47e102f3da12a3021929a48c525cb9c4a6b69f5d6cb4f3cc260e4a15ddac6f8b"
    ):
        raise FiqaDevRuntimeError("DEV runtime freeze prerequisite drifted")
    return value


def load_dev_views(
    base: Path,
    acquisition_result: Mapping[str, Any],
) -> tuple[train_v1.ViewItem, ...]:
    binding = acquisition_result.get("C_confirm_pack")
    if not isinstance(binding, Mapping) or binding.get("item_count") != ITEM_COUNT:
        raise FiqaDevRuntimeError("DEV view binding drifted")
    path = base / ACQUISITION_ROOT_RELATIVE / "C_confirm.view.jsonl"
    if (
        path.stat().st_size != binding.get("view_file_size_bytes")
        or train_v1.integration_v1.file_sha256(path)
        != binding.get("view_file_sha256")
    ):
        raise FiqaDevRuntimeError("DEV view file drifted")
    try:
        rows = train_v1._parse_canonical_jsonl(path, "DEV view pack")
    except train_v1.FiqaTrainRuntimeError as exc:
        raise FiqaDevRuntimeError(str(exc)) from exc
    if len(rows) != ITEM_COUNT:
        raise FiqaDevRuntimeError("DEV view row count drifted")
    items: list[train_v1.ViewItem] = []
    for ordinal, row in enumerate(rows):
        if set(row) != {
            "excluded_document_ids",
            "family",
            "item_key",
            "query",
            "source_query_id",
        } or row.get("family") != "FIQA":
            raise FiqaDevRuntimeError("DEV view row shape drifted")
        excluded_raw = row.get("excluded_document_ids")
        if not isinstance(excluded_raw, list):
            raise FiqaDevRuntimeError("DEV view exclusions drifted")
        items.append(
            train_v1.ViewItem(
                ordinal=ordinal,
                item_key=train_v1.integration_v1._required_text(
                    row.get("item_key"), "item key"
                ),
                query=train_v1.integration_v1._required_text(
                    row.get("query"), "query"
                ),
                excluded_ids=tuple(
                    train_v1.integration_v1._required_text(
                        value, "excluded document ID"
                    )
                    for value in excluded_raw
                ),
            )
        )
    if len({item.item_key for item in items}) != ITEM_COUNT:
        raise FiqaDevRuntimeError("DEV view item keys are duplicated")
    return tuple(items)


def load_dev_labels_after_seal(
    *,
    base: Path,
    acquisition_result: Mapping[str, Any],
    items: Sequence[train_v1.ViewItem],
    action_path: Path,
    expected_action_file_sha256: str,
) -> Mapping[str, tuple[str, ...]]:
    if (
        not action_path.is_file()
        or action_path.is_symlink()
        or train_v1.integration_v1.file_sha256(action_path)
        != expected_action_file_sha256
    ):
        raise FiqaDevRuntimeError("action seal is absent or drifted before DEV labels")
    binding = acquisition_result.get("C_confirm_pack")
    if not isinstance(binding, Mapping):
        raise FiqaDevRuntimeError("DEV label binding drifted")
    path = base / ACQUISITION_ROOT_RELATIVE / "C_confirm.labels.jsonl"
    if (
        path.stat().st_size != binding.get("label_file_size_bytes")
        or train_v1.integration_v1.file_sha256(path)
        != binding.get("label_file_sha256")
    ):
        raise FiqaDevRuntimeError("DEV label file drifted")
    try:
        rows = train_v1._parse_canonical_jsonl(path, "DEV label pack")
    except train_v1.FiqaTrainRuntimeError as exc:
        raise FiqaDevRuntimeError(str(exc)) from exc
    if len(rows) != len(items):
        raise FiqaDevRuntimeError("DEV label row count drifted")
    labels: dict[str, tuple[str, ...]] = {}
    for item, row in zip(items, rows):
        if (
            set(row) != {"family", "gold_document_ids", "item_key"}
            or row.get("family") != "FIQA"
            or row.get("item_key") != item.item_key
        ):
            raise FiqaDevRuntimeError("DEV label identity drifted")
        gold_raw = row.get("gold_document_ids")
        if not isinstance(gold_raw, list) or not gold_raw:
            raise FiqaDevRuntimeError("DEV gold list drifted")
        gold = tuple(
            train_v1.integration_v1._required_text(value, "gold document ID")
            for value in gold_raw
        )
        if len(set(gold)) != len(gold):
            raise FiqaDevRuntimeError("DEV gold IDs are duplicated")
        labels[item.item_key] = gold
    return labels


def load_reused_corpus(
    base: Path,
    integration: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    try:
        ids, contents = train_v1.load_filtered_corpus(base, integration)
    except train_v1.FiqaTrainRuntimeError as exc:
        raise FiqaDevRuntimeError(str(exc)) from exc
    ids_path = base / CORPUS_IDS_RELATIVE
    embeddings_path = base / CORPUS_EMBEDDINGS_RELATIVE
    if (
        not ids_path.is_file()
        or ids_path.is_symlink()
        or train_v1.integration_v1.file_sha256(ids_path) != CORPUS_IDS_FILE_SHA256
        or not embeddings_path.is_file()
        or embeddings_path.is_symlink()
        or train_v1.integration_v1.file_sha256(embeddings_path)
        != CORPUS_EMBEDDINGS_FILE_SHA256
    ):
        raise FiqaDevRuntimeError("reused corpus tensor file drifted")
    ids_pack = _read_json(ids_path, "reused corpus ID pack")
    try:
        train_v1.integration_v1.verify_self_hash(
            ids_pack, "pack_sha256", CORPUS_IDS_PACK_SHA256
        )
    except train_v1.integration_v1.FiqaTrainIntegrationError as exc:
        raise FiqaDevRuntimeError(str(exc)) from exc
    if tuple(ids_pack.get("document_ids", ())) != ids:
        raise FiqaDevRuntimeError("reused corpus ID order drifted")
    try:
        embeddings = np.asarray(
            np.load(embeddings_path, allow_pickle=False),
            dtype=np.float32,
        )
    except Exception as exc:
        raise FiqaDevRuntimeError("reused corpus embedding file is invalid") from exc
    if (
        embeddings.shape != (len(ids), 384)
        or not np.isfinite(embeddings).all()
        or train_v1.float32_matrix_sha256(embeddings)
        != CORPUS_EMBEDDINGS_FLOAT32_SHA256
    ):
        raise FiqaDevRuntimeError("reused corpus embedding tensor drifted")
    return ids, contents, embeddings


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def primary_decision(
    arm_scores: Mapping[str, Sequence[int]],
) -> tuple[bool, Mapping[str, Mapping[str, int]]]:
    if set(arm_scores) != {"P10", "RAW", "HippoRAG"}:
        raise FiqaDevRuntimeError("DEV arm registry drifted")
    if any(len(values) != ITEM_COUNT for values in arm_scores.values()):
        raise FiqaDevRuntimeError("DEV score vector length drifted")
    try:
        paired = {
            "P10_minus_HippoRAG": train_v1._paired(
                arm_scores["P10"], arm_scores["HippoRAG"]
            ),
            "P10_minus_RAW": train_v1._paired(
                arm_scores["P10"], arm_scores["RAW"]
            ),
        }
    except train_v1.FiqaTrainRuntimeError as exc:
        raise FiqaDevRuntimeError(str(exc)) from exc
    passed = all(row["net_integer_ndcg"] > 0 for row in paired.values())
    return passed, paired


def _git_head(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_formal(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    result_path = base / RESULT_RELATIVE
    if result_path.exists():
        raise OneShotRefusal("DEV runtime result already exists")
    try:
        preconditions = train_v1._load_preconditions(project_root)
    except train_v1.FiqaTrainRuntimeError as exc:
        raise FiqaDevRuntimeError(str(exc)) from exc
    acquisition_result = _load_acquisition(base)
    freeze = _verify_freeze(base)
    integration = preconditions["integration"]
    items = load_dev_views(base, acquisition_result)
    ids, contents, corpus_embeddings = load_reused_corpus(base, integration)

    root = base / RUN_ROOT_RELATIVE
    try:
        root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise OneShotRefusal("DEV runtime root already exists") from exc
    marker = {
        "DEV_acquisition_result_self_sha256": ACQUISITION_RESULT_SELF_SHA256,
        "implementation_freeze_self_sha256": freeze["self_sha256"],
        "schema": ATTEMPT_SCHEMA,
    }
    marker_path = root / "attempt.marker"
    train_v1.bright_runtime._write_json(marker_path, marker)

    bright_items = tuple(
        train_v1.bright_runtime.ViewItem(
            ordinal=item.ordinal,
            family="FIQA",
            commitment=item.item_key,
            query=item.query,
            excluded_ids=item.excluded_ids,
        )
        for item in items
    )
    try:
        qwen_output, qwen_receipt = train_v1.bright_runtime._run_qwen(
            base, root, bright_items
        )
    except train_v1.bright_runtime.BrightStudyError as exc:
        raise FiqaDevRuntimeError(str(exc)) from exc
    qwen_rows = qwen_output.get("items")
    if (
        not isinstance(qwen_rows, list)
        or len(qwen_rows) != ITEM_COUNT
        or not all(row.get("generation_valid") is True for row in qwen_rows)
    ):
        raise FiqaDevRuntimeError("DEV typed query generation was not fully valid")

    flattened_queries: list[str] = []
    query_slices: list[tuple[int, int]] = []
    for item, row in zip(items, qwen_rows):
        expansions = row.get("expansions")
        if not isinstance(expansions, list) or len(expansions) != 4:
            raise FiqaDevRuntimeError("DEV expansion row drifted")
        start = len(flattened_queries)
        flattened_queries.extend([item.query, *expansions])
        query_slices.append((start, len(flattened_queries)))
    try:
        encoder = train_v1.bright_runtime._new_minilm(base)
        query_embeddings = train_v1.bright_runtime._encode_chunks(
            encoder, flattened_queries
        )
    except train_v1.bright_runtime.BrightStudyError as exc:
        raise FiqaDevRuntimeError(str(exc)) from exc
    local_plans: list[train_v1.LocalPlan] = []
    for item, row, (start, end) in zip(items, qwen_rows, query_slices):
        score_vectors = [
            train_v1.quantized_scores(corpus_embeddings, query_embeddings[index])
            for index in range(start, end)
        ]
        try:
            local_plans.append(
                train_v1.build_local_plan(
                    item=item,
                    document_ids=ids,
                    document_contents=contents,
                    query_score_vectors=score_vectors,
                    expansions=row["expansions"],
                )
            )
        except train_v1.FiqaTrainRuntimeError as exc:
            raise FiqaDevRuntimeError(str(exc)) from exc
    flattened_bridges = [
        query.text for plan in local_plans for query in plan.bridge_queries
    ]
    try:
        bridge_embeddings = (
            train_v1.bright_runtime._encode_chunks(encoder, flattened_bridges)
            if flattened_bridges
            else np.empty((0, 384), dtype=np.float32)
        )
    except train_v1.bright_runtime.BrightStudyError as exc:
        raise FiqaDevRuntimeError(str(exc)) from exc
    expanded_plans: list[train_v1.ExpandedPlan] = []
    offset = 0
    for plan in local_plans:
        count = len(plan.bridge_queries)
        vectors = [
            train_v1.quantized_scores(corpus_embeddings, bridge_embeddings[index])
            for index in range(offset, offset + count)
        ]
        try:
            expanded_plans.append(train_v1.expand_plan(plan, vectors))
        except train_v1.FiqaTrainRuntimeError as exc:
            raise FiqaDevRuntimeError(str(exc)) from exc
        offset += count
    if offset != len(flattened_bridges):
        raise FiqaDevRuntimeError("DEV bridge embedding accounting drifted")
    query_embedding_path = root / "typed_query.embeddings.npy"
    bridge_embedding_path = root / "bridge_query.embeddings.npy"
    train_v1.bright_runtime._save_npy_exclusive(
        query_embedding_path, query_embeddings
    )
    train_v1.bright_runtime._save_npy_exclusive(
        bridge_embedding_path, bridge_embeddings
    )

    try:
        cross_payload = train_v1.build_cross_input(expanded_plans, contents)
    except train_v1.FiqaTrainRuntimeError as exc:
        raise FiqaDevRuntimeError(str(exc)) from exc
    cross_input_path = root / "cross_encoder.input.json"
    cross_output_path = root / "cross_encoder.output.json"
    train_v1.bright_runtime._write_exclusive(
        cross_input_path,
        train_v1.cross_contract.canonical_json_bytes(cross_payload),
        mode=0o600,
    )
    try:
        hippo_roots = train_v1._prepare_hipporag_inputs(
            root=root,
            plans=expanded_plans,
            contents=contents,
        )
    except train_v1.FiqaTrainRuntimeError as exc:
        raise FiqaDevRuntimeError(str(exc)) from exc
    if len(hippo_roots) != ITEM_COUNT:
        raise FiqaDevRuntimeError("DEV HippoRAG input count drifted")
    intents = train_v1.integration_v1.self_hashed(
        {
            "cross_encoder_input_file_sha256": train_v1.integration_v1.file_sha256(
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
    train_v1.bright_runtime._write_json(intent_path, intents)
    del encoder, query_embeddings, bridge_embeddings
    _release_cuda()

    semaphore = threading.Semaphore(HIPPORAG_CONCURRENCY)
    counter = train_v1.bright_runtime._ConcurrencyCounter()
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
                train_v1.cross_worker.run,
                input_path=cross_input_path,
                output_path=cross_output_path,
                model_root=base / train_v1.CROSS_MODEL_RELATIVE,
            )
            for index, (plan, item_root) in enumerate(
                zip(expanded_plans, hippo_roots)
            ):
                future = executor.submit(
                    train_v1.bright_runtime._run_hipporag_item,
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
    except Exception as exc:
        if isinstance(exc, FiqaDevRuntimeError):
            raise
        raise FiqaDevRuntimeError("DEV external action execution failed") from exc
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
        raise FiqaDevRuntimeError("DEV external action completion drifted")
    try:
        cross_output = train_v1.cross_contract.parse_output(
            cross_output_path.read_bytes()
        )
    except Exception as exc:
        raise FiqaDevRuntimeError("DEV cross-encoder output drifted") from exc
    if len(cross_output.get("items", ())) != ITEM_COUNT:
        raise FiqaDevRuntimeError("DEV cross-encoder item count drifted")

    action_rows: list[dict[str, Any]] = []
    for plan, cross_row in zip(expanded_plans, cross_output["items"]):
        if cross_row["document_count"] != len(plan.expanded.expanded_pool):
            raise FiqaDevRuntimeError("DEV cross-encoder row drifted")
        p10 = train_v1.bridge.rank_p10(
            expanded=plan.expanded,
            original_scores=plan.local.original_scores,
            relation_scores=plan.local.relation_scores,
            mechanism_scores=plan.local.mechanism_scores,
            cross_encoder_relation_scores=cross_row[
                "relation_scores_quantized"
            ],
            cross_encoder_mechanism_scores=cross_row[
                "mechanism_scores_quantized"
            ],
        )
        hippo = dict(completed_hippo[plan.local.item.ordinal])
        action_rows.append(
            {
                "bridge_anchor_count": len(plan.local.anchors),
                "bridge_query_count": len(plan.local.bridge_queries),
                "candidate_expansion": dict(
                    train_v1.bridge.candidate_expansion_diagnostics(
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
    if len(action_rows) != ITEM_COUNT:
        raise FiqaDevRuntimeError("DEV joined action count drifted")
    actions = train_v1.integration_v1.self_hashed(
        {
            "active_Agent": "P10_TYPED_BRIDGE_EXPAND_CE_RRF",
            "item_count": ITEM_COUNT,
            "items": action_rows,
            "schema": ACTION_SCHEMA,
            "intent_pack_sha256": intents["pack_sha256"],
        },
        "pack_sha256",
    )
    action_path = root / "three_arm.actions.json"
    train_v1.bright_runtime._write_json(action_path, actions)
    action_file_sha256 = train_v1.integration_v1.file_sha256(action_path)

    labels = load_dev_labels_after_seal(
        base=base,
        acquisition_result=acquisition_result,
        items=items,
        action_path=action_path,
        expected_action_file_sha256=action_file_sha256,
    )
    id_to_row = {identifier: index for index, identifier in enumerate(ids)}
    arm_scores: dict[str, list[int]] = {"P10": [], "RAW": [], "HippoRAG": []}
    diagnostics: list[Mapping[str, int]] = []
    for plan, action in zip(expanded_plans, action_rows):
        gold_ids = labels[plan.local.item.item_key]
        if not set(gold_ids) <= set(id_to_row):
            raise FiqaDevRuntimeError("DEV gold references filtered corpus absence")
        arm_scores["P10"].append(
            train_v1.core.integer_ndcg_at_10(action["P10_document_ids"], gold_ids)
        )
        arm_scores["RAW"].append(
            train_v1.core.integer_ndcg_at_10(action["RAW_document_ids"], gold_ids)
        )
        arm_scores["HippoRAG"].append(
            train_v1.core.integer_ndcg_at_10(
                action["HippoRAG"]["document_ids"], gold_ids
            )
        )
        diagnostics.append(
            train_v1.bridge.candidate_expansion_diagnostics(
                base_pool=plan.local.base_pool,
                expanded_pool=plan.expanded.expanded_pool,
                p10_rows=action["P10_rows"],
                gold_rows=[id_to_row[value] for value in gold_ids],
            )
        )
    aggregates = {
        arm: {
            "mean_ndcg_at_10": sum(values)
            / (ITEM_COUNT * train_v1.core.UTILITY_SCALE),
            "sum_integer_ndcg": sum(values),
        }
        for arm, values in arm_scores.items()
    }
    primary_passed, paired = primary_decision(arm_scores)
    status = (
        "FiQA_DEV_C_confirm_primary_passed_P10_frozen_TEST_unopened"
        if primary_passed
        else "FiQA_DEV_C_confirm_primary_failed_P10_development_stops_TEST_unopened"
    )
    result = train_v1.integration_v1.self_hashed(
        {
            "aggregates": aggregates,
            "candidate_expansion_aggregates": {
                key: sum(int(row[key]) for row in diagnostics)
                for key in diagnostics[0]
            },
            "claim_boundary": {
                "DEV_label_pack_open_count": 1,
                "external_network_call_count": 0,
                "labels_opened_after_all_action_seal": True,
                "online_evaluator_call_count": 0,
                "population_inference": False,
                "prospective_development_claim_eligible": True,
                "TEST_qrel_member_open_count": 0,
            },
            "execution": {
                "cross_encoder_pair_count": sum(
                    2 * len(plan.expanded.expanded_pool)
                    for plan in expanded_plans
                ),
                "external_process_concurrency_cap": EXTERNAL_PROCESS_CONCURRENCY,
                "HippoRAG_peak_process_concurrency": counter.peak,
                "HippoRAG_terminal_count": len(completed_hippo),
                "MiniLM_corpus_embedding_call_count": 0,
                "MiniLM_new_query_text_count": len(flattened_queries)
                + len(flattened_bridges),
                "qwen_network_audit": qwen_receipt["network_audit"],
                "valid_generation_count": qwen_receipt["valid_generation_count"],
            },
            "formal_binding": {
                "action_file_sha256": action_file_sha256,
                "action_pack_sha256": actions["pack_sha256"],
                "attempt_marker_sha256": train_v1.integration_v1.file_sha256(
                    marker_path
                ),
                "corpus_embeddings_file_sha256": CORPUS_EMBEDDINGS_FILE_SHA256,
                "DEV_acquisition_result_self_sha256": ACQUISITION_RESULT_SELF_SHA256,
                "formal_implementation_commit": _git_head(project_root),
                "implementation_freeze_self_sha256": freeze["self_sha256"],
                "intent_pack_sha256": intents["pack_sha256"],
                "P10_core_implementation_freeze_self_sha256": "47e102f3da12a3021929a48c525cb9c4a6b69f5d6cb4f3cc260e4a15ddac6f8b",
            },
            "item_count": ITEM_COUNT,
            "paired_descriptives": paired,
            "primary_rule": "P10_minus_RAW_and_P10_minus_candidate_restricted_HippoRAG_are_each_strictly_positive_in_aggregate",
            "primary_passed": primary_passed,
            "schema": SCHEMA,
            "status": status,
        },
        "result_sha256",
    )
    train_v1.bright_runtime._write_json(result_path, result, mode=0o644)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_formal(arguments.project_root)
    print(
        train_v1.integration_v1.canonical_json(
            {
                "primary_passed": result["primary_passed"],
                "result_sha256": result["result_sha256"],
                "schema": SCHEMA,
                "status": result["status"],
            }
        ).decode("ascii")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
