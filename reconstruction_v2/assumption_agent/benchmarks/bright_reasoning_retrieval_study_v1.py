"""One-shot offline BRIGHT retrieval and evaluator co-evolution study.

The staged CLI keeps future blocks and every label pack closed until the
pre-registered predecessor result exists.  Corpus tensors and label-free
actions are sealed before a stage opens its labels.  M_search submits all
RAW, Agent, and candidate-restricted official-HippoRAG actions before joining
any future, then scores only after every action is terminal.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from assumption_agent.benchmarks import bright_reasoning_retrieval_core_v1 as core
from replication_runtime.bright_minilm_v1.encoder import (
    BrightMiniLMEncoder,
    float32_matrix_sha256,
    quantized_scores,
)
from replication_runtime.bright_official_hipporag_v1.contract import (
    INPUT_SCHEMA as HIPPORAG_INPUT_SCHEMA,
    parse_output as parse_hipporag_output,
)
from replication_runtime.bright_query_generator_v1.contract import (
    INPUT_SCHEMA as QWEN_INPUT_SCHEMA,
    canonical_json_bytes as qwen_canonical_json_bytes,
    parse_output as parse_qwen_output,
)


VERSION = "bright_reasoning_retrieval_study_v1"
DESIGN_SCHEMA = f"{VERSION}_design"
FREEZE_SCHEMA = f"{VERSION}_implementation_freeze"
CORPUS_RESULT_SCHEMA = f"{VERSION}_corpus_result"
STAGE_RESULT_SCHEMA = f"{VERSION}_stage_result"
ACTION_SCHEMA = f"{VERSION}_local_action_pack"
SCORED_SCHEMA = f"{VERSION}_scored_pack"
MARKER_SCHEMA = f"{VERSION}_attempt"

DESIGN_RELATIVE = Path("manifests/bright_reasoning_retrieval_study_design_v1.json")
FREEZE_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_study_implementation_freeze_v1.json"
)
RUNTIME_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_runtime_qualification_v1.json"
)
ACQUISITION_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_acquisition_result_v1.json"
)
MINILM_MANIFEST_RELATIVE = Path("manifests/qasper_minilm_runtime_asset_v1.json")
MINILM_MODEL_RELATIVE = Path("artifacts/qasper_minilm_runtime_v1/model")
QWEN_MODEL_RELATIVE = Path(
    "artifacts/bright_reasoning_retrieval_runtime_v1/qwen2_5_1_5b_instruct"
)
HIPPORAG_LLM_RELATIVE = Path(
    "artifacts/bright_reasoning_retrieval_runtime_v1/smollm2_135m_instruct_exact"
)
HIPPORAG_PYTHON_RELATIVE = Path(
    "artifacts/bright_reasoning_retrieval_runtime_v1/hipporag_venv/bin/python"
)
SOURCE_ROOT_RELATIVE = Path("artifacts/bright_reasoning_retrieval_source_v1/dataset")
ACQUISITION_PRIVATE_RELATIVE = Path(
    "artifacts/bright_reasoning_retrieval_acquisition_v1/private"
)
FORMAL_ROOT_RELATIVE = Path("artifacts/bright_reasoning_retrieval_study_v1")

CORPUS_RESULT_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_corpus_tensor_v1.json"
)
G_RESULT_RELATIVE = Path("manifests/bright_reasoning_retrieval_G_form_v1.json")
A_FORM_RESULT_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_A_form_v1.json"
)
F_RESULT_RELATIVE = Path("manifests/bright_reasoning_retrieval_F_search_v1.json")
A_HOLD_RESULT_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_A_hold_v1.json"
)
M_RESULT_RELATIVE = Path("manifests/bright_reasoning_retrieval_M_search_v1.json")

PUBLIC_STAGE_RESULTS = {
    "G_form": G_RESULT_RELATIVE,
    "A_form": A_FORM_RESULT_RELATIVE,
    "F_search": F_RESULT_RELATIVE,
    "A_hold": A_HOLD_RESULT_RELATIVE,
    "M_search": M_RESULT_RELATIVE,
}
STAGE_PREDECESSORS = {
    "G_form": CORPUS_RESULT_RELATIVE,
    "A_form": G_RESULT_RELATIVE,
    "F_search": A_FORM_RESULT_RELATIVE,
    "A_hold": F_RESULT_RELATIVE,
    "M_search": A_HOLD_RESULT_RELATIVE,
}
BLOCK_COUNTS = {
    "G_form": 30,
    "A_form": 60,
    "F_search": 45,
    "A_hold": 45,
    "M_search": 45,
}

SOURCE_DOCUMENTS = {
    "BIOLOGY": {
        "path": Path("documents/biology-00000-of-00001.parquet"),
        "sha256": "8516d0c233f9c34e9eb6922b56e8a1698e5a6f6e504a9499fcd511cdd5741670",
        "size": 11_046_045,
        "rows": 57_359,
    },
    "ECONOMICS": {
        "path": Path("documents/economics-00000-of-00001.parquet"),
        "sha256": "f3ba8a0fbc9a9aed07b4970cc686e32cfefcd06d6922402587adf871f006394c",
        "size": 10_969_621,
        "rows": 50_220,
    },
    "ROBOTICS": {
        "path": Path("documents/robotics-00000-of-00001.parquet"),
        "sha256": "2c83f286006a3b2e11a677abe88f382009c5ee79f97c1f43f6a571f3f94e6d15",
        "size": 7_874_186,
        "rows": 61_961,
    },
}
DOCUMENT_SCHEMA = ("id", "content")
VIEW_SCHEMA = "bright_reasoning_retrieval_acquisition_v1_block_view"
LABEL_SCHEMA = "bright_reasoning_retrieval_acquisition_v1_block_labels"

DESIGN_SELF_SHA256 = "97b50a590e7bc663700eea85aa950eddcaa3246c3354d1756c7e4607934e7d42"
DESIGN_FILE_SHA256 = "d94784c5eaae343f066264771e307580677457b66f255c98222f5ffef7bd23b1"
RUNTIME_SELF_SHA256 = "80f4e846f3a1ad9ad2c1bd84d9df02aebd386074da628b06def9516422a98d18"
RUNTIME_FILE_SHA256 = "630d47f5f1d9bdab7d456ad437dec3e39d45378672ffffa3eee61b633e72708e"
ACQUISITION_RESULT_SHA256 = (
    "5736847df8a9a57f674ee02dc1fbc1fdf08120faa358631358d7d80498092ce7"
)
ACQUISITION_FILE_SHA256 = (
    "f637c369015b0e8d991a1d43373360e0292cf8362ba801347006bd297e7a8e1b"
)
MINILM_GPU_CANARY_SHA256 = (
    "bda9b93df8d6631eb224999335edf28a836d5ef5ebef5cb6dba350c111d823cf"
)
DOCUMENT_TEXT_CHARACTERS = 3_000
EMBEDDING_CHUNK_SIZE = 16_384
QWEN_BATCH_SIZE = 8
HIPPORAG_CONCURRENCY = 8
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class BrightStudyError(RuntimeError):
    """The formal BRIGHT study failed closed."""


class BrightStudyOneShotRefusal(BrightStudyError):
    """A formal stage or public result already exists."""


@dataclass(frozen=True)
class ViewItem:
    ordinal: int
    family: str
    commitment: str
    query: str
    excluded_ids: tuple[str, ...]


@dataclass(frozen=True)
class CorpusFamily:
    family: str
    ids: tuple[str, ...]
    embeddings: np.ndarray


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise BrightStudyError("value is not canonical JSON") from exc


def stable_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise BrightStudyError("self-hash field already exists")
    return {**body, field: stable_hash(body)}


def verify_self_hash(payload: Mapping[str, Any], field: str) -> str:
    declared = payload.get(field)
    if not isinstance(declared, str) or _SHA256_RE.fullmatch(declared) is None:
        raise BrightStudyError(f"{field} is absent")
    body = dict(payload)
    del body[field]
    if stable_hash(body) != declared:
        raise BrightStudyError(f"{field} drifted")
    return declared


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, field: str, *, canonical: bool = False) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BrightStudyError(f"{field} is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrightStudyError(f"{field} is invalid") from exc
    if not isinstance(value, dict):
        raise BrightStudyError(f"{field} root drifted")
    if canonical and canonical_json_bytes(value) != raw:
        raise BrightStudyError(f"{field} is not canonical")
    return value


def _write_exclusive(path: Path, raw: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any], *, mode: int = 0o600) -> None:
    _write_exclusive(path, canonical_json_bytes(value), mode=mode)


def _git_head(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise BrightStudyError("Git HEAD drifted")
    return value


def _verify_manifest(
    path: Path, *, file_hash: str, self_field: str, self_hash: str, field: str
) -> dict[str, Any]:
    if file_sha256(path) != file_hash:
        raise BrightStudyError(f"{field} file binding drifted")
    value = _load_json(path, field)
    if verify_self_hash(value, self_field) != self_hash:
        raise BrightStudyError(f"{field} self binding drifted")
    return value


def _verify_preconditions(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    design = _verify_manifest(
        project_root / DESIGN_RELATIVE,
        file_hash=DESIGN_FILE_SHA256,
        self_field="self_sha256",
        self_hash=DESIGN_SELF_SHA256,
        field="study design",
    )
    runtime = _verify_manifest(
        project_root / RUNTIME_RELATIVE,
        file_hash=RUNTIME_FILE_SHA256,
        self_field="self_sha256",
        self_hash=RUNTIME_SELF_SHA256,
        field="runtime qualification",
    )
    acquisition = _verify_manifest(
        project_root / ACQUISITION_RELATIVE,
        file_hash=ACQUISITION_FILE_SHA256,
        self_field="result_sha256",
        self_hash=ACQUISITION_RESULT_SHA256,
        field="acquisition result",
    )
    freeze = _load_json(project_root / FREEZE_RELATIVE, "implementation freeze")
    if freeze.get("schema") != FREEZE_SCHEMA:
        raise BrightStudyError("implementation freeze schema drifted")
    verify_self_hash(freeze, "self_sha256")
    if freeze.get("design_self_sha256") != DESIGN_SELF_SHA256:
        raise BrightStudyError("implementation freeze design binding drifted")
    bindings = freeze.get("implementation_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise BrightStudyError("implementation freeze bindings are invalid")
    observed = []
    for row in bindings:
        if not isinstance(row, Mapping) or set(row) != {"relative_path", "sha256"}:
            raise BrightStudyError("implementation freeze row drifted")
        relative = row.get("relative_path")
        digest = row.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
        ):
            raise BrightStudyError("implementation freeze row is invalid")
        path = project_root / relative
        if path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
            raise BrightStudyError("implementation file drifted from freeze")
        observed.append(relative)
    if len(set(observed)) != len(observed):
        raise BrightStudyError("implementation freeze paths are duplicated")
    if runtime.get("status") != "qualified_offline_runtime_and_graph_bearing_candidate_restricted_HippoRAG":
        raise BrightStudyError("runtime qualification did not pass")
    if acquisition.get("status") != "acquired_gold_separated_blocks_G_only_authorized":
        raise BrightStudyError("acquisition result did not authorize the study")
    return {
        "acquisition": acquisition,
        "design": design,
        "freeze": freeze,
        "runtime": runtime,
    }


def _start_stage(project_root: Path, stage: str, preconditions: Mapping[str, Any]) -> Path:
    if stage not in (*PUBLIC_STAGE_RESULTS, "corpus"):
        raise BrightStudyError("stage is invalid")
    public = (
        project_root / CORPUS_RESULT_RELATIVE
        if stage == "corpus"
        else project_root / PUBLIC_STAGE_RESULTS[stage]
    )
    if public.exists() or public.is_symlink():
        raise BrightStudyOneShotRefusal("public stage result already exists")
    root = project_root / FORMAL_ROOT_RELATIVE / stage
    try:
        root.mkdir(mode=0o700, parents=True)
    except FileExistsError as exc:
        raise BrightStudyOneShotRefusal("formal stage root already exists") from exc
    marker = self_hashed(
        {
            "design_self_sha256": DESIGN_SELF_SHA256,
            "implementation_freeze_self_sha256": preconditions["freeze"]["self_sha256"],
            "schema": MARKER_SCHEMA,
            "stage": stage,
        },
        "attempt_sha256",
    )
    _write_json(root / "attempt.marker", marker)
    return root


def _verify_stage_predecessor(project_root: Path, stage: str) -> dict[str, Any]:
    relative = STAGE_PREDECESSORS[stage]
    value = _load_json(project_root / relative, f"{stage} predecessor", canonical=True)
    if stage == "G_form":
        if value.get("schema") != CORPUS_RESULT_SCHEMA or value.get("status") != "corpus_tensor_complete":
            raise BrightStudyError("G_form predecessor did not complete")
        verify_self_hash(value, "result_sha256")
    else:
        if value.get("schema") != STAGE_RESULT_SCHEMA or value.get("stage") != {
            "A_form": "G_form",
            "F_search": "A_form",
            "A_hold": "F_search",
            "M_search": "A_hold",
        }[stage]:
            raise BrightStudyError("stage predecessor identity drifted")
        verify_self_hash(value, "result_sha256")
        if not str(value.get("status", "")).endswith("_complete"):
            raise BrightStudyError("stage predecessor did not complete")
    return value


def _view_and_label_bindings(
    preconditions: Mapping[str, Any], block: str
) -> Mapping[str, Any]:
    rows = preconditions["acquisition"].get("block_aggregates")
    if not isinstance(rows, Mapping) or not isinstance(rows.get(block), Mapping):
        raise BrightStudyError("acquisition block binding is unavailable")
    return rows[block]


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise BrightStudyError(f"{field} is invalid")
    return value


def _load_view(
    project_root: Path, preconditions: Mapping[str, Any], block: str
) -> tuple[ViewItem, ...]:
    binding = _view_and_label_bindings(preconditions, block)
    path = project_root / ACQUISITION_PRIVATE_RELATIVE / f"{block}.view.json"
    if file_sha256(path) != binding.get("view_pack_file_sha256"):
        raise BrightStudyError("view pack file binding drifted")
    value = _load_json(path, "view pack", canonical=True)
    if (
        value.get("schema") != VIEW_SCHEMA
        or value.get("block") != block
        or value.get("item_count") != BLOCK_COUNTS[block]
        or verify_self_hash(value, "pack_sha256") != binding.get("view_pack_sha256")
    ):
        raise BrightStudyError("view pack contract drifted")
    raw_items = value.get("items")
    if not isinstance(raw_items, list) or len(raw_items) != BLOCK_COUNTS[block]:
        raise BrightStudyError("view item count drifted")
    items: list[ViewItem] = []
    for position, raw in enumerate(raw_items):
        if not isinstance(raw, Mapping) or set(raw) != {
            "excluded_ids",
            "family",
            "item_commitment_sha256",
            "ordinal",
            "query",
        }:
            raise BrightStudyError("view item shape drifted")
        excluded = raw.get("excluded_ids")
        if isinstance(excluded, (str, bytes)) or not isinstance(excluded, list):
            raise BrightStudyError("excluded IDs drifted")
        family = raw.get("family")
        commitment = raw.get("item_commitment_sha256")
        if (
            raw.get("ordinal") != position
            or family not in core.FAMILY_ORDER
            or not isinstance(commitment, str)
            or _SHA256_RE.fullmatch(commitment) is None
        ):
            raise BrightStudyError("view item identity drifted")
        excluded_ids = tuple(_required_text(value, "excluded ID") for value in excluded)
        if len(set(excluded_ids)) != len(excluded_ids):
            raise BrightStudyError("excluded IDs are duplicated")
        items.append(
            ViewItem(
                ordinal=position,
                family=family,
                commitment=commitment,
                query=_required_text(raw.get("query"), "query"),
                excluded_ids=excluded_ids,
            )
        )
    if Counter(item.family for item in items) != Counter(
        {family: BLOCK_COUNTS[block] // 3 for family in core.FAMILY_ORDER}
    ):
        raise BrightStudyError("view family balance drifted")
    return tuple(items)


def _load_labels(
    project_root: Path,
    preconditions: Mapping[str, Any],
    block: str,
    items: Sequence[ViewItem],
) -> tuple[tuple[str, ...], ...]:
    binding = _view_and_label_bindings(preconditions, block)
    path = project_root / ACQUISITION_PRIVATE_RELATIVE / f"{block}.labels.json"
    if file_sha256(path) != binding.get("label_pack_file_sha256"):
        raise BrightStudyError("label pack file binding drifted")
    value = _load_json(path, "label pack", canonical=True)
    if (
        value.get("schema") != LABEL_SCHEMA
        or value.get("block") != block
        or value.get("item_count") != len(items)
        or verify_self_hash(value, "pack_sha256") != binding.get("label_pack_sha256")
    ):
        raise BrightStudyError("label pack contract drifted")
    raw_rows = value.get("items")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(items):
        raise BrightStudyError("label item count drifted")
    output: list[tuple[str, ...]] = []
    for position, (raw, item) in enumerate(zip(raw_rows, items)):
        if not isinstance(raw, Mapping) or set(raw) != {
            "gold_ids",
            "item_commitment_sha256",
            "ordinal",
        }:
            raise BrightStudyError("label item shape drifted")
        gold = raw.get("gold_ids")
        if (
            raw.get("ordinal") != position
            or raw.get("item_commitment_sha256") != item.commitment
            or isinstance(gold, (str, bytes))
            or not isinstance(gold, list)
        ):
            raise BrightStudyError("label item identity drifted")
        values = tuple(_required_text(value, "gold ID") for value in gold)
        if not values or len(set(values)) != len(values):
            raise BrightStudyError("gold IDs are empty or duplicated")
        output.append(values)
    return tuple(output)


def _read_source_documents(project_root: Path, family: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if family not in core.FAMILY_ORDER:
        raise BrightStudyError("document family is invalid")
    binding = SOURCE_DOCUMENTS[family]
    path = project_root / SOURCE_ROOT_RELATIVE / binding["path"]
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != binding["size"]
        or file_sha256(path) != binding["sha256"]
    ):
        raise BrightStudyError("document source binding drifted")
    try:
        import pyarrow.parquet as parquet

        reader = parquet.ParquetFile(path)
        if tuple(reader.schema_arrow.names) != DOCUMENT_SCHEMA:
            raise BrightStudyError("document parquet schema drifted")
        rows = reader.read(columns=list(DOCUMENT_SCHEMA), use_threads=False).to_pylist()
    except BrightStudyError:
        raise
    except Exception as exc:
        raise BrightStudyError("document parquet read failed") from exc
    if len(rows) != binding["rows"]:
        raise BrightStudyError("document row count drifted")
    ids: list[str] = []
    contents: list[str] = []
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != set(DOCUMENT_SCHEMA):
            raise BrightStudyError("document row shape drifted")
        identifier = _required_text(raw.get("id"), "document ID")
        content = _required_text(raw.get("content"), "document content")
        ids.append(identifier)
        contents.append(content[:DOCUMENT_TEXT_CHARACTERS])
    if len(set(ids)) != len(ids):
        raise BrightStudyError("document IDs are duplicated")
    return tuple(ids), tuple(contents)


def _save_npy_exclusive(path: Path, matrix: np.ndarray) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _new_minilm(project_root: Path) -> BrightMiniLMEncoder:
    encoder = BrightMiniLMEncoder(
        asset_manifest=project_root / MINILM_MANIFEST_RELATIVE,
        model_root=project_root / MINILM_MODEL_RELATIVE,
    )
    if encoder.canary_receipt != {
        "device": "cuda:0",
        "dtype": "float32",
        "float32_bytes_sha256": MINILM_GPU_CANARY_SHA256,
        "repeat_count": 2,
        "repeat_exact": True,
        "sentence_count": 256,
    }:
        raise BrightStudyError("BRIGHT GPU MiniLM canary drifted")
    return encoder


def _encode_chunks(
    encoder: BrightMiniLMEncoder, texts: Sequence[str]
) -> np.ndarray:
    chunks = [
        encoder.encode(texts[offset : offset + EMBEDDING_CHUNK_SIZE])
        for offset in range(0, len(texts), EMBEDDING_CHUNK_SIZE)
    ]
    matrix = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if matrix.shape != (len(texts), 384):
        raise BrightStudyError("concatenated embedding shape drifted")
    return matrix


def prepare_corpus(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    preconditions = _verify_preconditions(project_root)
    root = _start_stage(project_root, "corpus", preconditions)
    encoder = _new_minilm(project_root)
    family_rows: dict[str, Any] = {}
    for family in core.FAMILY_ORDER:
        ids, contents = _read_source_documents(project_root, family)
        matrix = _encode_chunks(encoder, contents)
        family_root = root / family
        family_root.mkdir(mode=0o700)
        id_pack = self_hashed(
            {
                "document_ids": list(ids),
                "family": family,
                "schema": f"{VERSION}_corpus_ids",
            },
            "pack_sha256",
        )
        id_path = family_root / "ids.json"
        matrix_path = family_root / "embeddings.npy"
        _write_json(id_path, id_pack)
        _save_npy_exclusive(matrix_path, matrix)
        family_rows[family] = {
            "document_count": len(ids),
            "embedding_file_sha256": file_sha256(matrix_path),
            "embedding_float32_bytes_sha256": float32_matrix_sha256(matrix),
            "embedding_shape": [len(ids), 384],
            "id_pack_file_sha256": file_sha256(id_path),
            "id_pack_sha256": id_pack["pack_sha256"],
            "source_document_file_sha256": SOURCE_DOCUMENTS[family]["sha256"],
        }
        del matrix, contents
    result = self_hashed(
        {
            "claim_boundary": {
                "block_label_or_view_open_count": 0,
                "document_content_read_count": sum(
                    SOURCE_DOCUMENTS[family]["rows"] for family in core.FAMILY_ORDER
                ),
                "external_network_call_count": 0,
                "model_call_role": "offline_MiniLM_corpus_embedding_only",
                "query_or_score_count": 0,
            },
            "family_tensors": family_rows,
            "formal_binding": {
                "attempt_marker_file_sha256": file_sha256(root / "attempt.marker"),
                "design_self_sha256": DESIGN_SELF_SHA256,
                "formal_implementation_commit": _git_head(project_root),
                "implementation_freeze_self_sha256": preconditions["freeze"]["self_sha256"],
                "minilm_gpu_canary_sha256": MINILM_GPU_CANARY_SHA256,
            },
            "schema": CORPUS_RESULT_SCHEMA,
            "status": "corpus_tensor_complete",
        },
        "result_sha256",
    )
    _write_json(project_root / CORPUS_RESULT_RELATIVE, result, mode=0o644)
    return result


def _load_corpus(project_root: Path) -> dict[str, CorpusFamily]:
    result = _load_json(
        project_root / CORPUS_RESULT_RELATIVE, "corpus result", canonical=True
    )
    if (
        result.get("schema") != CORPUS_RESULT_SCHEMA
        or result.get("status") != "corpus_tensor_complete"
    ):
        raise BrightStudyError("corpus result did not complete")
    verify_self_hash(result, "result_sha256")
    rows = result.get("family_tensors")
    if not isinstance(rows, Mapping) or set(rows) != set(core.FAMILY_ORDER):
        raise BrightStudyError("corpus family registry drifted")
    output: dict[str, CorpusFamily] = {}
    for family in core.FAMILY_ORDER:
        binding = rows[family]
        if not isinstance(binding, Mapping):
            raise BrightStudyError("corpus family binding drifted")
        root = project_root / FORMAL_ROOT_RELATIVE / "corpus" / family
        id_path = root / "ids.json"
        matrix_path = root / "embeddings.npy"
        if (
            file_sha256(id_path) != binding.get("id_pack_file_sha256")
            or file_sha256(matrix_path) != binding.get("embedding_file_sha256")
        ):
            raise BrightStudyError("corpus tensor files drifted")
        id_pack = _load_json(id_path, "corpus ID pack", canonical=True)
        if verify_self_hash(id_pack, "pack_sha256") != binding.get("id_pack_sha256"):
            raise BrightStudyError("corpus ID pack drifted")
        ids_raw = id_pack.get("document_ids")
        if not isinstance(ids_raw, list):
            raise BrightStudyError("corpus IDs drifted")
        ids = tuple(_required_text(value, "corpus document ID") for value in ids_raw)
        if len(ids) != binding.get("document_count") or len(set(ids)) != len(ids):
            raise BrightStudyError("corpus document identity drifted")
        try:
            matrix = np.load(matrix_path, allow_pickle=False)
        except Exception as exc:
            raise BrightStudyError("corpus embedding file is invalid") from exc
        matrix = np.asarray(matrix, dtype=np.float32)
        if (
            matrix.shape != (len(ids), 384)
            or binding.get("embedding_shape") != [len(ids), 384]
            or float32_matrix_sha256(matrix)
            != binding.get("embedding_float32_bytes_sha256")
            or not np.isfinite(matrix).all()
        ):
            raise BrightStudyError("corpus embedding matrix drifted")
        output[family] = CorpusFamily(family=family, ids=ids, embeddings=matrix)
    return output


def _network_trace_receipt(root: Path, prefix: str) -> dict[str, Any]:
    paths = sorted(root.glob(prefix + "*"), key=lambda path: path.name)
    if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
        raise BrightStudyError("network trace set is unavailable")
    rows = []
    loopback_bind_count = 0
    for path in paths:
        try:
            text = path.read_text(encoding="ascii")
        except (OSError, UnicodeDecodeError) as exc:
            raise BrightStudyError("network trace is invalid") from exc
        if any(
            token in text
            for token in ("connect(", "sendto(", "sendmsg(", "sendmmsg(")
        ):
            raise BrightStudyError("outbound network syscall was observed")
        for line in text.splitlines():
            if line.startswith("bind("):
                if 'inet_pton(AF_INET6, "::1"' not in line:
                    raise BrightStudyError("non-loopback network bind was observed")
                loopback_bind_count += 1
        rows.append(
            {
                "path": path.name,
                "sha256": file_sha256(path),
                "size": path.stat().st_size,
            }
        )
    return {
        "external_connect_syscall_count": 0,
        "external_send_syscall_count": 0,
        "loopback_bind_count": loopback_bind_count,
        "trace_file_count": len(rows),
        "trace_set_sha256": stable_hash(rows),
    }


def _run_qwen(
    project_root: Path, stage_root: Path, items: Sequence[ViewItem]
) -> tuple[dict[str, Any], dict[str, Any]]:
    input_payload = {
        "items": [
            {"ordinal": item.ordinal, "query": item.query} for item in items
        ],
        "schema": QWEN_INPUT_SCHEMA,
    }
    input_path = stage_root / "qwen.input.json"
    output_path = stage_root / "qwen.output.json"
    _write_exclusive(input_path, qwen_canonical_json_bytes(input_payload), mode=0o600)
    for name in ("home", "hf", "tmp"):
        (stage_root / name).mkdir(mode=0o700)
    trace_prefix = "qwen.network.trace"
    command = [
        "/usr/bin/strace",
        "-ff",
        "-e",
        "trace=network",
        "-o",
        str(stage_root / trace_prefix),
        str(project_root / HIPPORAG_PYTHON_RELATIVE),
        "-I",
        "-B",
        "-m",
        "replication_runtime.bright_query_generator_v1.worker",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--model",
        str(project_root / QWEN_MODEL_RELATIVE),
        "--batch-size",
        str(QWEN_BATCH_SIZE),
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HOME": str(stage_root / "hf"),
            "HF_HUB_OFFLINE": "1",
            "HOME": str(stage_root / "home"),
            "MPLCONFIGDIR": str(stage_root / "tmp" / "mpl"),
            "TOKENIZERS_PARALLELISM": "false",
            "TMPDIR": str(stage_root / "tmp"),
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            check=False,
            capture_output=True,
            timeout=1800,
        )
    except subprocess.TimeoutExpired as exc:
        raise BrightStudyError("Qwen worker timed out") from exc
    if completed.returncode != 0:
        raise BrightStudyError(
            "Qwen worker failed: "
            + hashlib.sha256(completed.stderr).hexdigest()
        )
    if not output_path.is_file() or output_path.is_symlink():
        raise BrightStudyError("Qwen worker output is unavailable")
    output = parse_qwen_output(output_path.read_bytes())
    if len(output["items"]) != len(items):
        raise BrightStudyError("Qwen output item count drifted")
    network = _network_trace_receipt(stage_root, trace_prefix)
    receipt = {
        "input_file_sha256": file_sha256(input_path),
        "network_audit": network,
        "output_file_sha256": file_sha256(output_path),
        "valid_generation_count": sum(
            row["generation_valid"] for row in output["items"]
        ),
        "worker_stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "worker_stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
    }
    return output, receipt


def _local_actions(
    *,
    project_root: Path,
    stage_root: Path,
    items: Sequence[ViewItem],
    qwen_output: Mapping[str, Any],
    corpus: Mapping[str, CorpusFamily],
) -> tuple[dict[str, Any], np.ndarray]:
    encoder = _new_minilm(project_root)
    qwen_rows = qwen_output.get("items")
    if not isinstance(qwen_rows, list) or len(qwen_rows) != len(items):
        raise BrightStudyError("Qwen rows drifted before local retrieval")
    flattened: list[str] = []
    slices: list[tuple[int, int]] = []
    for item, generated in zip(items, qwen_rows):
        if not isinstance(generated, Mapping) or generated.get("ordinal") != item.ordinal:
            raise BrightStudyError("Qwen row identity drifted")
        expansions = generated.get("expansions")
        if not isinstance(expansions, list):
            raise BrightStudyError("Qwen expansions drifted")
        queries = [item.query, *expansions]
        start = len(flattened)
        flattened.extend(queries)
        slices.append((start, len(flattened)))
    query_matrix = _encode_chunks(encoder, flattened)
    original_matrix = np.stack(
        [query_matrix[start] for start, _end in slices], axis=0
    ).astype(np.float32, copy=False)
    id_to_row = {
        family: {identifier: index for index, identifier in enumerate(corpus[family].ids)}
        for family in core.FAMILY_ORDER
    }
    action_rows: list[dict[str, Any]] = []
    for item, generated, (start, end) in zip(items, qwen_rows, slices):
        family_corpus = corpus[item.family]
        score_vectors = [
            quantized_scores(family_corpus.embeddings, query_matrix[index])
            for index in range(start, end)
        ]
        excluded_rows = tuple(
            id_to_row[item.family][identifier]
            for identifier in item.excluded_ids
            if identifier in id_to_row[item.family]
        )
        local = core.build_local_retrieval(
            score_vectors, excluded_rows=excluded_rows
        )
        action_rows.append(
            {
                "candidate_document_ids": [
                    family_corpus.ids[row] for row in local.candidate_rows
                ],
                "candidate_rows": list(local.candidate_rows),
                "family": item.family,
                "generation_valid": generated["generation_valid"],
                "item_commitment_sha256": item.commitment,
                "ordinal": item.ordinal,
                "raw_document_ids": [
                    family_corpus.ids[row] for row in local.raw_rows
                ],
                "raw_rows": list(local.raw_rows),
                "recipe_document_ids": {
                    recipe: [family_corpus.ids[row] for row in local.recipe_rows[recipe]]
                    for recipe in core.RECIPE_ORDER
                },
                "recipe_rows": {
                    recipe: list(local.recipe_rows[recipe])
                    for recipe in core.RECIPE_ORDER
                },
            }
        )
    action_pack = self_hashed(
        {
            "item_count": len(items),
            "items": action_rows,
            "recipe_order": list(core.RECIPE_ORDER),
            "schema": ACTION_SCHEMA,
        },
        "pack_sha256",
    )
    action_path = stage_root / "local.action.json"
    embedding_path = stage_root / "original_query_embeddings.npy"
    _write_json(action_path, action_pack)
    _save_npy_exclusive(embedding_path, original_matrix)
    return action_pack, original_matrix


def _validate_action_pack(
    pack: Mapping[str, Any], items: Sequence[ViewItem]
) -> list[Mapping[str, Any]]:
    if (
        pack.get("schema") != ACTION_SCHEMA
        or pack.get("item_count") != len(items)
        or pack.get("recipe_order") != list(core.RECIPE_ORDER)
    ):
        raise BrightStudyError("local action pack envelope drifted")
    verify_self_hash(pack, "pack_sha256")
    rows = pack.get("items")
    if not isinstance(rows, list) or len(rows) != len(items):
        raise BrightStudyError("local action rows drifted")
    for position, (row, item) in enumerate(zip(rows, items)):
        if (
            not isinstance(row, Mapping)
            or row.get("ordinal") != position
            or row.get("family") != item.family
            or row.get("item_commitment_sha256") != item.commitment
            or set(row.get("recipe_document_ids", {})) != set(core.RECIPE_ORDER)
            or set(row.get("recipe_rows", {})) != set(core.RECIPE_ORDER)
        ):
            raise BrightStudyError("local action row identity drifted")
        candidate = row.get("candidate_document_ids")
        raw = row.get("raw_document_ids")
        if (
            not isinstance(candidate, list)
            or len(candidate) != core.POOL_SIZE
            or len(set(candidate)) != core.POOL_SIZE
            or not isinstance(raw, list)
            or len(raw) != core.TOP_K
            or len(set(raw)) != core.TOP_K
            or not set(raw) <= set(candidate)
        ):
            raise BrightStudyError("local action document sets drifted")
        for recipe in core.RECIPE_ORDER:
            values = row["recipe_document_ids"].get(recipe)
            if (
                not isinstance(values, list)
                or len(values) != core.TOP_K
                or len(set(values)) != core.TOP_K
                or not set(values) <= set(candidate)
            ):
                raise BrightStudyError("recipe action drifted")
    return rows


def _score_local_actions(
    *,
    items: Sequence[ViewItem],
    action_pack: Mapping[str, Any],
    labels: Sequence[Sequence[str]],
) -> tuple[list[dict[str, int]], list[int]]:
    rows = _validate_action_pack(action_pack, items)
    utilities: list[dict[str, int]] = []
    raw_utilities: list[int] = []
    for row, gold in zip(rows, labels):
        raw_utilities.append(core.integer_ndcg_at_10(row["raw_document_ids"], gold))
        utilities.append(
            {
                recipe: core.integer_ndcg_at_10(
                    row["recipe_document_ids"][recipe], gold
                )
                for recipe in core.RECIPE_ORDER
            }
        )
    return utilities, raw_utilities


def _write_scored_pack(
    *,
    stage_root: Path,
    items: Sequence[ViewItem],
    utility_rows: Sequence[Mapping[str, int]],
    raw_utilities: Sequence[int],
    embedding_path: Path,
) -> dict[str, Any]:
    rows = [
        {
            "family": item.family,
            "item_commitment_sha256": item.commitment,
            "ordinal": item.ordinal,
            "raw_utility": raw,
            "recipe_utilities": dict(utility),
        }
        for item, utility, raw in zip(items, utility_rows, raw_utilities)
    ]
    payload = self_hashed(
        {
            "item_count": len(items),
            "items": rows,
            "original_query_embeddings_file_sha256": file_sha256(embedding_path),
            "recipe_order": list(core.RECIPE_ORDER),
            "schema": SCORED_SCHEMA,
        },
        "pack_sha256",
    )
    _write_json(stage_root / "scored.json", payload)
    return payload


def _aggregate_recipe_utilities(
    items: Sequence[ViewItem], utility_rows: Sequence[Mapping[str, int]]
) -> dict[str, Any]:
    return {
        recipe: {
            "family_sum_integer_ndcg": {
                family: sum(
                    utility[recipe]
                    for item, utility in zip(items, utility_rows)
                    if item.family == family
                )
                for family in core.FAMILY_ORDER
            },
            "sum_integer_ndcg": sum(utility[recipe] for utility in utility_rows),
        }
        for recipe in core.RECIPE_ORDER
    }


def run_g_form(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    preconditions = _verify_preconditions(project_root)
    predecessor = _verify_stage_predecessor(project_root, "G_form")
    root = _start_stage(project_root, "G_form", preconditions)
    items = _load_view(project_root, preconditions, "G_form")
    qwen_output, qwen_receipt = _run_qwen(project_root, root, items)
    qwen_pack = self_hashed(
        {
            "item_commitments": [item.commitment for item in items],
            "qwen_output": qwen_output,
            "schema": f"{VERSION}_G_form_operator_pack",
        },
        "pack_sha256",
    )
    pack_path = root / "operator.action.json"
    _write_json(pack_path, qwen_pack)
    family_valid = {
        family: sum(
            row["generation_valid"]
            for item, row in zip(items, qwen_output["items"])
            if item.family == family
        )
        for family in core.FAMILY_ORDER
    }
    result = self_hashed(
        {
            "claim_boundary": {
                "document_read_count": 0,
                "label_open_count": 0,
                "model_item_count": len(items),
                "network_external_connect_or_send_count": 0,
                "prompt_or_parser_adaptation_count": 0,
                "retrieval_or_score_count": 0,
            },
            "family_valid_generation_counts": family_valid,
            "formal_binding": {
                "attempt_marker_file_sha256": file_sha256(root / "attempt.marker"),
                "formal_implementation_commit": _git_head(project_root),
                "predecessor_result_sha256": predecessor["result_sha256"],
            },
            "item_count": len(items),
            "private_bindings": {
                "operator_pack_file_sha256": file_sha256(pack_path),
                "operator_pack_sha256": qwen_pack["pack_sha256"],
                "qwen": qwen_receipt,
            },
            "schema": STAGE_RESULT_SCHEMA,
            "stage": "G_form",
            "status": "G_form_complete",
            "valid_generation_count": sum(family_valid.values()),
        },
        "result_sha256",
    )
    _write_json(project_root / G_RESULT_RELATIVE, result, mode=0o644)
    return result


def _run_scored_local_stage(
    *,
    project_root: Path,
    block: str,
    preconditions: Mapping[str, Any],
    root: Path,
) -> tuple[
    tuple[ViewItem, ...],
    dict[str, Any],
    np.ndarray,
    list[dict[str, int]],
    list[int],
    dict[str, Any],
    dict[str, Any],
]:
    items = _load_view(project_root, preconditions, block)
    qwen_output, qwen_receipt = _run_qwen(project_root, root, items)
    corpus = _load_corpus(project_root)
    action_pack, embeddings = _local_actions(
        project_root=project_root,
        stage_root=root,
        items=items,
        qwen_output=qwen_output,
        corpus=corpus,
    )
    action_path = root / "local.action.json"
    embedding_path = root / "original_query_embeddings.npy"
    if not action_path.is_file() or not embedding_path.is_file():
        raise BrightStudyError("local action seal was not persisted")
    labels = _load_labels(project_root, preconditions, block, items)
    utility_rows, raw_utilities = _score_local_actions(
        items=items, action_pack=action_pack, labels=labels
    )
    scored_pack = _write_scored_pack(
        stage_root=root,
        items=items,
        utility_rows=utility_rows,
        raw_utilities=raw_utilities,
        embedding_path=embedding_path,
    )
    return (
        items,
        action_pack,
        embeddings,
        utility_rows,
        raw_utilities,
        scored_pack,
        qwen_receipt,
    )


def _local_stage_private_bindings(
    root: Path,
    *,
    action_pack: Mapping[str, Any],
    scored_pack: Mapping[str, Any],
    qwen_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "action_pack_file_sha256": file_sha256(root / "local.action.json"),
        "action_pack_sha256": action_pack["pack_sha256"],
        "original_query_embeddings_file_sha256": file_sha256(
            root / "original_query_embeddings.npy"
        ),
        "qwen": dict(qwen_receipt),
        "scored_pack_file_sha256": file_sha256(root / "scored.json"),
        "scored_pack_sha256": scored_pack["pack_sha256"],
    }


def run_a_form(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    preconditions = _verify_preconditions(project_root)
    predecessor = _verify_stage_predecessor(project_root, "A_form")
    root = _start_stage(project_root, "A_form", preconditions)
    (
        items,
        action_pack,
        _embeddings,
        utility_rows,
        raw_utilities,
        scored_pack,
        qwen_receipt,
    ) = _run_scored_local_stage(
        project_root=project_root,
        block="A_form",
        preconditions=preconditions,
        root=root,
    )
    result = self_hashed(
        {
            "claim_boundary": {
                "future_block_open_count": 0,
                "HippoRAG_action_count": 0,
                "label_open_count": 1,
                "online_model_or_evaluator_count": 0,
                "recipe_or_prompt_change_count": 0,
            },
            "evaluator_candidate_count": len(core.evaluator_specs()),
            "formal_binding": {
                "attempt_marker_file_sha256": file_sha256(root / "attempt.marker"),
                "formal_implementation_commit": _git_head(project_root),
                "predecessor_result_sha256": predecessor["result_sha256"],
            },
            "item_count": len(items),
            "private_bindings": _local_stage_private_bindings(
                root,
                action_pack=action_pack,
                scored_pack=scored_pack,
                qwen_receipt=qwen_receipt,
            ),
            "raw_sum_integer_ndcg": sum(raw_utilities),
            "recipe_aggregates": _aggregate_recipe_utilities(items, utility_rows),
            "schema": STAGE_RESULT_SCHEMA,
            "stage": "A_form",
            "status": "A_form_complete",
            "valid_generation_count": sum(
                row["generation_valid"] for row in action_pack["items"]
            ),
        },
        "result_sha256",
    )
    _write_json(project_root / A_FORM_RESULT_RELATIVE, result, mode=0o644)
    return result


def _load_scored_stage(
    project_root: Path, block: str
) -> tuple[list[str], list[dict[str, int]], list[int], np.ndarray]:
    if block not in ("A_form", "F_search", "A_hold"):
        raise BrightStudyError("scored stage name drifted")
    result = _load_json(
        project_root / PUBLIC_STAGE_RESULTS[block], f"{block} result", canonical=True
    )
    if result.get("schema") != STAGE_RESULT_SCHEMA or result.get("stage") != block:
        raise BrightStudyError("scored stage result drifted")
    verify_self_hash(result, "result_sha256")
    bindings = result.get("private_bindings")
    if not isinstance(bindings, Mapping):
        raise BrightStudyError("scored stage private bindings drifted")
    root = project_root / FORMAL_ROOT_RELATIVE / block
    scored_path = root / "scored.json"
    embedding_path = root / "original_query_embeddings.npy"
    if (
        file_sha256(scored_path) != bindings.get("scored_pack_file_sha256")
        or file_sha256(embedding_path)
        != bindings.get("original_query_embeddings_file_sha256")
    ):
        raise BrightStudyError("scored stage private files drifted")
    scored = _load_json(scored_path, "scored pack", canonical=True)
    if (
        scored.get("schema") != SCORED_SCHEMA
        or verify_self_hash(scored, "pack_sha256")
        != bindings.get("scored_pack_sha256")
        or scored.get("recipe_order") != list(core.RECIPE_ORDER)
    ):
        raise BrightStudyError("scored pack binding drifted")
    raw_rows = scored.get("items")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise BrightStudyError("scored pack rows drifted")
    families: list[str] = []
    utilities: list[dict[str, int]] = []
    raw_utilities: list[int] = []
    for position, row in enumerate(raw_rows):
        if (
            not isinstance(row, Mapping)
            or row.get("ordinal") != position
            or row.get("family") not in core.FAMILY_ORDER
            or set(row.get("recipe_utilities", {})) != set(core.RECIPE_ORDER)
        ):
            raise BrightStudyError("scored pack row identity drifted")
        recipe_values = {
            recipe: row["recipe_utilities"][recipe] for recipe in core.RECIPE_ORDER
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= core.UTILITY_SCALE
            for value in (*recipe_values.values(), row.get("raw_utility"))
        ):
            raise BrightStudyError("scored utility drifted")
        families.append(row["family"])
        utilities.append(recipe_values)
        raw_utilities.append(row["raw_utility"])
    try:
        embeddings = np.asarray(np.load(embedding_path, allow_pickle=False), dtype=np.float32)
    except Exception as exc:
        raise BrightStudyError("scored query embeddings are invalid") from exc
    if embeddings.shape != (len(families), 384):
        raise BrightStudyError("scored query embedding shape drifted")
    return families, utilities, raw_utilities, embeddings


def run_f_search(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    preconditions = _verify_preconditions(project_root)
    predecessor = _verify_stage_predecessor(project_root, "F_search")
    root = _start_stage(project_root, "F_search", preconditions)
    (
        items,
        action_pack,
        _embeddings,
        utility_rows,
        raw_utilities,
        scored_pack,
        qwen_receipt,
    ) = _run_scored_local_stage(
        project_root=project_root,
        block="F_search",
        preconditions=preconditions,
        root=root,
    )
    portfolio = core.select_f_portfolio(utility_rows)
    a_families, a_utilities, _a_raw, a_embeddings = _load_scored_stage(
        project_root, "A_form"
    )
    challenger, crossfit_total = core.select_evaluator_challenger(
        families=a_families,
        query_embeddings=a_embeddings,
        utility_rows=a_utilities,
        portfolio=portfolio,
    )
    result = self_hashed(
        {
            "claim_boundary": {
                "A_hold_or_M_search_open_count": 0,
                "HippoRAG_action_count": 0,
                "label_open_count": 1,
                "online_model_or_evaluator_count": 0,
                "portfolio_or_evaluator_retry_count": 0,
            },
            "evaluator_challenger": {
                "A_form_leave_one_out_selected_utility_sum": crossfit_total,
                "alpha": challenger.alpha,
                "evaluator_id": challenger.evaluator_id,
                "k": challenger.k,
                "scope": challenger.scope,
            },
            "formal_binding": {
                "A_form_result_sha256": predecessor["result_sha256"],
                "attempt_marker_file_sha256": file_sha256(root / "attempt.marker"),
                "formal_implementation_commit": _git_head(project_root),
            },
            "item_count": len(items),
            "P_base": portfolio[0],
            "private_bindings": _local_stage_private_bindings(
                root,
                action_pack=action_pack,
                scored_pack=scored_pack,
                qwen_receipt=qwen_receipt,
            ),
            "recipe_aggregates": _aggregate_recipe_utilities(items, utility_rows),
            "recipe_portfolio": list(portfolio),
            "raw_sum_integer_ndcg": sum(raw_utilities),
            "schema": STAGE_RESULT_SCHEMA,
            "stage": "F_search",
            "status": "F_search_complete",
            "valid_generation_count": sum(
                row["generation_valid"] for row in action_pack["items"]
            ),
        },
        "result_sha256",
    )
    _write_json(project_root / F_RESULT_RELATIVE, result, mode=0o644)
    return result


def _evaluator_spec_from_result(result: Mapping[str, Any]) -> core.EvaluatorSpec:
    raw = result.get("evaluator_challenger")
    if not isinstance(raw, Mapping):
        raise BrightStudyError("evaluator challenger binding is absent")
    spec = core.EvaluatorSpec(
        scope=raw.get("scope"),  # type: ignore[arg-type]
        k=raw.get("k"),  # type: ignore[arg-type]
        alpha=raw.get("alpha"),  # type: ignore[arg-type]
    )
    if spec not in core.evaluator_specs() or spec.evaluator_id != raw.get("evaluator_id"):
        raise BrightStudyError("evaluator challenger binding drifted")
    return spec


def _paired_counts(left: Sequence[int], right: Sequence[int]) -> dict[str, int]:
    if len(left) != len(right):
        raise BrightStudyError("paired vectors differ in length")
    return {
        "gain": sum(a > b for a, b in zip(left, right)),
        "harm": sum(a < b for a, b in zip(left, right)),
        "tie": sum(a == b for a, b in zip(left, right)),
    }


def run_a_hold(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    preconditions = _verify_preconditions(project_root)
    predecessor = _verify_stage_predecessor(project_root, "A_hold")
    root = _start_stage(project_root, "A_hold", preconditions)
    (
        items,
        action_pack,
        hold_embeddings,
        utility_rows,
        raw_utilities,
        scored_pack,
        qwen_receipt,
    ) = _run_scored_local_stage(
        project_root=project_root,
        block="A_hold",
        preconditions=preconditions,
        root=root,
    )
    f_result = predecessor
    portfolio_raw = f_result.get("recipe_portfolio")
    p_base = f_result.get("P_base")
    if (
        not isinstance(portfolio_raw, list)
        or len(portfolio_raw) != 4
        or len(set(portfolio_raw)) != 4
        or p_base != portfolio_raw[0]
        or any(recipe not in core.RECIPE_ORDER for recipe in portfolio_raw)
    ):
        raise BrightStudyError("frozen F portfolio drifted")
    portfolio = tuple(portfolio_raw)
    spec = _evaluator_spec_from_result(f_result)
    a_families, a_utilities, _a_raw, a_embeddings = _load_scored_stage(
        project_root, "A_form"
    )
    challenger_recipes = [
        core.route_with_evaluator(
            target_family=item.family,
            target_embedding=embedding,
            training_families=a_families,
            training_embeddings=a_embeddings,
            training_utility_rows=a_utilities,
            portfolio=portfolio,
            spec=spec,
        )
        for item, embedding in zip(items, hold_embeddings)
    ]
    e0 = [row[p_base] for row in utility_rows]
    e1 = [row[recipe] for row, recipe in zip(utility_rows, challenger_recipes)]
    promoted = sum(e1) > sum(e0)
    routing_pack = self_hashed(
        {
            "challenger_recipes": challenger_recipes,
            "evaluator_id": spec.evaluator_id,
            "item_commitments": [item.commitment for item in items],
            "schema": f"{VERSION}_A_hold_routing",
        },
        "pack_sha256",
    )
    routing_path = root / "routing.json"
    _write_json(routing_path, routing_pack)
    result = self_hashed(
        {
            "active_evaluator": spec.evaluator_id if promoted else "E0_GLOBAL_P_BASE",
            "challenger_evaluator": spec.evaluator_id,
            "claim_boundary": {
                "M_search_open_count": 0,
                "HippoRAG_action_count": 0,
                "label_open_count": 1,
                "online_model_or_evaluator_count": 0,
                "promotion_retry_or_threshold_change_count": 0,
            },
            "E0_sum_integer_ndcg": sum(e0),
            "E1_minus_E0_sum_integer_ndcg": sum(e1) - sum(e0),
            "E1_sum_integer_ndcg": sum(e1),
            "formal_binding": {
                "attempt_marker_file_sha256": file_sha256(root / "attempt.marker"),
                "F_search_result_sha256": f_result["result_sha256"],
                "formal_implementation_commit": _git_head(project_root),
            },
            "item_count": len(items),
            "paired_E1_minus_E0": _paired_counts(e1, e0),
            "private_bindings": {
                **_local_stage_private_bindings(
                    root,
                    action_pack=action_pack,
                    scored_pack=scored_pack,
                    qwen_receipt=qwen_receipt,
                ),
                "routing_pack_file_sha256": file_sha256(routing_path),
                "routing_pack_sha256": routing_pack["pack_sha256"],
            },
            "promoted": promoted,
            "raw_sum_integer_ndcg": sum(raw_utilities),
            "schema": STAGE_RESULT_SCHEMA,
            "stage": "A_hold",
            "status": "A_hold_complete",
            "valid_generation_count": sum(
                row["generation_valid"] for row in action_pack["items"]
            ),
        },
        "result_sha256",
    )
    _write_json(project_root / A_HOLD_RESULT_RELATIVE, result, mode=0o644)
    return result


class _ConcurrencyCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def enter(self) -> None:
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def leave(self) -> None:
        with self._lock:
            self.current -= 1
            if self.current < 0:
                raise BrightStudyError("HippoRAG concurrency counter underflowed")


def _prepare_hipporag_inputs(
    *,
    project_root: Path,
    root: Path,
    items: Sequence[ViewItem],
    action_pack: Mapping[str, Any],
    corpus: Mapping[str, CorpusFamily],
) -> tuple[Path, ...]:
    action_rows = _validate_action_pack(action_pack, items)
    contents_by_family: dict[str, tuple[str, ...]] = {}
    for family in core.FAMILY_ORDER:
        ids, contents = _read_source_documents(project_root, family)
        if ids != corpus[family].ids:
            raise BrightStudyError("M_search source document order drifted")
        contents_by_family[family] = contents
    hippo_root = root / "hipporag"
    hippo_root.mkdir(mode=0o700)
    output: list[Path] = []
    for item, action in zip(items, action_rows):
        item_root = hippo_root / f"item_{item.ordinal:03d}"
        item_root.mkdir(mode=0o700)
        for name in ("home", "hf", "tmp"):
            (item_root / name).mkdir(mode=0o700)
        candidate_rows = action.get("candidate_rows")
        if not isinstance(candidate_rows, list) or len(candidate_rows) != core.POOL_SIZE:
            raise BrightStudyError("HippoRAG candidate rows drifted")
        payload = {
            "documents": [
                {
                    "content": contents_by_family[item.family][int(row)],
                    "ordinal": ordinal,
                }
                for ordinal, row in enumerate(candidate_rows)
            ],
            "query": item.query,
            "schema": HIPPORAG_INPUT_SCHEMA,
        }
        _write_json(item_root / "input.json", payload)
        output.append(item_root)
    return tuple(output)


def _run_hipporag_item(
    *,
    project_root: Path,
    item_root: Path,
    candidate_rows: Sequence[int],
    semaphore: threading.Semaphore,
    counter: _ConcurrencyCounter,
) -> dict[str, Any]:
    with semaphore:
        counter.enter()
        try:
            command = [
                "/usr/bin/bwrap",
                "--die-with-parent",
                "--unshare-all",
                "--new-session",
                "--ro-bind",
                "/",
                "/",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--tmpfs",
                "/tmp",
                "--dir",
                "/tmp/models",
                "--ro-bind",
                str(project_root / HIPPORAG_LLM_RELATIVE),
                "/tmp/models/llm",
                "--ro-bind",
                str(project_root / MINILM_MODEL_RELATIVE),
                "/tmp/models/embed",
                "--bind",
                str(item_root),
                str(item_root),
                "--chdir",
                str(project_root),
                "--setenv",
                "CUDA_VISIBLE_DEVICES",
                "",
                "--setenv",
                "HF_HOME",
                str(item_root / "hf"),
                "--setenv",
                "HF_HUB_OFFLINE",
                "1",
                "--setenv",
                "HOME",
                str(item_root / "home"),
                "--setenv",
                "MPLCONFIGDIR",
                str(item_root / "tmp" / "mpl"),
                "--setenv",
                "OMP_NUM_THREADS",
                "2",
                "--setenv",
                "TOKENIZERS_PARALLELISM",
                "false",
                "--setenv",
                "TMPDIR",
                str(item_root / "tmp"),
                "--setenv",
                "TRANSFORMERS_OFFLINE",
                "1",
                str(project_root / HIPPORAG_PYTHON_RELATIVE),
                "-I",
                "-B",
                "-m",
                "replication_runtime.bright_official_hipporag_v1.worker",
                "--input",
                str(item_root / "input.json"),
                "--output",
                str(item_root / "output.json"),
                "--index-root",
                str(item_root / "index"),
                "--llm-model",
                "/tmp/models/llm",
                "--embedding-model",
                "/tmp/models/embed",
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=project_root,
                    check=False,
                    capture_output=True,
                    timeout=1800,
                )
            except subprocess.TimeoutExpired as exc:
                raise BrightStudyError("HippoRAG item timed out") from exc
            _write_exclusive(item_root / "stdout.log", completed.stdout, mode=0o600)
            _write_exclusive(item_root / "stderr.log", completed.stderr, mode=0o600)
            if completed.returncode != 0:
                raise BrightStudyError(
                    "HippoRAG item failed: "
                    + hashlib.sha256(completed.stderr).hexdigest()
                )
            output_path = item_root / "output.json"
            payload = parse_hipporag_output(output_path.read_bytes())
            if (
                payload["graph_node_count"] <= core.POOL_SIZE
                or payload["graph_edge_count"] <= 0
            ):
                raise BrightStudyError("HippoRAG item did not build a graph")
            top_rows = tuple(candidate_rows[index] for index in payload["top_ordinals"])
            if len(top_rows) != core.TOP_K or len(set(top_rows)) != core.TOP_K:
                raise BrightStudyError("HippoRAG mapped top-k drifted")
            return {
                "graph_edge_count": payload["graph_edge_count"],
                "graph_node_count": payload["graph_node_count"],
                "output_file_sha256": file_sha256(output_path),
                "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
                "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
                "top_rows": list(top_rows),
            }
        finally:
            counter.leave()


def _family_arm_aggregates(
    items: Sequence[ViewItem], arm_scores: Mapping[str, Sequence[int]]
) -> dict[str, Any]:
    return {
        arm: {
            "family_sum_integer_ndcg": {
                family: sum(
                    value
                    for item, value in zip(items, values)
                    if item.family == family
                )
                for family in core.FAMILY_ORDER
            },
            "mean_ndcg_at_10": sum(values)
            / (len(values) * core.UTILITY_SCALE),
            "sum_integer_ndcg": sum(values),
        }
        for arm, values in arm_scores.items()
    }


def run_m_search(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    preconditions = _verify_preconditions(project_root)
    predecessor = _verify_stage_predecessor(project_root, "M_search")
    root = _start_stage(project_root, "M_search", preconditions)
    items = _load_view(project_root, preconditions, "M_search")
    qwen_output, qwen_receipt = _run_qwen(project_root, root, items)
    corpus = _load_corpus(project_root)
    action_pack, m_embeddings = _local_actions(
        project_root=project_root,
        stage_root=root,
        items=items,
        qwen_output=qwen_output,
        corpus=corpus,
    )
    action_rows = _validate_action_pack(action_pack, items)
    item_roots = _prepare_hipporag_inputs(
        project_root=project_root,
        root=root,
        items=items,
        action_pack=action_pack,
        corpus=corpus,
    )
    f_result = _load_json(
        project_root / F_RESULT_RELATIVE, "F_search result", canonical=True
    )
    if f_result.get("stage") != "F_search":
        raise BrightStudyError("F_search result identity drifted before M_search")
    verify_self_hash(f_result, "result_sha256")
    a_hold = predecessor
    if a_hold.get("stage") != "A_hold" or not isinstance(a_hold.get("promoted"), bool):
        raise BrightStudyError("A_hold result identity drifted before M_search")
    portfolio_raw = f_result.get("recipe_portfolio")
    p_base = f_result.get("P_base")
    if (
        not isinstance(portfolio_raw, list)
        or len(portfolio_raw) != 4
        or p_base != portfolio_raw[0]
    ):
        raise BrightStudyError("M_search portfolio binding drifted")
    portfolio = tuple(portfolio_raw)
    spec = _evaluator_spec_from_result(f_result)
    a_families, a_utilities, _a_raw, a_embeddings = _load_scored_stage(
        project_root, "A_form"
    )

    semaphore = threading.Semaphore(HIPPORAG_CONCURRENCY)
    counter = _ConcurrencyCounter()
    futures: dict[Future[Any], tuple[str, int]] = {}
    results: dict[str, list[Any]] = {
        "Agent": [None] * len(items),
        "HippoRAG": [None] * len(items),
        "RAW": [None] * len(items),
    }

    def raw_action(index: int) -> dict[str, Any]:
        return {
            "document_ids": list(action_rows[index]["raw_document_ids"]),
            "rows": list(action_rows[index]["raw_rows"]),
        }

    def agent_action(index: int) -> dict[str, Any]:
        challenger_recipe = core.route_with_evaluator(
            target_family=items[index].family,
            target_embedding=m_embeddings[index],
            training_families=a_families,
            training_embeddings=a_embeddings,
            training_utility_rows=a_utilities,
            portfolio=portfolio,
            spec=spec,
        )
        active_recipe = challenger_recipe if a_hold["promoted"] else p_base
        return {
            "active_document_ids": list(
                action_rows[index]["recipe_document_ids"][active_recipe]
            ),
            "active_recipe": active_recipe,
            "active_rows": list(action_rows[index]["recipe_rows"][active_recipe]),
            "challenger_document_ids": list(
                action_rows[index]["recipe_document_ids"][challenger_recipe]
            ),
            "challenger_recipe": challenger_recipe,
            "P_base_document_ids": list(
                action_rows[index]["recipe_document_ids"][p_base]
            ),
        }

    def hippo_action(index: int) -> dict[str, Any]:
        payload = _run_hipporag_item(
            project_root=project_root,
            item_root=item_roots[index],
            candidate_rows=action_rows[index]["candidate_rows"],
            semaphore=semaphore,
            counter=counter,
        )
        payload["document_ids"] = [
            corpus[items[index].family].ids[row] for row in payload["top_rows"]
        ]
        return payload

    with ThreadPoolExecutor(max_workers=64) as executor:
        for index in range(len(items)):
            futures[executor.submit(raw_action, index)] = ("RAW", index)
            futures[executor.submit(agent_action, index)] = ("Agent", index)
            futures[executor.submit(hippo_action, index)] = ("HippoRAG", index)
        submitted_count = len(futures)
        if submitted_count != len(items) * 3:
            raise BrightStudyError("M_search logical action submission drifted")
        for future in as_completed(futures):
            arm, index = futures[future]
            results[arm][index] = future.result()
    if counter.current != 0 or counter.peak > HIPPORAG_CONCURRENCY:
        raise BrightStudyError("HippoRAG concurrency cap drifted")
    if any(value is None for rows in results.values() for value in rows):
        raise BrightStudyError("M_search action result is missing")

    m_action_pack = self_hashed(
        {
            "active_evaluator": a_hold["active_evaluator"],
            "item_commitments": [item.commitment for item in items],
            "item_count": len(items),
            "results": results,
            "schema": f"{VERSION}_M_search_action_pack",
            "submission_count_before_first_join": submitted_count,
        },
        "pack_sha256",
    )
    m_action_path = root / "three_arm.action.json"
    _write_json(m_action_path, m_action_pack)
    labels = _load_labels(project_root, preconditions, "M_search", items)

    arm_scores: dict[str, list[int]] = {
        "Agent": [],
        "HippoRAG": [],
        "RAW": [],
        "P_base": [],
        "E1_counterfactual": [],
    }
    for index, gold in enumerate(labels):
        arm_scores["Agent"].append(
            core.integer_ndcg_at_10(results["Agent"][index]["active_document_ids"], gold)
        )
        arm_scores["HippoRAG"].append(
            core.integer_ndcg_at_10(results["HippoRAG"][index]["document_ids"], gold)
        )
        arm_scores["RAW"].append(
            core.integer_ndcg_at_10(results["RAW"][index]["document_ids"], gold)
        )
        arm_scores["P_base"].append(
            core.integer_ndcg_at_10(results["Agent"][index]["P_base_document_ids"], gold)
        )
        arm_scores["E1_counterfactual"].append(
            core.integer_ndcg_at_10(
                results["Agent"][index]["challenger_document_ids"], gold
            )
        )
    aggregates = _family_arm_aggregates(items, arm_scores)
    family_agent_minus_hippo = {
        family: aggregates["Agent"]["family_sum_integer_ndcg"][family]
        - aggregates["HippoRAG"]["family_sum_integer_ndcg"][family]
        for family in core.FAMILY_ORDER
    }
    q_minus_p = sum(arm_scores["Agent"]) - sum(arm_scores["P_base"])
    l5_supported = bool(a_hold["promoted"] and q_minus_p > 0)
    result = self_hashed(
        {
            "active_evaluator": a_hold["active_evaluator"],
            "active_Q_minus_retained_P_sum_integer_ndcg": q_minus_p,
            "arm_aggregates": aggregates,
            "claim_boundary": {
                "answer_generation_count": 0,
                "candidate_restricted_HippoRAG": True,
                "external_network_call_count": 0,
                "label_open_count_after_all_actions": 1,
                "online_evaluator_count": 0,
                "replay_retry_or_resample_count": 0,
            },
            "descriptive_evidence": {
                "Agent_minus_HippoRAG_family_sum_integer_ndcg": family_agent_minus_hippo,
                "Agent_minus_HippoRAG_positive_in_all_three_families": all(
                    value > 0 for value in family_agent_minus_hippo.values()
                ),
                "evaluator_promoted_on_A_hold": a_hold["promoted"],
                "L4_retained_Q_minus_P_positive": q_minus_p > 0,
                "L5_promoted_evaluator_improved_untouched_search": l5_supported,
            },
            "formal_binding": {
                "A_hold_result_sha256": a_hold["result_sha256"],
                "attempt_marker_file_sha256": file_sha256(root / "attempt.marker"),
                "F_search_result_sha256": f_result["result_sha256"],
                "formal_implementation_commit": _git_head(project_root),
            },
            "HippoRAG_execution": {
                "graph_edge_count_min": min(
                    row["graph_edge_count"] for row in results["HippoRAG"]
                ),
                "graph_node_count_min": min(
                    row["graph_node_count"] for row in results["HippoRAG"]
                ),
                "observed_peak_process_concurrency": counter.peak,
                "process_concurrency_cap": HIPPORAG_CONCURRENCY,
                "terminal_action_count": len(results["HippoRAG"]),
            },
            "item_count": len(items),
            "paired": {
                "Agent_minus_HippoRAG": _paired_counts(
                    arm_scores["Agent"], arm_scores["HippoRAG"]
                ),
                "Agent_minus_RAW": _paired_counts(
                    arm_scores["Agent"], arm_scores["RAW"]
                ),
                "active_Q_minus_retained_P": _paired_counts(
                    arm_scores["Agent"], arm_scores["P_base"]
                ),
            },
            "private_bindings": {
                "local_action_pack_file_sha256": file_sha256(
                    root / "local.action.json"
                ),
                "local_action_pack_sha256": action_pack["pack_sha256"],
                "original_query_embeddings_file_sha256": file_sha256(
                    root / "original_query_embeddings.npy"
                ),
                "qwen": qwen_receipt,
                "three_arm_action_pack_file_sha256": file_sha256(m_action_path),
                "three_arm_action_pack_sha256": m_action_pack["pack_sha256"],
            },
            "schema": STAGE_RESULT_SCHEMA,
            "stage": "M_search",
            "status": "M_search_complete",
            "submission_count_before_first_join": submitted_count,
            "valid_generation_count": sum(
                row["generation_valid"] for row in action_pack["items"]
            ),
        },
        "result_sha256",
    )
    _write_json(project_root / M_RESULT_RELATIVE, result, mode=0o644)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare-corpus", "G-form", "A-form", "F-search", "A-hold", "M-search"),
    )
    parser.add_argument("--project-root", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    functions = {
        "prepare-corpus": prepare_corpus,
        "G-form": run_g_form,
        "A-form": run_a_form,
        "F-search": run_f_search,
        "A-hold": run_a_hold,
        "M-search": run_m_search,
    }
    result = functions[arguments.command](arguments.project_root)
    print(
        json.dumps(
            {
                "result_sha256": result["result_sha256"],
                "status": result["status"],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
