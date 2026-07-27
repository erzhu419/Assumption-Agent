"""Consumed-data architecture diagnostic for marginal typed replacements.

This module is deliberately not a fresh efficacy study.  It reuses the
already-consumed, mutually disjoint HybridQA P6/E2 A_form, A_hold and
M_search packs to answer one architecture question: can an evaluator learn
to score concrete, item-local replacements that introduce typed evidence
outside RAW top five?

The diagnostic has one fixed terminal.  It enumerates every query-anchored
candidate reachable within two typed edges, permits replacement of each
still-present original RAW slot, and places a zero-score no-op in the same
action space.  A no-intercept weighted ridge model predicts exact marginal
utility.  Three leave-one-block-out folds provide descriptive cross-fitting;
the result cannot be promoted to fresh evidence or reused as a formal test.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from importlib import import_module, metadata
import json
import math
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np

from assumption_agent.benchmarks import hybridqa_direct_acquisition_v2 as acquisition
from assumption_agent.benchmarks import hybridqa_query_anchored_formal_runner_v1 as runner
from assumption_agent.benchmarks import hybridqa_query_anchored_operator_v1 as operator
from replication_runtime.bright_minilm_v1.encoder import BrightMiniLMEncoder
from replication_runtime.multihoprag_minilm_v1 import adapter as minilm_adapter
from replication_runtime.qasper_minilm_v1 import binding as minilm_asset
from replication_runtime.qasper_minilm_v1.binding import quantized_cosine_similarity


VERSION = "hybridqa_marginal_replacement_meta_development_v1"
ARCHITECTURE_DECISION_SHA256 = (
    "66040a2776d1293ed8dd86090a57ba6df04a32b9bfa2c0a05a9b7604ef51a958"
)
LABELED_BLOCKS = ("A_form", "A_hold", "M_search")
FAMILIES = acquisition.FAMILIES
TOP_K = operator.TOP_K
MAX_REPLACEMENTS = 2
MAX_PATH_LENGTH = 2
RIDGE_LAMBDA = 1.0
UTILITY_INTEGER_SCALE = 6
PROMOTION_ALPHA = Fraction(1, 10)
INTEGER_SCALE = operator.INTEGER_SCALE
EXPECTED_GPU_RUNTIME_VERSIONS = {
    "huggingface_hub": "0.25.2",
    "numpy": "2.1.3",
    "python": "3.10.12",
    "safetensors": "0.4.5",
    "sentence_transformers": "3.1.1",
    "tokenizers": "0.20.3",
    "torch": "2.4.1+cu118",
    "transformers": "4.45.2",
}
EXPECTED_GPU_CANARY = {
    "device": "cuda:0",
    "dtype": "float32",
    "float32_bytes_sha256": (
        "62fc4780635af3b2e791b2b5eefa02ad8636ef5db93ea5e3342e9d1a96b15f8c"
    ),
    "repeat_count": 2,
    "repeat_exact": True,
    "sentence_count": 256,
}

FEATURE_ORDER = (
    "facet_coverage_mean_delta",
    "facet_coverage_minimum_delta",
    "candidate_residual_gain_mean",
    "candidate_dense_relevance",
    "replaced_dense_relevance",
    "dense_relevance_delta",
    "replaced_unit_deletion_loss_mean",
    "negative_pairwise_redundancy_delta",
    "direct_anchor_indicator",
    "path_length_zero_indicator",
    "path_length_one_indicator",
    "path_length_two_indicator",
    "path_strength",
    "candidate_table_row_indicator",
    "replaced_table_row_indicator",
    "unit_type_change_indicator",
    "original_raw_slot_fraction",
    "candidate_raw_rank_reciprocal",
    "path_family_adjacent_row_indicator",
    "path_family_row_to_passage_indicator",
    "path_family_shared_link_indicator",
    "second_replacement_indicator",
)


class HybridQaMarginalMetaError(RuntimeError):
    """A frozen input, action, model, or aggregate-result contract drifted."""


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise HybridQaMarginalMetaError("value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def self_hashed(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    if field in body:
        raise HybridQaMarginalMetaError("self-hash field already exists")
    return {**dict(body), field: stable_hash(body)}


def verify_self_hash(value: Mapping[str, Any], field: str) -> str:
    if not isinstance(value, Mapping):
        raise HybridQaMarginalMetaError("self-hashed value is not a mapping")
    body = dict(value)
    declared = body.pop(field, None)
    if (
        not isinstance(declared, str)
        or len(declared) != 64
        or stable_hash(body) != declared
    ):
        raise HybridQaMarginalMetaError(f"{field} drifted")
    return declared


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_canonical_json(
    path: Path,
    *,
    expected_file_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
    except OSError as exc:
        raise HybridQaMarginalMetaError(f"input unavailable: {path.name}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or len(raw) > 32 * 1024 * 1024
        or (
            expected_file_sha256 is not None
            and _sha256_bytes(raw) != expected_file_sha256
        )
    ):
        raise HybridQaMarginalMetaError(f"input identity drifted: {path.name}")
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HybridQaMarginalMetaError(f"input JSON invalid: {path.name}") from exc
    if (
        not isinstance(value, dict)
        or raw != _canonical_bytes(value, newline=True)
    ):
        raise HybridQaMarginalMetaError(f"input is not canonical: {path.name}")
    return value


@dataclass(frozen=True)
class ViewItem:
    commitment: str
    question: str
    question_postag: str


@dataclass(frozen=True)
class Label:
    commitment: str
    family: str
    gold: tuple[int, ...]


@dataclass(frozen=True)
class Corpus:
    articles: tuple[minilm_adapter.ArticleText, ...]
    graph: operator.TypedCorpusGraph
    pack_sha256: str


@dataclass(frozen=True)
class PortableCorpusIndex:
    article_count: int
    encoder_receipt_sha256: str
    article_chunk_ranges: tuple[tuple[int, int], ...]
    chunk_vectors: np.ndarray
    normalized_article_sha256s: tuple[str, ...]
    index_sha256: str


@dataclass(frozen=True)
class DiagnosticItem:
    block: str
    family: str
    commitment: str
    gold: tuple[int, ...]
    tensor: operator.QuerySemanticTensor
    raw_top5: tuple[int, ...]
    raw_rank: tuple[int, ...]
    reachability: tuple[operator.ReachabilityRecord, ...]
    candidates: tuple[int, ...]


@dataclass(frozen=True)
class CandidateAction:
    slot: int
    candidate: int
    output: tuple[int, ...]
    features: tuple[float, ...]


@dataclass(frozen=True)
class PolicyOutcome:
    output: tuple[int, ...]
    replacements: int


def verify_gpu_runtime_binding(
    *,
    asset_manifest_path: Path,
    model_root: Path,
) -> Mapping[str, object]:
    """Verify immutable model bytes and the already-qualified 311 GPU runtime."""

    manifest_path, asset = minilm_asset._load_asset_manifest(asset_manifest_path)  # noqa: SLF001
    minilm_asset._verify_manifest_contract(asset)  # noqa: SLF001
    verified_root = minilm_asset._verify_model_tree(asset, model_root)  # noqa: SLF001
    versions: dict[str, str] = {
        "python": ".".join(map(str, __import__("sys").version_info[:3]))
    }
    for key, (distribution, module_name) in minilm_asset._PACKAGE_TO_MODULE.items():  # noqa: SLF001
        try:
            distribution_version = metadata.version(distribution)
            module_version = str(getattr(import_module(module_name), "__version__"))
        except (ImportError, AttributeError, metadata.PackageNotFoundError) as exc:
            raise HybridQaMarginalMetaError(
                f"GPU MiniLM package missing: {distribution}"
            ) from exc
        if key != "torch" and distribution_version != module_version:
            raise HybridQaMarginalMetaError("GPU MiniLM package identity drifted")
        versions[key] = module_version
    if versions != EXPECTED_GPU_RUNTIME_VERSIONS:
        raise HybridQaMarginalMetaError("GPU MiniLM runtime versions drifted")
    return {
        "asset_file_sha256": minilm_asset.ASSET_FILE_SHA256,
        "asset_manifest_path": str(manifest_path),
        "asset_sha256": minilm_asset.ASSET_SELF_SHA256,
        "embedding_dimension": minilm_asset.EMBEDDING_DIMENSION,
        "maximum_sequence_length": minilm_asset.MAXIMUM_SEQUENCE_LENGTH,
        "model_root": str(verified_root),
        "model_tree_sha256": minilm_asset.MODEL_TREE_SHA256,
        "runtime_versions": versions,
        "status": "verified_HybridQA_marginal_meta_GPU_runtime",
        "weights_sha256": minilm_asset.WEIGHTS_SHA256,
    }


def portable_encoder_receipt_sha256(encoder: runner.Encoder) -> str:
    runtime = getattr(encoder, "runtime_receipt", None)
    canary = getattr(encoder, "canary_receipt", None)
    if (
        not isinstance(runtime, Mapping)
        or not isinstance(canary, Mapping)
        or dict(canary) != EXPECTED_GPU_CANARY
        or runtime.get("runtime_versions") != EXPECTED_GPU_RUNTIME_VERSIONS
        or runtime.get("model_tree_sha256") != minilm_asset.MODEL_TREE_SHA256
        or runtime.get("weights_sha256") != minilm_asset.WEIGHTS_SHA256
        or runtime.get("status")
        != "verified_HybridQA_marginal_meta_GPU_runtime"
    ):
        raise HybridQaMarginalMetaError("portable GPU encoder receipt drifted")
    return stable_hash(
        {
            "canary_receipt": dict(canary),
            "runtime_receipt": dict(runtime),
            "version": VERSION,
        }
    )


def open_gpu_encoder(
    *,
    asset_manifest_path: Path,
    model_root: Path,
) -> BrightMiniLMEncoder:
    encoder = BrightMiniLMEncoder(
        asset_manifest=asset_manifest_path,
        model_root=model_root,
        runtime_binding_verifier=verify_gpu_runtime_binding,
    )
    portable_encoder_receipt_sha256(encoder)
    return encoder


def _validated_embedding_matrix(
    matrix: object,
    *,
    rows: int,
) -> np.ndarray:
    values = np.asarray(matrix)
    if (
        values.shape != (rows, minilm_asset.EMBEDDING_DIMENSION)
        or values.dtype != np.float32
        or not np.isfinite(values).all()
    ):
        raise HybridQaMarginalMetaError("portable embedding matrix drifted")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-5):
        raise HybridQaMarginalMetaError("portable embedding normalization drifted")
    return np.ascontiguousarray(values, dtype=np.float32)


def build_portable_corpus_index(
    *,
    articles: Sequence[minilm_adapter.ArticleText],
    encoder: runner.Encoder,
) -> PortableCorpusIndex:
    encoder_sha = portable_encoder_receipt_sha256(encoder)
    rows = tuple(articles)
    if not rows or any(
        not isinstance(row, minilm_adapter.ArticleText)
        or row.article_i != index
        for index, row in enumerate(rows)
    ):
        raise HybridQaMarginalMetaError("portable corpus article order drifted")
    chunks: list[str] = []
    ranges: list[tuple[int, int]] = []
    article_hashes: list[str] = []
    for row in rows:
        serialized = minilm_adapter.serialize_article_chunks(row.title, row.body)
        start = len(chunks)
        chunks.extend(serialized)
        ranges.append((start, len(chunks)))
        article_hashes.append(
            stable_hash(
                {
                    "article_i": row.article_i,
                    "chunks": list(serialized),
                    "serialization_version": minilm_adapter.VERSION,
                }
            )
        )
    if not chunks or len(chunks) > 16_384:
        raise HybridQaMarginalMetaError("portable corpus chunk bound drifted")
    matrix = _validated_embedding_matrix(
        encoder.encode(tuple(chunks)),
        rows=len(chunks),
    )
    body = {
        "article_chunk_ranges": [list(value) for value in ranges],
        "article_count": len(rows),
        "article_sha256s": article_hashes,
        "chunk_matrix_sha256": _sha256_bytes(
            matrix.astype("<f4", copy=False).tobytes(order="C")
        ),
        "chunk_shape": list(matrix.shape),
        "encoder_receipt_sha256": encoder_sha,
        "serialization_version": minilm_adapter.VERSION,
        "version": VERSION,
    }
    matrix.setflags(write=False)
    return PortableCorpusIndex(
        article_count=len(rows),
        encoder_receipt_sha256=encoder_sha,
        article_chunk_ranges=tuple(ranges),
        chunk_vectors=matrix,
        normalized_article_sha256s=tuple(article_hashes),
        index_sha256=stable_hash(body),
    )


def build_portable_query_tensors(
    *,
    rows: Sequence[runner.BulkQueryInput],
    index: PortableCorpusIndex,
    encoder: runner.Encoder,
) -> dict[str, operator.QuerySemanticTensor]:
    canonical_rows = tuple(sorted(rows, key=lambda row: row.item_commitment_sha256))
    if (
        not canonical_rows
        or any(not isinstance(row, runner.BulkQueryInput) for row in canonical_rows)
        or len({row.item_commitment_sha256 for row in canonical_rows})
        != len(canonical_rows)
        or index.article_count != operator.CORPUS_UNIT_COUNT
        or portable_encoder_receipt_sha256(encoder)
        != index.encoder_receipt_sha256
    ):
        raise HybridQaMarginalMetaError("portable query input binding drifted")

    facets_by_item: dict[str, tuple[operator.QueryFacet, ...]] = {}
    questions_by_item: dict[str, str] = {}
    facet_slices: dict[str, tuple[int, int]] = {}
    texts: list[str] = []
    for row in canonical_rows:
        facets = runner.extract_query_facets(row.question, row.question_postag)
        facets_by_item[row.item_commitment_sha256] = facets
        questions_by_item[row.item_commitment_sha256] = minilm_adapter.canonical_text(
            row.question,
            field="question",
        )
        start = len(texts)
        texts.extend(facet.normalized_text for facet in facets)
        facet_slices[row.item_commitment_sha256] = (start, len(texts))
    question_offsets: dict[str, int] = {}
    for row in canonical_rows:
        question_offsets[row.item_commitment_sha256] = len(texts)
        texts.append(questions_by_item[row.item_commitment_sha256])
    prototype_start = len(texts)
    texts.extend(
        minilm_adapter.CAPABILITY_PROTOTYPES[name]
        for name in minilm_adapter.CAPABILITY_ORDER
    )
    if len(texts) > 16_384:
        raise HybridQaMarginalMetaError("portable query batch bound drifted")
    matrix = _validated_embedding_matrix(
        encoder.encode(tuple(texts)),
        rows=len(texts),
    )
    prototype_vectors = matrix[prototype_start : prototype_start + 3]
    output: dict[str, operator.QuerySemanticTensor] = {}
    for row in canonical_rows:
        commitment = row.item_commitment_sha256
        facets = facets_by_item[commitment]
        start, stop = facet_slices[commitment]
        coverage = tuple(
            tuple(
                max(
                    quantized_cosine_similarity(
                        facet_vector,
                        index.chunk_vectors[chunk_i],
                    )
                    for chunk_i in range(chunk_start, chunk_stop)
                )
                for chunk_start, chunk_stop in index.article_chunk_ranges
            )
            for facet_vector in matrix[start:stop]
        )
        question_vector = matrix[question_offsets[commitment]]
        # Preserve the prior complete semantic schedule although these three
        # values do not enter the HybridQA tensor.
        tuple(
            quantized_cosine_similarity(question_vector, prototype_vector)
            for prototype_vector in prototype_vectors
        )
        dense = tuple(
            max(
                quantized_cosine_similarity(
                    question_vector,
                    index.chunk_vectors[chunk_i],
                )
                for chunk_i in range(chunk_start, chunk_stop)
            )
            for chunk_start, chunk_stop in index.article_chunk_ranges
        )
        anchors: list[tuple[int, ...]] = []
        for coverage_row in coverage:
            selected = {
                ordinal
                for ordinal in sorted(
                    (
                        index_i
                        for index_i, score in enumerate(coverage_row)
                        if score > 0
                    ),
                    key=lambda index_i: (-coverage_row[index_i], index_i),
                )[: runner.DIRECT_ANCHOR_K]
            }
            anchors.append(
                tuple(
                    coverage_row[index_i] if index_i in selected else 0
                    for index_i in range(operator.CORPUS_UNIT_COUNT)
                )
            )
        query_sha = hashlib.sha256(
            questions_by_item[commitment].casefold().encode("utf-8")
        ).hexdigest()
        try:
            output[commitment] = operator.make_query_semantic_tensor(
                query_sha256=query_sha,
                facets=facets,
                semantic_coverage_ints=coverage,
                direct_anchor_strength_ints=anchors,
                dense_relevance_ints=dense,
            )
        except operator.HybridQaOperatorError as exc:
            raise HybridQaMarginalMetaError(
                "portable semantic tensor formation failed"
            ) from exc
    return output


def _validate_gold_topology(
    *,
    family: str,
    gold: Sequence[int],
    graph: operator.TypedCorpusGraph,
) -> None:
    units = tuple(graph.units[index] for index in gold)
    rows = tuple(unit for unit in units if unit.unit_type == "table_row")
    passages = tuple(unit for unit in units if unit.unit_type == "linked_passage")
    if len({unit.table_key for unit in units}) != 1:
        raise HybridQaMarginalMetaError("gold topology crosses tables")
    if family == "TABLE_ONLY":
        valid = len(units) == 1 and len(rows) == 1 and not passages
    elif family == "PASSAGE_ONLY":
        valid = len(units) == 2 and len(rows) == 1 and len(passages) == 1
    elif family == "DUAL_TABLE_PASSAGE":
        valid = (
            len(units) in {2, 3}
            and len(rows) in {1, 2}
            and len(passages) == 1
        )
    else:
        valid = False
    if not valid:
        raise HybridQaMarginalMetaError("gold family topology drifted")
    if passages:
        target = passages[0].link_target_keys[0]
        if not any(target in row.link_target_keys for row in rows):
            raise HybridQaMarginalMetaError("gold typed link topology drifted")


def load_consumed_packs(
    acquisition_root: Path,
) -> tuple[Corpus, dict[str, tuple[ViewItem, ...]], dict[str, dict[str, Label]]]:
    """Load and fully revalidate only the three already-consumed labeled blocks."""

    try:
        root_metadata = acquisition_root.lstat()
    except OSError as exc:
        raise HybridQaMarginalMetaError("acquisition root unavailable") from exc
    if (
        stat.S_ISLNK(root_metadata.st_mode)
        or not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_IMODE(root_metadata.st_mode) != 0o500
    ):
        raise HybridQaMarginalMetaError("acquisition root mode drifted")
    public = _load_canonical_json(acquisition_root / acquisition.PUBLIC_FILENAME)
    verify_self_hash(public, "acquisition_receipt_sha256")
    file_hashes = public.get("private_pack_file_sha256s")
    if (
        public.get("schema") != f"{acquisition.VERSION}_public_receipt"
        or public.get("status") != "formal_acquisition_complete"
        or public.get("corpus_unit_count") != operator.CORPUS_UNIT_COUNT
        or public.get("retry_replay_or_resample") != 0
        or public.get("online_evaluator_calls") != 0
        or not isinstance(file_hashes, dict)
    ):
        raise HybridQaMarginalMetaError("acquisition public contract drifted")
    required_private_names = {
        acquisition.CORPUS_FILENAME,
        *(
            f"{block}.view.private.json"
            for block in LABELED_BLOCKS
        ),
        *(
            f"{block}.labels.sealed.json"
            for block in LABELED_BLOCKS
        ),
    }
    if any(
        not isinstance(file_hashes.get(name), str)
        or len(file_hashes[name]) != 64
        for name in required_private_names
    ):
        raise HybridQaMarginalMetaError("required private pack binding drifted")

    corpus_pack = _load_canonical_json(
        acquisition_root / acquisition.CORPUS_FILENAME,
        expected_file_sha256=file_hashes.get(acquisition.CORPUS_FILENAME),
    )
    corpus_sha = verify_self_hash(corpus_pack, "corpus_pack_sha256")
    units = corpus_pack.get("units")
    if (
        corpus_pack.get("schema") != f"{acquisition.VERSION}_corpus_pack"
        or corpus_pack.get("unit_count") != operator.CORPUS_UNIT_COUNT
        or not isinstance(units, list)
        or len(units) != operator.CORPUS_UNIT_COUNT
    ):
        raise HybridQaMarginalMetaError("corpus pack envelope drifted")
    articles: list[minilm_adapter.ArticleText] = []
    atomic_units: list[operator.AtomicUnit] = []
    for expected_index, raw in enumerate(units):
        if not isinstance(raw, dict) or set(raw) != {
            "idx",
            "unit_type",
            "title",
            "body",
            "sidecar",
        }:
            raise HybridQaMarginalMetaError("corpus row schema drifted")
        sidecar = raw.get("sidecar")
        if (
            raw.get("idx") != expected_index
            or raw.get("unit_type") not in operator.UNIT_TYPES
            or not isinstance(raw.get("title"), str)
            or not raw["title"].strip()
            or not isinstance(raw.get("body"), str)
            or not raw["body"].strip()
            or not isinstance(sidecar, dict)
            or set(sidecar) != {
                "table_key",
                "row_ordinal",
                "link_target_keys",
            }
            or not isinstance(sidecar.get("table_key"), str)
            or not sidecar["table_key"]
            or (
                raw.get("unit_type") == "table_row"
                and (
                    type(sidecar.get("row_ordinal")) is not int
                    or sidecar["row_ordinal"] < 0
                )
            )
            or (
                raw.get("unit_type") == "linked_passage"
                and sidecar.get("row_ordinal") is not None
            )
            or not isinstance(sidecar.get("link_target_keys"), list)
            or any(
                not isinstance(value, str)
                for value in sidecar["link_target_keys"]
            )
        ):
            raise HybridQaMarginalMetaError("corpus row content drifted")
        articles.append(
            minilm_adapter.ArticleText(
                expected_index,
                str(raw["title"]),
                str(raw["body"]),
            )
        )
        try:
            atomic_units.append(
                operator.AtomicUnit(
                    expected_index,
                    str(raw["unit_type"]),
                    str(sidecar["table_key"]),
                    sidecar["row_ordinal"],
                    tuple(sidecar["link_target_keys"]),
                )
            )
        except (KeyError, operator.HybridQaOperatorError) as exc:
            raise HybridQaMarginalMetaError("corpus typed sidecar drifted") from exc
    try:
        graph = operator.build_typed_graph(atomic_units)
    except operator.HybridQaOperatorError as exc:
        raise HybridQaMarginalMetaError("typed graph formation failed") from exc
    corpus = Corpus(tuple(articles), graph, corpus_sha)

    views: dict[str, tuple[ViewItem, ...]] = {}
    labels: dict[str, dict[str, Label]] = {}
    for block in LABELED_BLOCKS:
        view_name = f"{block}.view.private.json"
        label_name = f"{block}.labels.sealed.json"
        view_pack = _load_canonical_json(
            acquisition_root / view_name,
            expected_file_sha256=file_hashes.get(view_name),
        )
        label_pack = _load_canonical_json(
            acquisition_root / label_name,
            expected_file_sha256=file_hashes.get(label_name),
        )
        view_sha = verify_self_hash(view_pack, "block_view_sha256")
        verify_self_hash(label_pack, "label_pack_sha256")
        view_rows = view_pack.get("items")
        label_rows = label_pack.get("items")
        expected_count = acquisition.BLOCK_COUNTS[block]
        if (
            view_pack.get("schema") != f"{acquisition.VERSION}_block_view"
            or view_pack.get("block") != block
            or view_pack.get("item_count") != expected_count
            or view_pack.get("labels_family_gold_or_table_included") is not False
            or not isinstance(view_rows, list)
            or len(view_rows) != expected_count
            or label_pack.get("schema") != f"{acquisition.VERSION}_label_pack"
            or label_pack.get("block") != block
            or label_pack.get("item_count") != expected_count
            or label_pack.get("block_view_sha256") != view_sha
            or label_pack.get("corpus_pack_sha256") != corpus_sha
            or not isinstance(label_rows, list)
            or len(label_rows) != expected_count
        ):
            raise HybridQaMarginalMetaError("view/label envelope drifted")

        block_view: list[ViewItem] = []
        for ordinal, row in enumerate(view_rows):
            if not isinstance(row, dict) or set(row) != {
                "item_commitment_sha256",
                "question",
                "question_postag",
            }:
                raise HybridQaMarginalMetaError("view row schema drifted")
            commitment = row.get("item_commitment_sha256")
            question = row.get("question")
            postag = row.get("question_postag")
            if (
                not isinstance(commitment, str)
                or not isinstance(question, str)
                or not isinstance(postag, str)
                or commitment
                != acquisition.item_commitment(
                    block=block,
                    ordinal=ordinal,
                    question=question,
                    question_postag=postag,
                )
            ):
                raise HybridQaMarginalMetaError("view commitment drifted")
            block_view.append(ViewItem(commitment, question, postag))
        if len({row.commitment for row in block_view}) != expected_count:
            raise HybridQaMarginalMetaError("view commitment duplicated")

        block_labels: dict[str, Label] = {}
        family_counts: Counter[str] = Counter()
        for row in label_rows:
            if not isinstance(row, dict) or set(row) != {
                "item_commitment_sha256",
                "family",
                "gold_indices",
            }:
                raise HybridQaMarginalMetaError("label row schema drifted")
            commitment = row.get("item_commitment_sha256")
            family = row.get("family")
            gold = row.get("gold_indices")
            if (
                not isinstance(commitment, str)
                or commitment in block_labels
                or family not in FAMILIES
                or not isinstance(gold, list)
                or gold != sorted(set(gold))
                or not 1 <= len(gold) <= 3
                or any(
                    type(value) is not int
                    or not 0 <= value < operator.CORPUS_UNIT_COUNT
                    for value in gold
                )
            ):
                raise HybridQaMarginalMetaError("label row content drifted")
            _validate_gold_topology(family=family, gold=gold, graph=graph)
            block_labels[commitment] = Label(
                commitment,
                str(family),
                tuple(gold),
            )
            family_counts[str(family)] += 1
        if (
            set(block_labels) != {row.commitment for row in block_view}
            or family_counts
            != Counter(
                {
                    family: acquisition.PER_FAMILY_QUOTA[block]
                    for family in FAMILIES
                }
            )
        ):
            raise HybridQaMarginalMetaError("view/label alignment drifted")
        views[block] = tuple(block_view)
        labels[block] = block_labels
    return corpus, views, labels


def _raw_order(tensor: operator.QuerySemanticTensor) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(operator.CORPUS_UNIT_COUNT),
            key=lambda ordinal: (-tensor.dense_relevance_ints[ordinal], ordinal),
        )
    )


def form_items(
    *,
    corpus: Corpus,
    views: Mapping[str, Sequence[ViewItem]],
    labels: Mapping[str, Mapping[str, Label]],
    index: PortableCorpusIndex,
    encoder: runner.Encoder,
) -> tuple[DiagnosticItem, ...]:
    bulk_rows = tuple(
        runner.BulkQueryInput(row.commitment, row.question, row.question_postag)
        for block in LABELED_BLOCKS
        for row in views[block]
    )
    tensors = build_portable_query_tensors(
        rows=bulk_rows,
        index=index,
        encoder=encoder,
    )
    items: list[DiagnosticItem] = []
    for block in LABELED_BLOCKS:
        for row in views[block]:
            tensor = tensors[row.commitment]
            order = _raw_order(tensor)
            ranks = [0] * operator.CORPUS_UNIT_COUNT
            for rank, ordinal in enumerate(order):
                ranks[ordinal] = rank
            reachability = operator._query_anchored_reachability(  # noqa: SLF001
                corpus.graph,
                tensor,
            )
            raw_set = set(order[:TOP_K])
            candidates = tuple(
                ordinal
                for ordinal, record in enumerate(reachability)
                if ordinal not in raw_set
                and record.path_length is not None
                and record.path_length <= MAX_PATH_LENGTH
            )
            label = labels[block][row.commitment]
            items.append(
                DiagnosticItem(
                    block=block,
                    family=label.family,
                    commitment=row.commitment,
                    gold=label.gold,
                    tensor=tensor,
                    raw_top5=tuple(order[:TOP_K]),
                    raw_rank=tuple(ranks),
                    reachability=reachability,
                    candidates=candidates,
                )
            )
    expected_count = sum(acquisition.BLOCK_COUNTS[block] for block in LABELED_BLOCKS)
    if len(items) != expected_count or len({item.commitment for item in items}) != len(
        items
    ):
        raise HybridQaMarginalMetaError("diagnostic item formation drifted")
    return tuple(items)


def _facet_maxima(
    tensor: operator.QuerySemanticTensor,
    selected: Sequence[int],
) -> tuple[int, ...]:
    return tuple(
        max(row.semantic_coverage_ints[ordinal] for ordinal in selected)
        for row in tensor.rows
    )


def _redundancy_int(
    tensor: operator.QuerySemanticTensor,
    selected: Sequence[int],
) -> int:
    total = 0
    for left_index in range(len(selected)):
        for right_index in range(left_index + 1, len(selected)):
            left = selected[left_index]
            right = selected[right_index]
            for row in tensor.rows:
                total += min(
                    max(0, row.semantic_coverage_ints[left]),
                    max(0, row.semantic_coverage_ints[right]),
                )
    return total


def action_features(
    *,
    item: DiagnosticItem,
    graph: operator.TypedCorpusGraph,
    state: Sequence[int],
    slot: int,
    candidate: int,
    step: int,
) -> tuple[float, ...]:
    if (
        len(state) != TOP_K
        or len(set(state)) != TOP_K
        or not 0 <= slot < TOP_K
        or candidate in state
        or candidate not in item.candidates
        or step not in {0, 1}
    ):
        raise HybridQaMarginalMetaError("candidate action input drifted")
    output = list(state)
    replaced = output[slot]
    output[slot] = candidate
    tensor = item.tensor
    old_maxima = _facet_maxima(tensor, state)
    new_maxima = _facet_maxima(tensor, output)
    without = tuple(value for index, value in enumerate(state) if index != slot)
    without_maxima = _facet_maxima(tensor, without)
    facet_count = len(tensor.rows)
    reach = item.reachability[candidate]
    if reach.path_length not in {0, 1, 2}:
        raise HybridQaMarginalMetaError("candidate reachability drifted")
    family_orders = set(reach.path_family_orders)
    redundancy_denominator = (
        math.comb(TOP_K, 2) * facet_count * INTEGER_SCALE
    )
    values = (
        sum(new - old for old, new in zip(old_maxima, new_maxima))
        / (facet_count * INTEGER_SCALE),
        (min(new_maxima) - min(old_maxima)) / INTEGER_SCALE,
        sum(
            max(0, row.semantic_coverage_ints[candidate] - old_maxima[row.facet_i])
            for row in tensor.rows
        )
        / (facet_count * INTEGER_SCALE),
        tensor.dense_relevance_ints[candidate] / INTEGER_SCALE,
        tensor.dense_relevance_ints[replaced] / INTEGER_SCALE,
        (
            tensor.dense_relevance_ints[candidate]
            - tensor.dense_relevance_ints[replaced]
        )
        / INTEGER_SCALE,
        sum(old - deleted for old, deleted in zip(old_maxima, without_maxima))
        / (facet_count * INTEGER_SCALE),
        -(
            _redundancy_int(tensor, output)
            - _redundancy_int(tensor, state)
        )
        / redundancy_denominator,
        float(reach.direct_anchor),
        float(reach.path_length == 0),
        float(reach.path_length == 1),
        float(reach.path_length == 2),
        reach.path_strength_int / INTEGER_SCALE,
        float(graph.units[candidate].unit_type == "table_row"),
        float(graph.units[replaced].unit_type == "table_row"),
        float(graph.units[candidate].unit_type != graph.units[replaced].unit_type),
        slot / (TOP_K - 1),
        1.0 / (item.raw_rank[candidate] + 1),
        float(operator.EDGE_FAMILIES.index(operator.SAME_TABLE_ADJACENT_ROW) in family_orders),
        float(operator.EDGE_FAMILIES.index(operator.ROW_TO_LINKED_PASSAGE) in family_orders),
        float(
            operator.EDGE_FAMILIES.index(operator.SAME_TABLE_SHARED_LINK_TARGET)
            in family_orders
        ),
        float(step == 1),
    )
    if (
        len(values) != len(FEATURE_ORDER)
        or not all(math.isfinite(value) for value in values)
    ):
        raise HybridQaMarginalMetaError("candidate feature vector drifted")
    return tuple(values)


def enumerate_actions(
    *,
    item: DiagnosticItem,
    graph: operator.TypedCorpusGraph,
    state: Sequence[int],
    available_slots: Sequence[int],
    step: int,
) -> tuple[CandidateAction, ...]:
    rows: list[CandidateAction] = []
    selected = set(state)
    for slot in sorted(available_slots):
        for candidate in item.candidates:
            if candidate in selected:
                continue
            output = list(state)
            output[slot] = candidate
            rows.append(
                CandidateAction(
                    slot,
                    candidate,
                    tuple(output),
                    action_features(
                        item=item,
                        graph=graph,
                        state=state,
                        slot=slot,
                        candidate=candidate,
                        step=step,
                    ),
                )
            )
    return tuple(rows)


def _utility(selected: Sequence[int], gold: Sequence[int]) -> Fraction:
    return runner.item_utility(selected, gold)[0]


def oracle_trajectory(
    *,
    item: DiagnosticItem,
    graph: operator.TypedCorpusGraph,
    collect_training_rows: bool,
) -> tuple[PolicyOutcome, tuple[tuple[tuple[float, ...], Fraction], ...]]:
    state = item.raw_top5
    available = list(range(TOP_K))
    states: list[tuple[tuple[tuple[float, ...], Fraction], ...]] = []
    replacements = 0
    for step in range(MAX_REPLACEMENTS):
        actions = enumerate_actions(
            item=item,
            graph=graph,
            state=state,
            available_slots=available,
            step=step,
        )
        if not actions:
            break
        current_utility = _utility(state, item.gold)
        labeled = tuple(
            (action, _utility(action.output, item.gold) - current_utility)
            for action in actions
        )
        if collect_training_rows:
            states.append(tuple((action.features, delta) for action, delta in labeled))
        best_action, best_delta = max(
            labeled,
            key=lambda row: (row[1], -row[0].slot, -row[0].candidate),
        )
        if best_delta <= 0:
            break
        state = best_action.output
        available.remove(best_action.slot)
        replacements += 1
    flattened = tuple(row for state_rows in states for row in state_rows)
    return PolicyOutcome(tuple(state), replacements), flattened


def fit_marginal_ridge(
    *,
    items: Sequence[DiagnosticItem],
    graph: operator.TypedCorpusGraph,
) -> tuple[np.ndarray, dict[str, Any]]:
    state_groups: list[tuple[tuple[tuple[float, ...], Fraction], ...]] = []
    item_state_counts: list[int] = []
    for item in items:
        state = item.raw_top5
        available = list(range(TOP_K))
        groups_for_item: list[tuple[tuple[tuple[float, ...], Fraction], ...]] = []
        for step in range(MAX_REPLACEMENTS):
            actions = enumerate_actions(
                item=item,
                graph=graph,
                state=state,
                available_slots=available,
                step=step,
            )
            if not actions:
                break
            current_utility = _utility(state, item.gold)
            group = tuple(
                (
                    action.features,
                    _utility(action.output, item.gold) - current_utility,
                )
                for action in actions
            )
            groups_for_item.append(group)
            best_index = max(
                range(len(actions)),
                key=lambda index: (
                    group[index][1],
                    -actions[index].slot,
                    -actions[index].candidate,
                ),
            )
            if group[best_index][1] <= 0:
                break
            best = actions[best_index]
            state = best.output
            available.remove(best.slot)
        if not groups_for_item:
            raise HybridQaMarginalMetaError("training item has no candidate actions")
        item_state_counts.append(len(groups_for_item))
        state_groups.extend(groups_for_item)

    dimension = len(FEATURE_ORDER)
    gram = np.eye(dimension, dtype=np.float64) * RIDGE_LAMBDA
    target = np.zeros(dimension, dtype=np.float64)
    row_count = 0
    group_cursor = 0
    for state_count in item_state_counts:
        for _ in range(state_count):
            group = state_groups[group_cursor]
            group_cursor += 1
            weight = 1.0 / (state_count * len(group))
            matrix = np.asarray([row[0] for row in group], dtype=np.float64)
            response = np.asarray([float(row[1]) for row in group], dtype=np.float64)
            gram += matrix.T @ (matrix * weight)
            target += matrix.T @ (response * weight)
            row_count += len(group)
    try:
        weights = np.linalg.solve(gram, target)
    except np.linalg.LinAlgError as exc:
        raise HybridQaMarginalMetaError("fixed ridge solve failed") from exc
    if (
        weights.shape != (dimension,)
        or not np.isfinite(weights).all()
        or group_cursor != len(state_groups)
    ):
        raise HybridQaMarginalMetaError("fixed ridge output drifted")
    little_endian = weights.astype("<f8", copy=False).tobytes(order="C")
    receipt = {
        "feature_count": dimension,
        "item_count": len(items),
        "oracle_state_count": len(state_groups),
        "candidate_action_row_count": row_count,
        "lambda": "1",
        "no_intercept": True,
        "per_item_total_weight_per_oracle_state_count": True,
        "weights_float64_le_sha256": _sha256_bytes(little_endian),
        "weights_persisted": False,
    }
    return weights, receipt


def apply_learned_policy(
    *,
    item: DiagnosticItem,
    graph: operator.TypedCorpusGraph,
    weights: np.ndarray,
) -> PolicyOutcome:
    state = item.raw_top5
    available = list(range(TOP_K))
    replacements = 0
    for step in range(MAX_REPLACEMENTS):
        actions = enumerate_actions(
            item=item,
            graph=graph,
            state=state,
            available_slots=available,
            step=step,
        )
        if not actions:
            break
        matrix = np.asarray([action.features for action in actions], dtype=np.float64)
        scores = matrix @ weights
        best_index = max(
            range(len(actions)),
            key=lambda index: (
                float(scores[index]),
                -actions[index].slot,
                -actions[index].candidate,
            ),
        )
        # The zero vector and score are the explicit no-op action.
        if not math.isfinite(float(scores[best_index])) or scores[best_index] <= 0:
            break
        best = actions[best_index]
        state = best.output
        available.remove(best.slot)
        replacements += 1
    return PolicyOutcome(tuple(state), replacements)


def exact_sign_flip_p(deltas: Sequence[Fraction]) -> Fraction:
    nonzero = tuple(delta for delta in deltas if delta)
    if not nonzero or sum(nonzero, Fraction(0)) <= 0:
        return Fraction(1)
    scaled: list[int] = []
    for delta in nonzero:
        value = delta * UTILITY_INTEGER_SCALE
        if value.denominator != 1:
            raise HybridQaMarginalMetaError("utility scale drifted")
        scaled.append(abs(value.numerator))
    observed = sum(
        int(delta * UTILITY_INTEGER_SCALE)
        for delta in nonzero
    )
    distribution: Counter[int] = Counter({0: 1})
    for magnitude in scaled:
        following: Counter[int] = Counter()
        for subtotal, count in distribution.items():
            following[subtotal + magnitude] += count
            following[subtotal - magnitude] += count
        distribution = following
    tail = sum(count for subtotal, count in distribution.items() if subtotal >= observed)
    return Fraction(tail, 1 << len(nonzero))


def _fraction_payload(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _arm_summary(
    rows: Sequence[tuple[Fraction, bool]],
) -> dict[str, Any]:
    return {
        "total_utility": _fraction_payload(sum((row[0] for row in rows), Fraction(0))),
        "complete_count": sum(row[1] for row in rows),
        "item_count": len(rows),
    }


def _comparison_summary(
    treatment: Sequence[tuple[Fraction, bool]],
    control: Sequence[tuple[Fraction, bool]],
) -> dict[str, Any]:
    deltas = tuple(left[0] - right[0] for left, right in zip(treatment, control, strict=True))
    return {
        "total_utility_delta": _fraction_payload(sum(deltas, Fraction(0))),
        "complete_count_delta": sum(left[1] - right[1] for left, right in zip(treatment, control, strict=True)),
        "positive_item_count": sum(delta > 0 for delta in deltas),
        "negative_item_count": sum(delta < 0 for delta in deltas),
        "zero_item_count": sum(delta == 0 for delta in deltas),
        "exact_one_sided_magnitude_sign_flip_p": _fraction_payload(
            exact_sign_flip_p(deltas)
        ),
    }


def evaluate_crossfit(
    *,
    items: Sequence[DiagnosticItem],
    graph: operator.TypedCorpusGraph,
) -> dict[str, Any]:
    by_block = {
        block: tuple(item for item in items if item.block == block)
        for block in LABELED_BLOCKS
    }
    fold_receipts: dict[str, Any] = {}
    pooled_rows: list[dict[str, Any]] = []
    for held_block in LABELED_BLOCKS:
        train_blocks = tuple(block for block in LABELED_BLOCKS if block != held_block)
        train_items = tuple(item for block in train_blocks for item in by_block[block])
        weights, fit_receipt = fit_marginal_ridge(items=train_items, graph=graph)
        held_rows: list[dict[str, Any]] = []
        for item in by_block[held_block]:
            learned = apply_learned_policy(item=item, graph=graph, weights=weights)
            oracle, _unused = oracle_trajectory(
                item=item,
                graph=graph,
                collect_training_rows=False,
            )
            p6 = operator.run_recipe(
                recipe_id="R3_P6_PATH2_B2",
                graph=graph,
                semantic_tensor=item.tensor,
            ).output_top5
            row = {
                "block": held_block,
                "family": item.family,
                "raw": runner.item_utility(item.raw_top5, item.gold),
                "learned": runner.item_utility(learned.output, item.gold),
                "oracle": runner.item_utility(oracle.output, item.gold),
                "p6_path2": runner.item_utility(p6, item.gold),
                "learned_replacements": learned.replacements,
                "oracle_replacements": oracle.replacements,
                "candidate_count": len(item.candidates),
            }
            held_rows.append(row)
            pooled_rows.append(row)

        def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
            arms = {
                arm: [row[arm] for row in rows]
                for arm in ("raw", "learned", "oracle", "p6_path2")
            }
            return {
                "arms": {arm: _arm_summary(values) for arm, values in arms.items()},
                "comparisons": {
                    "learned_minus_raw": _comparison_summary(arms["learned"], arms["raw"]),
                    "learned_minus_p6_path2": _comparison_summary(
                        arms["learned"], arms["p6_path2"]
                    ),
                    "oracle_minus_raw": _comparison_summary(arms["oracle"], arms["raw"]),
                    "p6_path2_minus_raw": _comparison_summary(
                        arms["p6_path2"], arms["raw"]
                    ),
                },
                "learned_replacement_count": sum(
                    row["learned_replacements"] for row in rows
                ),
                "oracle_replacement_count": sum(
                    row["oracle_replacements"] for row in rows
                ),
                "candidate_count": {
                    "minimum": min(row["candidate_count"] for row in rows),
                    "maximum": max(row["candidate_count"] for row in rows),
                    "sum": sum(row["candidate_count"] for row in rows),
                },
            }

        fold_receipts[held_block] = {
            "held_block": held_block,
            "train_blocks": list(train_blocks),
            "fit": fit_receipt,
            "aggregate": summarize(held_rows),
            "families": {
                family: summarize(
                    [row for row in held_rows if row["family"] == family]
                )
                for family in FAMILIES
            },
        }

    def pooled_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        arms = {
            arm: [row[arm] for row in rows]
            for arm in ("raw", "learned", "oracle", "p6_path2")
        }
        return {
            "arms": {arm: _arm_summary(values) for arm, values in arms.items()},
            "comparisons": {
                "learned_minus_raw": _comparison_summary(arms["learned"], arms["raw"]),
                "learned_minus_p6_path2": _comparison_summary(
                    arms["learned"], arms["p6_path2"]
                ),
                "oracle_minus_raw": _comparison_summary(arms["oracle"], arms["raw"]),
            },
            "learned_replacement_count": sum(
                row["learned_replacements"] for row in rows
            ),
            "oracle_replacement_count": sum(
                row["oracle_replacements"] for row in rows
            ),
        }

    pooled = pooled_summary(pooled_rows)
    pooled_families = {
        family: pooled_summary(
            [row for row in pooled_rows if row["family"] == family]
        )
        for family in FAMILIES
    }

    def delta_positive(summary: Mapping[str, Any], comparison: str) -> bool:
        numerator, _denominator = summary["comparisons"][comparison][
            "total_utility_delta"
        ]
        return numerator > 0

    requirements = {
        "oracle_positive_every_block_and_family": all(
            delta_positive(fold_receipts[block]["aggregate"], "oracle_minus_raw")
            and all(
                delta_positive(
                    fold_receipts[block]["families"][family],
                    "oracle_minus_raw",
                )
                for family in FAMILIES
            )
            for block in LABELED_BLOCKS
        ),
        "learned_positive_every_held_block": all(
            delta_positive(
                fold_receipts[block]["aggregate"],
                "learned_minus_raw",
            )
            for block in LABELED_BLOCKS
        ),
        "learned_positive_every_pooled_family": all(
            delta_positive(pooled_families[family], "learned_minus_raw")
            for family in FAMILIES
        ),
        "learned_positive_over_fixed_p6_path2_pooled": delta_positive(
            pooled,
            "learned_minus_p6_path2",
        ),
        "learned_vs_raw_pooled_exact_p_at_most_point_one": (
            Fraction(
                *pooled["comparisons"]["learned_minus_raw"][
                    "exact_one_sided_magnitude_sign_flip_p"
                ]
            )
            <= PROMOTION_ALPHA
        ),
    }
    return {
        "folds": fold_receipts,
        "pooled": pooled,
        "pooled_families": pooled_families,
        "go_requirements": requirements,
        "decision": (
            "GO_ONE_INDEPENDENT_CONFIRMATORY_STUDY"
            if all(requirements.values())
            else "STOP_CURRENT_ARCHITECTURE"
        ),
    }


def build_safe_result(
    *,
    corpus: Corpus,
    items: Sequence[DiagnosticItem],
    evaluation: Mapping[str, Any],
    encoder: runner.Encoder,
    freeze_self_sha256: str,
) -> dict[str, Any]:
    block_counts = Counter(item.block for item in items)
    family_counts = Counter(item.family for item in items)
    body = {
        "schema": f"{VERSION}_safe_result",
        "version": VERSION,
        "status": "complete",
        "architecture_decision_sha256": ARCHITECTURE_DECISION_SHA256,
        "implementation_freeze_self_sha256": freeze_self_sha256,
        "scope": {
            "fresh_efficacy_claim": False,
            "source_or_cohort_newly_consumed": False,
            "consumed_blocks": list(LABELED_BLOCKS),
            "item_count": len(items),
            "block_counts": dict(sorted(block_counts.items())),
            "family_counts": dict(sorted(family_counts.items())),
            "per_item_or_private_content_persisted": False,
        },
        "mechanism": {
            "maximum_replacements": MAX_REPLACEMENTS,
            "maximum_typed_path_length": MAX_PATH_LENGTH,
            "candidate_must_be_outside_original_RAW_top5": True,
            "original_RAW_slot_may_be_replaced_at_most_once": True,
            "candidate_filter_beyond_typed_reachability": False,
            "explicit_no_op_score": "0",
            "feature_order": list(FEATURE_ORDER),
            "ridge_lambda": "1",
            "ridge_intercept": False,
        },
        "bindings": {
            "corpus_pack_sha256": corpus.pack_sha256,
            "graph_sha256": corpus.graph.graph_sha256,
            "minilm_encoder_receipt_sha256": portable_encoder_receipt_sha256(encoder),
            "minilm_canary_receipt_sha256": stable_hash(
                dict(encoder.canary_receipt)
            ),
        },
        "evaluation": dict(evaluation),
        "activity_counts": {
            "new_source_download": 0,
            "fresh_selection": 0,
            "official_TEST_access": 0,
            "online_or_API_evaluation": 0,
            "HippoRAG_candidate_or_feature_access": 0,
            "retry_replay_resample": 0,
        },
    }
    return self_hashed(body, "result_self_sha256")


__all__ = [
    "ARCHITECTURE_DECISION_SHA256",
    "CandidateAction",
    "DiagnosticItem",
    "FEATURE_ORDER",
    "FAMILIES",
    "HybridQaMarginalMetaError",
    "LABELED_BLOCKS",
    "MAX_PATH_LENGTH",
    "MAX_REPLACEMENTS",
    "PROMOTION_ALPHA",
    "PolicyOutcome",
    "RIDGE_LAMBDA",
    "VERSION",
    "action_features",
    "apply_learned_policy",
    "build_safe_result",
    "enumerate_actions",
    "evaluate_crossfit",
    "exact_sign_flip_p",
    "fit_marginal_ridge",
    "form_items",
    "load_consumed_packs",
    "oracle_trajectory",
    "self_hashed",
    "stable_hash",
    "verify_self_hash",
]
