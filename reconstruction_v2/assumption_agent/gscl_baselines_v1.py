"""Frozen non-scoring baselines for the GSCL controlled harness."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib import import_module
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

# The legacy implementation intentionally remains outside the reconstruction_v2
# package.  Resolve that frozen workspace dependency from this file's location,
# rather than relying on the caller's current directory or PYTHONPATH.
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from assumption_os.structural_patterns import (
    DEFAULT_STRUCTURAL_PATTERNS,
    score_structural_morphism,
    search_structural_patterns,
)

from .generalized_structural_correspondence_v1 import (
    GSCLSchemaRegistry,
    strict_content_hash,
)
from .meta_assumption import UniversalAssumptionOntology
from replication_runtime.qasper_minilm_v1.binding import (
    ASSET_SELF_SHA256,
    EMBEDDING_DIMENSION,
    MAXIMUM_SEQUENCE_LENGTH,
    MODEL_TREE_SHA256,
    WEIGHTS_SHA256,
    OfflineMiniLMEncoder,
    quantized_cosine_similarity,
)


SEMANTIC_BASELINE_VERSION = "gscl.semantic.only.minilm.v1"
LEGACY_BASELINE_VERSION = "gscl.legacy.keyword.morphism.v1"
LEGACY_SOURCE_SHA256 = (
    "64602b951cb53e3d4baad588668ab4ac7b9d91fcbe6ac8e6bf6d7460830dad93"
)
LEGACY_REGISTRY_HASH = (
    "f31abd61bee409a916f3fccf3ce82f3728e8271972f65d8c628408679ff2e2b3"
)
LEGACY_MIN_SCORE = 0.22

_LEGACY_LAW_MAP = {
    "pat_conservation_balance": "gscl.v1.t15_closed_balance",
    "pat_decomposition_composition": "gscl.v1.t09_path_composition",
    "pat_monotone_progress": "gscl.v1.t17_monotone_order",
}

BASELINE_CONTRACT = {
    "implementation_closure": {
        "version": "gscl.baseline.implementation.closure.v1",
        "actual_import_origin_required": True,
        "content_address_every_internal_module": True,
        "semantic_and_legacy_lanes_separately_bound": True,
        "closure_must_remain_stable_during_qualification": True,
    },
    "semantic_only": {
        "version": SEMANTIC_BASELINE_VERSION,
        "input": "same_raw_evidence_lossless_utf8_text",
        "utf8_roundtrip_required": True,
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "asset_sha256": ASSET_SELF_SHA256,
        "model_tree_sha256": MODEL_TREE_SHA256,
        "weights_sha256": WEIGHTS_SHA256,
        "prototype_fields": ["law_kind", "uao_claim_schema"],
        "source_chunking": (
            "deterministic_greedy_complete_record_lines"
        ),
        "maximum_chunk_tokens_including_special_tokens": (
            MAXIMUM_SEQUENCE_LENGTH
        ),
        "single_record_line_overflow": "fail_closed",
        "source_aggregation": (
            "equal_weight_float64_mean_then_l2_normalize_float32"
        ),
        "full_source_character_coverage_required": True,
        "truncated_chunks_allowed": False,
        "role_binding": False,
        "residual_verifier": False,
        "threshold_tuned_on_controlled_corpus": False,
    },
    "legacy_keyword": {
        "version": LEGACY_BASELINE_VERSION,
        "input": "same_raw_evidence_lossless_utf8_text",
        "utf8_roundtrip_required": True,
        "legacy_source_sha256": LEGACY_SOURCE_SHA256,
        "legacy_registry_hash": LEGACY_REGISTRY_HASH,
        "legacy_pattern_count": 10,
        "min_score_microunits": 220_000,
        "top_n": 1,
        "compatible_law_map": dict(sorted(_LEGACY_LAW_MAP.items())),
        "new_markers_added": False,
    },
    "flat_label_no_verifier": {
        "version": "gscl.flat.label.no.verifier.v1",
        "shared_extractor_and_binder": True,
        "episode_validation": True,
        "generic_binding_validation": True,
        "semantic_bridge": False,
        "residual_verifier": False,
        "hard_negative_verifier": False,
        "correspondence_verifier": False,
    },
}
BASELINE_CONTRACT_HASH = strict_content_hash(BASELINE_CONTRACT)


_IMPLEMENTATION_CLOSURE_VERSION = (
    "gscl.baseline.implementation.closure.v1"
)
_IMPLEMENTATION_MODULE_PATHS = {
    "assumption_agent.evaluation": (
        "reconstruction_v2/assumption_agent/evaluation.py"
    ),
    "assumption_agent.events": (
        "reconstruction_v2/assumption_agent/events.py"
    ),
    "assumption_agent.generalized_structural_correspondence_v1": (
        "reconstruction_v2/assumption_agent/"
        "generalized_structural_correspondence_v1.py"
    ),
    "assumption_agent.gscl_baselines_v1": (
        "reconstruction_v2/assumption_agent/gscl_baselines_v1.py"
    ),
    "assumption_agent.meta_assumption": (
        "reconstruction_v2/assumption_agent/meta_assumption.py"
    ),
    "assumption_agent.models": (
        "reconstruction_v2/assumption_agent/models.py"
    ),
    "assumption_agent.runtime": (
        "reconstruction_v2/assumption_agent/runtime.py"
    ),
    "assumption_agent.universal_assumption_ontology_v1": (
        "reconstruction_v2/assumption_agent/"
        "universal_assumption_ontology_v1.py"
    ),
    "assumption_os.formal_mapping": (
        "assumption_os/formal_mapping.py"
    ),
    "assumption_os.graph_memory": "assumption_os/graph_memory.py",
    "assumption_os.schema": "assumption_os/schema.py",
    "assumption_os.structural_patterns": (
        "assumption_os/structural_patterns.py"
    ),
    "replication_runtime.qasper_minilm_v1.binding": (
        "reconstruction_v2/replication_runtime/"
        "qasper_minilm_v1/binding.py"
    ),
}
_IMPLEMENTATION_LANE_MODULES = {
    "semantic_only": (
        "assumption_agent.evaluation",
        "assumption_agent.events",
        "assumption_agent.generalized_structural_correspondence_v1",
        "assumption_agent.gscl_baselines_v1",
        "assumption_agent.meta_assumption",
        "assumption_agent.models",
        "assumption_agent.runtime",
        "assumption_agent.universal_assumption_ontology_v1",
        "replication_runtime.qasper_minilm_v1.binding",
    ),
    "legacy_keyword": (
        "assumption_agent.gscl_baselines_v1",
        "assumption_os.formal_mapping",
        "assumption_os.graph_memory",
        "assumption_os.schema",
        "assumption_os.structural_patterns",
    ),
}


@dataclass(frozen=True)
class SemanticOnlyPrediction:
    item_key: str
    predicted_law_id: str
    top_score: int
    score_commitment: str

    def safe_payload(self) -> dict[str, Any]:
        return {
            "item_key": self.item_key,
            "predicted_law_id": self.predicted_law_id,
            "top_score": self.top_score,
            "score_commitment": self.score_commitment,
        }


@dataclass(frozen=True)
class SemanticOnlyBatch:
    predictions: tuple[SemanticOnlyPrediction, ...]
    pair_similarities: Mapping[str, int]
    prototype_hash: str
    embedding_matrix_hash: str
    chunk_embedding_matrix_hash: str
    runtime_receipt_hash: str
    canary_receipt_hash: str
    actual_chunk_batch_replay_exact: bool
    maximum_sequence_length: int
    source_text_count: int
    source_texts_requiring_chunking: int
    source_chunk_count: int
    maximum_chunk_token_count: int
    truncated_chunk_count: int
    full_token_coverage: bool
    chunk_plan_commitment: str

    def safe_payload(self) -> dict[str, Any]:
        return {
            "version": SEMANTIC_BASELINE_VERSION,
            "prototype_hash": self.prototype_hash,
            "embedding_matrix_hash": self.embedding_matrix_hash,
            "chunk_embedding_matrix_hash": (
                self.chunk_embedding_matrix_hash
            ),
            "runtime_receipt_hash": self.runtime_receipt_hash,
            "canary_receipt_hash": self.canary_receipt_hash,
            "actual_chunk_batch_replay_exact": (
                self.actual_chunk_batch_replay_exact
            ),
            "maximum_sequence_length": self.maximum_sequence_length,
            "source_text_count": self.source_text_count,
            "source_texts_requiring_chunking": (
                self.source_texts_requiring_chunking
            ),
            "source_chunk_count": self.source_chunk_count,
            "maximum_chunk_token_count": (
                self.maximum_chunk_token_count
            ),
            "truncated_chunk_count": self.truncated_chunk_count,
            "full_token_coverage": self.full_token_coverage,
            "chunk_plan_commitment": self.chunk_plan_commitment,
            "predictions": [
                row.safe_payload() for row in self.predictions
            ],
            "pair_similarities": dict(
                sorted(self.pair_similarities.items())
            ),
        }


@dataclass(frozen=True)
class LegacyKeywordPrediction:
    item_key: str
    top_pattern_id: str | None
    predicted_law_id: str | None
    score: int | None
    score_decision: str | None
    gate_decision: str | None
    accepted: bool
    row_commitment: str

    def safe_payload(self) -> dict[str, Any]:
        return {
            "item_key": self.item_key,
            "top_pattern_id": self.top_pattern_id,
            "predicted_law_id": self.predicted_law_id,
            "score_microunits": self.score,
            "score_decision": self.score_decision,
            "gate_decision": self.gate_decision,
            "accepted": self.accepted,
            "row_commitment": self.row_commitment,
        }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _workspace_relative_path(path: Path) -> str | None:
    try:
        return path.relative_to(_WORKSPACE_ROOT.resolve()).as_posix()
    except ValueError:
        return None


def _has_symlink_component(path: Path) -> bool:
    workspace = _WORKSPACE_ROOT.resolve()
    try:
        relative = path.relative_to(workspace)
    except ValueError:
        return True
    cursor = workspace
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            return True
    return False


def build_baseline_implementation_closure(
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Content-address both baseline lanes and their live import origins.

    This is an implementation closure, not a model-performance receipt.  The
    expected paths name the internal dependency graph, while every digest is
    recomputed from the module that Python actually imported.
    """

    issues: list[str] = []
    module_rows: list[dict[str, Any]] = []
    module_by_name: dict[str, dict[str, Any]] = {}
    for module_name, expected_relative in sorted(
        _IMPLEMENTATION_MODULE_PATHS.items()
    ):
        expected = (_WORKSPACE_ROOT / expected_relative).resolve()
        row: dict[str, Any] = {
            "module_name": module_name,
            "expected_import_origin": expected_relative,
            "actual_import_origin": None,
            "implementation_sha256": None,
            "implementation_size": None,
            "loader_type": None,
        }
        try:
            module = import_module(module_name)
        except Exception:
            issues.append(
                "baseline_implementation_module_import_failed"
            )
            module_rows.append(row)
            module_by_name[module_name] = row
            continue
        spec = getattr(module, "__spec__", None)
        spec_origin = getattr(spec, "origin", None)
        module_file = getattr(module, "__file__", None)
        loader = getattr(spec, "loader", None)
        row["loader_type"] = (
            None if loader is None else type(loader).__name__
        )
        if not isinstance(spec_origin, str) or not isinstance(
            module_file, str
        ):
            issues.append(
                "baseline_implementation_import_origin_missing"
            )
            module_rows.append(row)
            module_by_name[module_name] = row
            continue
        try:
            origin_path = Path(spec_origin).resolve(strict=True)
            file_path = Path(module_file).resolve(strict=True)
        except OSError:
            issues.append(
                "baseline_implementation_import_origin_unreadable"
            )
            module_rows.append(row)
            module_by_name[module_name] = row
            continue
        relative_origin = _workspace_relative_path(origin_path)
        row["actual_import_origin"] = (
            relative_origin
            if relative_origin is not None
            else "outside_workspace"
        )
        if (
            origin_path != expected
            or file_path != expected
            or relative_origin != expected_relative
            or _has_symlink_component(Path(spec_origin).absolute())
            or _has_symlink_component(Path(module_file).absolute())
        ):
            issues.append(
                "baseline_implementation_import_origin_mismatch"
            )
        if not origin_path.is_file():
            issues.append(
                "baseline_implementation_origin_not_file"
            )
        else:
            raw = origin_path.read_bytes()
            row["implementation_sha256"] = hashlib.sha256(
                raw
            ).hexdigest()
            row["implementation_size"] = len(raw)
        module_rows.append(row)
        module_by_name[module_name] = row

    callable_contracts = (
        (
            "legacy_keyword",
            "score_structural_morphism",
            score_structural_morphism,
            "assumption_os.structural_patterns",
        ),
        (
            "legacy_keyword",
            "search_structural_patterns",
            search_structural_patterns,
            "assumption_os.structural_patterns",
        ),
        (
            "semantic_only",
            "OfflineMiniLMEncoder",
            OfflineMiniLMEncoder,
            "replication_runtime.qasper_minilm_v1.binding",
        ),
        (
            "semantic_only",
            "quantized_cosine_similarity",
            quantized_cosine_similarity,
            "replication_runtime.qasper_minilm_v1.binding",
        ),
    )
    callable_rows: list[dict[str, Any]] = []
    for lane, callable_name, value, expected_module in callable_contracts:
        actual_module = getattr(value, "__module__", None)
        source_file = inspect.getsourcefile(value)
        relative_source: str | None = None
        if isinstance(source_file, str):
            try:
                relative_source = _workspace_relative_path(
                    Path(source_file).resolve(strict=True)
                )
            except OSError:
                relative_source = None
        expected_source = _IMPLEMENTATION_MODULE_PATHS[
            expected_module
        ]
        if (
            actual_module != expected_module
            or relative_source != expected_source
        ):
            issues.append(
                "baseline_implementation_callable_origin_mismatch"
            )
        callable_rows.append(
            {
                "lane": lane,
                "callable_name": callable_name,
                "module_name": actual_module,
                "actual_import_origin": (
                    relative_source
                    if relative_source is not None
                    else "outside_workspace_or_unavailable"
                ),
            }
        )

    lane_closures: dict[str, dict[str, Any]] = {}
    for lane, module_names in sorted(
        _IMPLEMENTATION_LANE_MODULES.items()
    ):
        lane_body = {
            "module_names": list(module_names),
            "module_commitments": [
                strict_content_hash(module_by_name[module_name])
                for module_name in module_names
            ],
            "callable_commitments": [
                strict_content_hash(row)
                for row in callable_rows
                if row["lane"] == lane
            ],
        }
        lane_closures[lane] = {
            "module_count": len(module_names),
            "callable_count": sum(
                row["lane"] == lane for row in callable_rows
            ),
            "closure_hash": strict_content_hash(lane_body),
        }

    unique_issues = tuple(sorted(set(issues)))
    closure: dict[str, Any] = {
        "version": _IMPLEMENTATION_CLOSURE_VERSION,
        "verified": not unique_issues,
        "module_count": len(module_rows),
        "modules": module_rows,
        "callable_count": len(callable_rows),
        "callables": callable_rows,
        "lanes": lane_closures,
        "issue_count": len(unique_issues),
        "issue_commitment": strict_content_hash(
            list(unique_issues)
        ),
    }
    closure["closure_hash"] = strict_content_hash(closure)
    return closure, unique_issues


@lru_cache(maxsize=1)
def _default_encoder() -> OfflineMiniLMEncoder:
    root = _project_root()
    return OfflineMiniLMEncoder(
        asset_manifest_path=(
            root / "manifests/qasper_minilm_runtime_asset_v1.json"
        ),
        model_root=(
            root / "artifacts/qasper_minilm_runtime_v1/model"
        ),
        run_canary=True,
    )


def build_semantic_law_prototypes(
    registry: GSCLSchemaRegistry,
    ontology: UniversalAssumptionOntology,
) -> tuple[tuple[str, str], ...]:
    templates = {
        template.template_id: template
        for template in ontology.templates
    }
    rows = []
    for schema in sorted(
        registry.schemas, key=lambda row: row.law_id
    ):
        template = templates[schema.ontology_template_id]
        rows.append(
            (
                schema.law_id,
                f"{schema.law_kind.value}. {template.claim_schema}",
            )
        )
    return tuple(rows)


def _untruncated_token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        verbose=False,
    )
    input_ids = encoded.get("input_ids")
    if (
        not isinstance(input_ids, list)
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            for value in input_ids
        )
    ):
        raise RuntimeError(
            "semantic-only token coverage audit returned invalid ids"
        )
    return len(input_ids)


def _record_line_chunks(
    text: str, tokenizer: Any
) -> tuple[tuple[str, int], ...]:
    """Greedily pack complete record lines without truncation."""

    lines = tuple(text.splitlines(keepends=True))
    if not lines or "".join(lines) != text:
        raise ValueError(
            "semantic-only source is not complete record-line text"
        )
    chunks: list[tuple[str, int]] = []
    current = ""
    current_count = 0
    for line in lines:
        if not line.strip():
            raise ValueError(
                "semantic-only source contains an empty record line"
            )
        line_count = _untruncated_token_count(tokenizer, line)
        if line_count > MAXIMUM_SEQUENCE_LENGTH:
            raise ValueError(
                "semantic-only record line exceeds frozen token bound"
            )
        candidate = current + line
        candidate_count = _untruncated_token_count(
            tokenizer, candidate
        )
        if current and candidate_count > MAXIMUM_SEQUENCE_LENGTH:
            chunks.append((current, current_count))
            current = line
            current_count = line_count
        else:
            current = candidate
            current_count = candidate_count
    if current:
        chunks.append((current, current_count))
    if (
        not chunks
        or "".join(chunk for chunk, _ in chunks) != text
        or any(
            not 1 <= token_count <= MAXIMUM_SEQUENCE_LENGTH
            for _, token_count in chunks
        )
    ):
        raise RuntimeError(
            "semantic-only record-line coverage contract failed"
        )
    return tuple(chunks)


def _normalized_mean_pool(matrix: np.ndarray) -> np.ndarray:
    if (
        matrix.ndim != 2
        or matrix.shape[0] < 1
        or matrix.shape[1] != EMBEDDING_DIMENSION
        or not np.isfinite(matrix).all()
    ):
        raise RuntimeError(
            "semantic-only chunk matrix is invalid for pooling"
        )
    mean = np.mean(matrix.astype(np.float64), axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError(
            "semantic-only chunk mean cannot be normalized"
        )
    pooled = np.asarray(mean / norm, dtype=np.float32)
    pooled_norm = float(
        np.linalg.norm(pooled.astype(np.float64))
    )
    if not np.isfinite(pooled_norm) or pooled_norm <= 0.0:
        raise RuntimeError(
            "semantic-only pooled vector is invalid"
        )
    return np.asarray(pooled / pooled_norm, dtype=np.float32)


def run_semantic_only_batch(
    *,
    item_texts: Mapping[str, str],
    pair_keys: Sequence[tuple[str, str, str]],
    registry: GSCLSchemaRegistry,
    ontology: UniversalAssumptionOntology,
    encoder: OfflineMiniLMEncoder | None = None,
) -> SemanticOnlyBatch:
    """Replay the actual batch exactly and return untuned prototypes."""

    for item_key, text in item_texts.items():
        if not isinstance(item_key, str) or not isinstance(text, str):
            raise TypeError(
                "semantic-only inputs must map text keys to text values"
            )
        try:
            if text.encode("utf-8").decode("utf-8") != text:
                raise UnicodeError
        except UnicodeError as exc:
            raise ValueError(
                "semantic-only input is not lossless UTF-8 text"
            ) from exc
    prototypes = build_semantic_law_prototypes(registry, ontology)
    ordered_items = tuple(sorted(item_texts))
    actual_encoder = encoder or _default_encoder()
    tokenizer = getattr(
        getattr(actual_encoder, "_model", None), "tokenizer", None
    )
    if tokenizer is None:
        raise RuntimeError(
            "semantic-only tokenizer is unavailable for coverage audit"
        )
    prototype_texts = tuple(text for _, text in prototypes)
    prototype_token_counts = tuple(
        _untruncated_token_count(tokenizer, text)
        for text in prototype_texts
    )
    if any(
        count > MAXIMUM_SEQUENCE_LENGTH
        for count in prototype_token_counts
    ):
        raise RuntimeError(
            "semantic-only prototype exceeds frozen token bound"
        )

    chunk_texts: list[str] = []
    chunk_token_counts: list[int] = []
    item_chunk_slices: dict[str, tuple[int, int]] = {}
    chunk_plan_rows = []
    source_texts_requiring_chunking = 0
    for item_key in ordered_items:
        text = item_texts[item_key]
        chunks = _record_line_chunks(text, tokenizer)
        source_texts_requiring_chunking += len(chunks) > 1
        start = len(chunk_texts)
        chunk_texts.extend(chunk for chunk, _ in chunks)
        chunk_token_counts.extend(count for _, count in chunks)
        end = len(chunk_texts)
        item_chunk_slices[item_key] = (start, end)
        chunk_plan_rows.append(
            {
                "item_key": item_key,
                "source_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
                "source_size": len(text.encode("utf-8")),
                "chunk_count": len(chunks),
                "chunk_rows": [
                    {
                        "chunk_sha256": hashlib.sha256(
                            chunk.encode("utf-8")
                        ).hexdigest(),
                        "chunk_size": len(chunk.encode("utf-8")),
                        "token_count": token_count,
                    }
                    for chunk, token_count in chunks
                ],
                "complete_character_reconstruction": (
                    "".join(chunk for chunk, _ in chunks) == text
                ),
            }
        )
    texts = (*prototype_texts, *chunk_texts)
    matrices = (
        actual_encoder.encode(texts),
        actual_encoder.encode(texts),
    )
    if not np.array_equal(matrices[0], matrices[1]):
        raise RuntimeError(
            "semantic-only actual batch replay is not byte exact"
        )
    chunk_matrix = matrices[0]
    prototype_count = len(prototypes)
    predictions = []
    vectors: dict[str, np.ndarray] = {}
    for item_key in ordered_items:
        start, end = item_chunk_slices[item_key]
        vector = _normalized_mean_pool(
            chunk_matrix[
                prototype_count + start : prototype_count + end
            ]
        )
        vectors[item_key] = vector
        scores = {
            law_id: quantized_cosine_similarity(
                vector, chunk_matrix[prototype_index]
            )
            for prototype_index, (law_id, _) in enumerate(prototypes)
        }
        predicted_law_id, top_score = min(
            scores.items(), key=lambda row: (-row[1], row[0])
        )
        predictions.append(
            SemanticOnlyPrediction(
                item_key=item_key,
                predicted_law_id=predicted_law_id,
                top_score=top_score,
                score_commitment=strict_content_hash(
                    dict(sorted(scores.items()))
                ),
            )
        )
    pair_similarities = {}
    for pair_key, left_key, right_key in pair_keys:
        if left_key not in vectors or right_key not in vectors:
            raise KeyError("semantic pair references an unknown item")
        pair_similarities[pair_key] = quantized_cosine_similarity(
            vectors[left_key], vectors[right_key]
        )
    runtime_receipt = actual_encoder.runtime_receipt
    canary_receipt = actual_encoder.canary_receipt
    pooled_matrix = np.vstack(
        [
            chunk_matrix[:prototype_count],
            *(vectors[item_key] for item_key in ordered_items),
        ]
    ).astype(np.float32, copy=False)
    full_token_coverage = all(
        row["complete_character_reconstruction"] is True
        for row in chunk_plan_rows
    ) and all(
        1 <= count <= MAXIMUM_SEQUENCE_LENGTH
        for count in chunk_token_counts
    )
    if not full_token_coverage:
        raise RuntimeError(
            "semantic-only source chunking lost input coverage"
        )
    return SemanticOnlyBatch(
        predictions=tuple(predictions),
        pair_similarities=pair_similarities,
        prototype_hash=strict_content_hash(
            [
                {"law_id": law_id, "text": text}
                for law_id, text in prototypes
            ]
        ),
        embedding_matrix_hash=hashlib.sha256(
            pooled_matrix.astype("<f4", copy=False).tobytes(order="C")
        ).hexdigest(),
        chunk_embedding_matrix_hash=hashlib.sha256(
            chunk_matrix.astype("<f4", copy=False).tobytes(order="C")
        ).hexdigest(),
        runtime_receipt_hash=strict_content_hash(runtime_receipt),
        canary_receipt_hash=strict_content_hash(canary_receipt),
        actual_chunk_batch_replay_exact=True,
        maximum_sequence_length=MAXIMUM_SEQUENCE_LENGTH,
        source_text_count=len(ordered_items),
        source_texts_requiring_chunking=(
            source_texts_requiring_chunking
        ),
        source_chunk_count=len(chunk_texts),
        maximum_chunk_token_count=max(chunk_token_counts),
        truncated_chunk_count=0,
        full_token_coverage=full_token_coverage,
        chunk_plan_commitment=strict_content_hash(
            {
                "prototype_token_counts": list(
                    prototype_token_counts
                ),
                "source_chunk_plan": chunk_plan_rows,
            }
        ),
    )


def _legacy_registry_hash() -> str:
    return hashlib.sha256(
        json.dumps(
            DEFAULT_STRUCTURAL_PATTERNS,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def verify_legacy_baseline_contract() -> tuple[str, ...]:
    issues: list[str] = []
    source_path = (
        Path(__file__).resolve().parents[2]
        / "assumption_os/structural_patterns.py"
    )
    if (
        not source_path.is_file()
        or hashlib.sha256(source_path.read_bytes()).hexdigest()
        != LEGACY_SOURCE_SHA256
    ):
        issues.append("legacy_source_hash_mismatch")
    if (
        len(DEFAULT_STRUCTURAL_PATTERNS) != 10
        or _legacy_registry_hash() != LEGACY_REGISTRY_HASH
    ):
        issues.append("legacy_registry_hash_mismatch")
    return tuple(sorted(set(issues)))


def run_legacy_keyword_baseline(
    item_texts: Mapping[str, str],
    *,
    execution_audit: MutableMapping[str, int] | None = None,
) -> tuple[LegacyKeywordPrediction, ...]:
    contract_issues = verify_legacy_baseline_contract()
    if contract_issues:
        raise PermissionError(
            "legacy baseline contract invalid: "
            + ",".join(contract_issues)
        )
    predictions = []
    for item_key, text in sorted(item_texts.items()):
        if not isinstance(item_key, str) or not isinstance(text, str):
            raise TypeError(
                "legacy inputs must map text keys to text values"
            )
        try:
            if text.encode("utf-8").decode("utf-8") != text:
                raise UnicodeError
        except UnicodeError as exc:
            raise ValueError(
                "legacy input is not lossless UTF-8 text"
            ) from exc
        row_payload = _legacy_predict_text(
            text, execution_audit=execution_audit
        )
        payload = {"item_key": item_key, **row_payload}
        predictions.append(
            LegacyKeywordPrediction(
                item_key=item_key,
                top_pattern_id=payload["top_pattern_id"],
                predicted_law_id=payload["predicted_law_id"],
                score=payload["score_microunits"],
                score_decision=payload["score_decision"],
                gate_decision=payload["gate_decision"],
                accepted=payload["accepted"],
                row_commitment=strict_content_hash(payload),
            )
        )
    return tuple(predictions)


def _legacy_predict_text(
    text: str,
    *,
    execution_audit: MutableMapping[str, int] | None = None,
) -> dict[str, Any]:
    # This count lives inside the uncached compute path.  Qualification uses
    # it to prove that replay 2 actually re-executed all legacy predictions.
    if execution_audit is not None:
        previous = execution_audit.get(
            "prediction_compute_calls", 0
        )
        if not isinstance(previous, int) or isinstance(previous, bool):
            raise TypeError(
                "legacy execution audit counter must be an integer"
            )
        execution_audit["prediction_compute_calls"] = previous + 1
    rows = search_structural_patterns(
        None,
        text,
        top_n=1,
        min_score=LEGACY_MIN_SCORE,
        include_defaults=True,
    )
    if not rows:
        return {
            "top_pattern_id": None,
            "predicted_law_id": None,
            "score_microunits": None,
            "score_decision": None,
            "gate_decision": None,
            "accepted": False,
        }
    row = rows[0]
    candidate = row.get("candidate")
    gate = score_structural_morphism(
        candidate if isinstance(candidate, dict) else {}
    )
    metrics = row.get("metrics")
    score_value = (
        metrics.get("score")
        if isinstance(metrics, dict)
        else row.get("score")
    )
    score_microunits = (
        int(round(float(score_value) * 1_000_000))
        if isinstance(score_value, (int, float))
        and not isinstance(score_value, bool)
        else None
    )
    pattern_id = row.get("pattern_id")
    predicted_law_id = (
        _LEGACY_LAW_MAP.get(pattern_id)
        if isinstance(pattern_id, str)
        else None
    )
    score_decision = (
        metrics.get("decision")
        if isinstance(metrics, dict)
        else None
    )
    gate_decision = gate.get("decision")
    accepted = (
        predicted_law_id is not None
        and score_decision == "allow"
        and gate_decision == "allow"
        and gate.get("blocks_policy_update") is False
    )
    return {
        "top_pattern_id": pattern_id,
        "predicted_law_id": predicted_law_id,
        "score_microunits": score_microunits,
        "score_decision": score_decision,
        "gate_decision": gate_decision,
        "accepted": accepted,
    }


__all__ = [
    "BASELINE_CONTRACT",
    "BASELINE_CONTRACT_HASH",
    "LEGACY_BASELINE_VERSION",
    "LEGACY_MIN_SCORE",
    "LEGACY_REGISTRY_HASH",
    "LEGACY_SOURCE_SHA256",
    "LegacyKeywordPrediction",
    "SEMANTIC_BASELINE_VERSION",
    "SemanticOnlyBatch",
    "SemanticOnlyPrediction",
    "build_baseline_implementation_closure",
    "build_semantic_law_prototypes",
    "run_legacy_keyword_baseline",
    "run_semantic_only_batch",
    "verify_legacy_baseline_contract",
]
