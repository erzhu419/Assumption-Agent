"""Frozen, source-free scorers for the ARN intrinsic four-arm comparison.

The module contains no benchmark reader and no label access.  A caller first
constructs :class:`FrozenNarrativeScorers` from already-separated narrative
extractions.  The resulting object exposes the three callables required by
``gscl_arn_intrinsic_arms_v1``:

* lossless full-text MiniLM similarity for ``semantic_only``;
* the unchanged ten-pattern legacy keyword morphism score vector; and
* MiniLM similarities between grounded object and generator-anchor quotes for
  proposal formation.

Semantic scores never enter the structural checker.  Long text is split into
complete Unicode substrings whose concatenation is byte-exact; no tokenizer
truncation is permitted.  Construction replays the complete embedding batch
twice and fails closed unless the float32 matrices are byte-identical.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import hashlib
import json
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

# The frozen legacy implementation remains at the workspace root.  Resolve it
# from this module's origin rather than trusting the caller's working directory.
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from assumption_os.structural_patterns import (
    DEFAULT_STRUCTURAL_PATTERNS,
    search_structural_patterns,
)
from replication_runtime.gscl_minilm_portable_v1.binding import (
    GSCLPortableOfflineMiniLMEncoder,
)
from replication_runtime.qasper_minilm_v1.binding import (
    EMBEDDING_DIMENSION,
    MAXIMUM_SEQUENCE_LENGTH,
    MAXIMUM_TEXTS_PER_CALL,
    quantized_cosine_similarity,
)

from .gscl_narrative_correspondence_v1 import (
    NarrativeExtraction,
    SemanticScoreTable,
)


SCORER_VERSION = "gscl.arn.intrinsic.scorers.v1"
MAXIMUM_PRIMED_EXTRACTIONS = 4_096
MAXIMUM_CHUNKS = 65_536
MAXIMUM_CHUNK_START_CHARACTERS = 1_024
LEGACY_SCORE_SCALE = 1_000_000


class IntrinsicScorerError(RuntimeError):
    """Stable fail-closed scorer error without private narrative text."""

    def __init__(self, issue_id: str) -> None:
        self.issue_id = issue_id
        super().__init__(issue_id)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _content_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _legacy_registry_rows() -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for raw in DEFAULT_STRUCTURAL_PATTERNS:
        if not isinstance(raw, dict):
            raise IntrinsicScorerError("legacy_registry_row_invalid")
        pattern_id = raw.get("pattern_id")
        if not isinstance(pattern_id, str) or not pattern_id:
            raise IntrinsicScorerError("legacy_pattern_id_invalid")
        rows.append(dict(raw))
    if len(rows) != 10 or len({row["pattern_id"] for row in rows}) != 10:
        raise IntrinsicScorerError("legacy_registry_topology_invalid")
    return tuple(sorted(rows, key=lambda row: str(row["pattern_id"])))


_LEGACY_REGISTRY_ROWS = _legacy_registry_rows()
LEGACY_FEATURE_IDS = tuple(
    str(row["pattern_id"]) for row in _LEGACY_REGISTRY_ROWS
)
LEGACY_REGISTRY_SHA256 = hashlib.sha256(
    _canonical_bytes(list(_LEGACY_REGISTRY_ROWS))
).hexdigest()

SCORER_CONTRACT = {
    "version": SCORER_VERSION,
    "benchmark_reader_present": False,
    "label_access_present": False,
    "semantic_only": {
        "input": "lossless_full_narrative_utf8",
        "encoder": (
            "gscl_target_local_qasper_minilm_portable_v2_"
            "cpu_float32_offline"
        ),
        "chunking": "unicode_substring_complete_no_truncation",
        "aggregation": "float64_mean_l2_normalized_float32",
        "score": "integer_quantized_cosine",
    },
    "legacy_keyword": {
        "registry_sha256": LEGACY_REGISTRY_SHA256,
        "feature_ids": list(LEGACY_FEATURE_IDS),
        "source": "unchanged_default_structural_patterns",
        "minimum_pattern_score": 0,
        "score_scale": LEGACY_SCORE_SCALE,
    },
    "structural_proposal": {
        "object_score": "grounded_quote_minilm_cosine",
        "generator_score": "grounded_anchor_quote_minilm_cosine",
        "checker_receives_semantic_score": False,
    },
    "actual_batch_replay_count": 2,
}
SCORER_CONTRACT_HASH = _content_hash(SCORER_CONTRACT)
_FORMAL_CONSTRUCTION_DOMAIN = (
    "formal_exact_gscl_target_local_portable_minilm_v1"
)
_QUALIFICATION_CONSTRUCTION_DOMAIN = "qualification_injected_encoder"
_SCORER_FACTORY_MARKER = object()
_SCORER_RECEIPT_KEYS = frozenset(
    {
        "actual_batch_replay_exact",
        "benchmark_source_accessed",
        "canary_receipt_commitment",
        "chunk_count",
        "chunk_plan_commitment",
        "construction_domain",
        "contract_hash",
        "embedding_float32_commitment",
        "extraction_count",
        "extraction_set_commitment",
        "labels_accessed",
        "legacy_registry_sha256",
        "mention_count",
        "mention_vector_commitment",
        "runtime_receipt_commitment",
        "self_hash",
        "source_count",
        "source_vector_commitment",
        "version",
    }
)


def _sealed_vector_mapping(
    value: Mapping[str, np.ndarray], *, issue_prefix: str
) -> Mapping[str, np.ndarray]:
    if not isinstance(value, Mapping) or not value:
        raise IntrinsicScorerError(f"{issue_prefix}_vectors_invalid")
    sealed: dict[str, np.ndarray] = {}
    for key, raw_vector in sorted(value.items()):
        if (
            not isinstance(key, str)
            or len(key) != 64
            or any(character not in "0123456789abcdef" for character in key)
        ):
            raise IntrinsicScorerError(
                f"{issue_prefix}_vector_key_invalid"
            )
        vector = np.asarray(raw_vector, dtype="<f4")
        if (
            vector.shape != (EMBEDDING_DIMENSION,)
            or not np.isfinite(vector).all()
        ):
            raise IntrinsicScorerError(
                f"{issue_prefix}_vector_value_invalid"
            )
        # A bytes-backed ndarray cannot be made writeable again.  This avoids
        # the shallow-frozen-dataclass pitfall where a caller mutates vectors
        # after the receipt has been formed.
        payload = vector.astype("<f4", copy=False).tobytes(order="C")
        immutable = np.frombuffer(payload, dtype="<f4")
        if immutable.flags.writeable:
            raise IntrinsicScorerError(
                f"{issue_prefix}_vector_mutable"
            )
        sealed[key] = immutable
    return MappingProxyType(sealed)


def _vector_mapping_commitment(
    value: Mapping[str, np.ndarray],
) -> str:
    return _content_hash(
        [
            {
                "content_sha256": key,
                "float32_sha256": hashlib.sha256(
                    vector.astype("<f4", copy=False).tobytes(order="C")
                ).hexdigest(),
            }
            for key, vector in sorted(value.items())
        ]
    )


def _token_count(tokenizer: object, text: str) -> int:
    if not isinstance(text, str) or not text:
        raise IntrinsicScorerError("chunk_text_invalid")
    try:
        encoded = tokenizer(  # type: ignore[operator]
            text,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            verbose=False,
        )
    except Exception as exc:
        raise IntrinsicScorerError("tokenizer_failed") from exc
    if not isinstance(encoded, Mapping):
        raise IntrinsicScorerError("tokenizer_result_invalid")
    input_ids = encoded.get("input_ids")
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or any(
            not isinstance(item, int) or isinstance(item, bool)
            for item in input_ids
        )
    ):
        raise IntrinsicScorerError("tokenizer_result_invalid")
    return len(input_ids)


def lossless_token_chunks(
    text: str, tokenizer: object
) -> tuple[tuple[str, int], ...]:
    """Split into byte-exact Unicode substrings without model truncation."""

    if (
        not isinstance(text, str)
        or not text
        or "\x00" in text
    ):
        raise IntrinsicScorerError("source_text_invalid")
    try:
        source_bytes = text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise IntrinsicScorerError("source_utf8_invalid") from exc
    chunks: list[tuple[str, int]] = []
    offset = 0
    while offset < len(text):
        width = min(
            MAXIMUM_CHUNK_START_CHARACTERS, len(text) - offset
        )
        accepted: tuple[str, int] | None = None
        while width >= 1:
            candidate = text[offset : offset + width]
            count = _token_count(tokenizer, candidate)
            if count <= MAXIMUM_SEQUENCE_LENGTH:
                accepted = (candidate, count)
                break
            width //= 2
        if accepted is None:
            raise IntrinsicScorerError("single_character_token_overflow")
        chunks.append(accepted)
        if len(chunks) > MAXIMUM_CHUNKS:
            raise IntrinsicScorerError("chunk_count_exceeded")
        offset += len(accepted[0])
    if (
        not chunks
        or "".join(chunk for chunk, _ in chunks) != text
        or b"".join(
            chunk.encode("utf-8") for chunk, _ in chunks
        )
        != source_bytes
        or any(
            not 1 <= count <= MAXIMUM_SEQUENCE_LENGTH
            for _, count in chunks
        )
    ):
        raise IntrinsicScorerError("lossless_chunk_contract_failed")
    return tuple(chunks)


def _normalize_mean(matrix: np.ndarray) -> np.ndarray:
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.ndim != 2
        or matrix.shape[0] < 1
        or matrix.shape[1] != EMBEDDING_DIMENSION
        or not np.isfinite(matrix).all()
    ):
        raise IntrinsicScorerError("embedding_chunk_matrix_invalid")
    mean = np.mean(matrix.astype(np.float64), axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm <= 0.0:
        raise IntrinsicScorerError("embedding_mean_invalid")
    vector = np.asarray(mean / norm, dtype=np.float32)
    second_norm = float(np.linalg.norm(vector.astype(np.float64)))
    if not np.isfinite(second_norm) or second_norm <= 0.0:
        raise IntrinsicScorerError("embedding_normalization_invalid")
    return np.asarray(vector / second_norm, dtype=np.float32)


def _encoder_tokenizer(encoder: object) -> object:
    tokenizer = getattr(
        getattr(encoder, "_model", None), "tokenizer", None
    )
    if tokenizer is None:
        tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is None or not callable(tokenizer):
        raise IntrinsicScorerError("encoder_tokenizer_unavailable")
    return tokenizer


def _source_key(raw: bytes) -> str:
    if not isinstance(raw, bytes) or not raw:
        raise IntrinsicScorerError("raw_text_invalid")
    try:
        if raw.decode("utf-8", errors="strict").encode("utf-8") != raw:
            raise UnicodeError
    except UnicodeError as exc:
        raise IntrinsicScorerError("raw_text_utf8_invalid") from exc
    return hashlib.sha256(raw).hexdigest()


def _validate_scorer_binding(
    *,
    source_vectors: Mapping[str, np.ndarray],
    mention_vectors: Mapping[str, np.ndarray],
    primed_extraction_hashes: frozenset[str],
    receipt_value: Mapping[str, object],
    require_immutable_storage: bool,
    inspect_vector_storage: bool,
    recompute_extraction_commitment: bool,
    recompute_vector_commitments: bool,
) -> tuple[dict[str, object], str]:
    """Validate already-sealed state without copying or re-sealing arrays."""

    if require_immutable_storage and (
        type(source_vectors) is not MappingProxyType
        or type(mention_vectors) is not MappingProxyType
        or type(receipt_value) is not MappingProxyType
    ):
        raise IntrinsicScorerError("scorer_storage_not_immutable")
    for issue_prefix, values in (
        ("source", source_vectors),
        ("mention", mention_vectors),
    ):
        if not isinstance(values, Mapping) or not values:
            raise IntrinsicScorerError(
                f"{issue_prefix}_vectors_invalid"
            )
        if not inspect_vector_storage:
            continue
        for key, vector in values.items():
            if (
                not isinstance(key, str)
                or len(key) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in key
                )
                or not isinstance(vector, np.ndarray)
                or vector.dtype != np.dtype("<f4")
                or vector.shape != (EMBEDDING_DIMENSION,)
                or not vector.flags.c_contiguous
                or not np.isfinite(vector).all()
            ):
                raise IntrinsicScorerError(
                    f"{issue_prefix}_vector_value_invalid"
                )
            if require_immutable_storage and (
                vector.flags.writeable
                or not isinstance(vector.base, bytes)
            ):
                raise IntrinsicScorerError(
                    f"{issue_prefix}_vector_mutable"
                )
    if (
        not isinstance(primed_extraction_hashes, frozenset)
        or not primed_extraction_hashes
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(
                character not in "0123456789abcdef"
                for character in value
            )
            for value in primed_extraction_hashes
        )
    ):
        raise IntrinsicScorerError(
            "primed_extraction_hashes_invalid"
        )
    if not isinstance(receipt_value, Mapping):
        raise IntrinsicScorerError("scorer_receipt_invalid")
    receipt = dict(receipt_value)
    if set(receipt) != _SCORER_RECEIPT_KEYS:
        raise IntrinsicScorerError("scorer_receipt_fields_invalid")
    self_hash = receipt.pop("self_hash")
    if (
        not isinstance(self_hash, str)
        or len(self_hash) != 64
        or _content_hash(receipt) != self_hash
        or receipt.get("version") != SCORER_VERSION
        or receipt.get("contract_hash") != SCORER_CONTRACT_HASH
        or receipt.get("construction_domain")
        not in {
            _FORMAL_CONSTRUCTION_DOMAIN,
            _QUALIFICATION_CONSTRUCTION_DOMAIN,
        }
        or receipt.get("extraction_count")
        != len(primed_extraction_hashes)
        or (
            recompute_extraction_commitment
            and receipt.get("extraction_set_commitment")
            != _content_hash(sorted(primed_extraction_hashes))
        )
        or receipt.get("source_count") != len(source_vectors)
        or receipt.get("mention_count") != len(mention_vectors)
        or (
            recompute_vector_commitments
            and receipt.get("source_vector_commitment")
            != _vector_mapping_commitment(source_vectors)
        )
        or (
            recompute_vector_commitments
            and receipt.get("mention_vector_commitment")
            != _vector_mapping_commitment(mention_vectors)
        )
        or receipt.get("actual_batch_replay_exact") is not True
        or receipt.get("legacy_registry_sha256")
        != LEGACY_REGISTRY_SHA256
        or receipt.get("benchmark_source_accessed") is not False
        or receipt.get("labels_accessed") is not False
    ):
        raise IntrinsicScorerError("scorer_receipt_binding_invalid")
    for field_name in (
        "canary_receipt_commitment",
        "chunk_plan_commitment",
        "embedding_float32_commitment",
        "mention_vector_commitment",
        "runtime_receipt_commitment",
        "source_vector_commitment",
    ):
        value = receipt.get(field_name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(
                character not in "0123456789abcdef"
                for character in value
            )
        ):
            raise IntrinsicScorerError(
                "scorer_receipt_hash_invalid"
            )
    for field_name in (
        "chunk_count",
        "extraction_count",
        "mention_count",
        "source_count",
    ):
        value = receipt.get(field_name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
        ):
            raise IntrinsicScorerError(
                "scorer_receipt_count_invalid"
            )
    return receipt, self_hash


@dataclass(frozen=True, slots=True)
class FrozenNarrativeScorers:
    """Precomputed, replay-checked scorer callables for one sealed input pack."""

    source_vectors: Mapping[str, np.ndarray]
    mention_vectors: Mapping[str, np.ndarray]
    primed_extraction_hashes: frozenset[str]
    receipt: Mapping[str, object]
    _construction_token: InitVar[object]
    _authority_marker: object = field(
        init=False, repr=False, compare=False
    )
    _storage_identity: tuple[int, int, int, int] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self, _construction_token: object) -> None:
        if type(self) is not FrozenNarrativeScorers:
            raise IntrinsicScorerError("scorer_subclass_forbidden")
        if _construction_token is not _SCORER_FACTORY_MARKER:
            raise IntrinsicScorerError(
                "scorer_construction_not_authorized"
            )
        sealed_sources = _sealed_vector_mapping(
            self.source_vectors, issue_prefix="source"
        )
        sealed_mentions = _sealed_vector_mapping(
            self.mention_vectors, issue_prefix="mention"
        )
        object.__setattr__(self, "source_vectors", sealed_sources)
        object.__setattr__(self, "mention_vectors", sealed_mentions)
        receipt, self_hash = _validate_scorer_binding(
            source_vectors=sealed_sources,
            mention_vectors=sealed_mentions,
            primed_extraction_hashes=self.primed_extraction_hashes,
            receipt_value=self.receipt,
            require_immutable_storage=False,
            inspect_vector_storage=True,
            recompute_extraction_commitment=True,
            recompute_vector_commitments=True,
        )
        object.__setattr__(
            self,
            "receipt",
            MappingProxyType({**receipt, "self_hash": self_hash}),
        )
        object.__setattr__(
            self, "_authority_marker", _SCORER_FACTORY_MARKER
        )
        object.__setattr__(
            self,
            "_storage_identity",
            (
                id(self.source_vectors),
                id(self.mention_vectors),
                id(self.primed_extraction_hashes),
                id(self.receipt),
            ),
        )

    def validate_internal(self) -> None:
        if (
            type(self) is not FrozenNarrativeScorers
            or getattr(self, "_authority_marker", None)
            is not _SCORER_FACTORY_MARKER
            or getattr(self, "_storage_identity", None)
            != (
                id(self.source_vectors),
                id(self.mention_vectors),
                id(self.primed_extraction_hashes),
                id(self.receipt),
            )
        ):
            raise IntrinsicScorerError(
                "scorer_construction_not_authorized"
            )
        _validate_scorer_binding(
            source_vectors=self.source_vectors,
            mention_vectors=self.mention_vectors,
            primed_extraction_hashes=self.primed_extraction_hashes,
            receipt_value=self.receipt,
            require_immutable_storage=True,
            # The only references to the backing dictionaries were discarded
            # after wrapping them in MappingProxyType.  Their exact identities,
            # the frozenset identity, and the bytes-backed vectors were sealed
            # during __post_init__, so per-item validation need not rescan them.
            inspect_vector_storage=False,
            recompute_extraction_commitment=False,
            # The constructor already formed these commitments while copying
            # every vector into an immutable bytes-backed ndarray.  Re-hashing
            # the complete vector table for every item is quadratic in the
            # formal cohort size and adds no integrity once the MappingProxy
            # and bytes-backed storage checks above have passed.
            recompute_vector_commitments=False,
        )

    @classmethod
    def build(
        cls,
        extractions: Sequence[NarrativeExtraction],
        *,
        encoder: object,
    ) -> "FrozenNarrativeScorers":
        if cls is not FrozenNarrativeScorers:
            raise IntrinsicScorerError("scorer_subclass_forbidden")
        rows = tuple(extractions)
        if (
            not 1 <= len(rows) <= MAXIMUM_PRIMED_EXTRACTIONS
            or any(not isinstance(row, NarrativeExtraction) for row in rows)
        ):
            raise IntrinsicScorerError("primed_extractions_invalid")
        for row in rows:
            row.__post_init__()
        is_formal_encoder = (
            type(encoder) is GSCLPortableOfflineMiniLMEncoder
        )
        if is_formal_encoder:
            try:
                encoder.validate_internal()
            except Exception as exc:
                raise IntrinsicScorerError(
                    "formal_encoder_binding_invalid"
                ) from exc
        extraction_hashes = tuple(
            sorted({row.extraction_hash for row in rows})
        )
        source_texts: dict[str, str] = {}
        mention_texts: dict[str, str] = {}
        for row in rows:
            source_texts[row.source.source_sha256] = row.source.text
            for mention in row.mentions:
                existing = mention_texts.get(mention.quote_sha256)
                if existing is not None and existing != mention.quote:
                    raise IntrinsicScorerError(
                        "quote_hash_collision_detected"
                    )
                mention_texts[mention.quote_sha256] = mention.quote

        tokenizer = _encoder_tokenizer(encoder)
        ordered_entries = tuple(
            ("source", key, value)
            for key, value in sorted(source_texts.items())
        ) + tuple(
            ("mention", key, value)
            for key, value in sorted(mention_texts.items())
        )
        chunk_texts: list[str] = []
        chunk_slices: dict[tuple[str, str], tuple[int, int]] = {}
        chunk_plan: list[dict[str, object]] = []
        for kind, key, text in ordered_entries:
            chunks = lossless_token_chunks(text, tokenizer)
            start = len(chunk_texts)
            chunk_texts.extend(chunk for chunk, _ in chunks)
            end = len(chunk_texts)
            chunk_slices[(kind, key)] = (start, end)
            chunk_plan.append(
                {
                    "kind": kind,
                    "content_sha256": key,
                    "utf8_size": len(text.encode("utf-8")),
                    "chunk_count": len(chunks),
                    "chunks": [
                        {
                            "sha256": hashlib.sha256(
                                chunk.encode("utf-8")
                            ).hexdigest(),
                            "utf8_size": len(chunk.encode("utf-8")),
                            "token_count": count,
                        }
                        for chunk, count in chunks
                    ],
                }
            )
        if not chunk_texts or len(chunk_texts) > MAXIMUM_CHUNKS:
            raise IntrinsicScorerError("embedding_chunk_count_invalid")
        def encode_all() -> np.ndarray:
            batches: list[np.ndarray] = []
            for start in range(0, len(chunk_texts), MAXIMUM_TEXTS_PER_CALL):
                batch = tuple(
                    chunk_texts[start : start + MAXIMUM_TEXTS_PER_CALL]
                )
                batches.append(
                    np.asarray(encoder.encode(batch), dtype=np.float32)
                )
            return np.vstack(batches).astype(np.float32, copy=False)

        try:
            matrices = (encode_all(), encode_all())
        except Exception as exc:
            raise IntrinsicScorerError("encoder_batch_failed") from exc
        expected_shape = (len(chunk_texts), EMBEDDING_DIMENSION)
        if (
            matrices[0].shape != expected_shape
            or matrices[1].shape != expected_shape
            or not np.isfinite(matrices[0]).all()
            or not np.isfinite(matrices[1]).all()
            or not np.array_equal(matrices[0], matrices[1])
        ):
            raise IntrinsicScorerError("embedding_batch_replay_failed")
        if is_formal_encoder:
            try:
                encoder.validate_internal()
            except Exception as exc:
                raise IntrinsicScorerError(
                    "formal_encoder_binding_invalid"
                ) from exc

        vectors: dict[tuple[str, str], np.ndarray] = {}
        for kind, key, _ in ordered_entries:
            start, end = chunk_slices[(kind, key)]
            vectors[(kind, key)] = _normalize_mean(
                matrices[0][start:end]
            )
        source_vectors = {
            key: vectors[("source", key)] for key in source_texts
        }
        mention_vectors = {
            key: vectors[("mention", key)] for key in mention_texts
        }
        runtime_receipt = getattr(encoder, "runtime_receipt", None)
        canary_receipt = getattr(encoder, "canary_receipt", None)
        runtime_receipt_payload = (
            dict(runtime_receipt)
            if isinstance(runtime_receipt, Mapping)
            else runtime_receipt
        )
        canary_receipt_payload = (
            dict(canary_receipt)
            if isinstance(canary_receipt, Mapping)
            else canary_receipt
        )
        receipt: dict[str, object] = {
            "version": SCORER_VERSION,
            "contract_hash": SCORER_CONTRACT_HASH,
            "construction_domain": (
                _FORMAL_CONSTRUCTION_DOMAIN
                if is_formal_encoder
                else _QUALIFICATION_CONSTRUCTION_DOMAIN
            ),
            "extraction_count": len(extraction_hashes),
            "extraction_set_commitment": _content_hash(
                list(extraction_hashes)
            ),
            "source_count": len(source_vectors),
            "mention_count": len(mention_vectors),
            "source_vector_commitment": _vector_mapping_commitment(
                source_vectors
            ),
            "mention_vector_commitment": _vector_mapping_commitment(
                mention_vectors
            ),
            "chunk_count": len(chunk_texts),
            "chunk_plan_commitment": _content_hash(chunk_plan),
            "embedding_float32_commitment": hashlib.sha256(
                matrices[0].astype("<f4", copy=False).tobytes(order="C")
            ).hexdigest(),
            "actual_batch_replay_exact": True,
            "runtime_receipt_commitment": _content_hash(
                runtime_receipt_payload
            ),
            "canary_receipt_commitment": _content_hash(
                canary_receipt_payload
            ),
            "legacy_registry_sha256": LEGACY_REGISTRY_SHA256,
            "benchmark_source_accessed": False,
            "labels_accessed": False,
        }
        receipt["self_hash"] = _content_hash(receipt)
        return FrozenNarrativeScorers(
            source_vectors=source_vectors,
            mention_vectors=mention_vectors,
            primed_extraction_hashes=frozenset(extraction_hashes),
            receipt=receipt,
            _construction_token=_SCORER_FACTORY_MARKER,
        )

    def _validate_extraction(self, extraction: NarrativeExtraction) -> None:
        if (
            not isinstance(extraction, NarrativeExtraction)
            or extraction.extraction_hash
            not in self.primed_extraction_hashes
        ):
            raise IntrinsicScorerError("extraction_not_primed")
        extraction.__post_init__()

    def raw_text_scorer(self, left: bytes, right: bytes) -> int:
        left_key = _source_key(left)
        right_key = _source_key(right)
        try:
            left_vector = self.source_vectors[left_key]
            right_vector = self.source_vectors[right_key]
        except KeyError as exc:
            raise IntrinsicScorerError("raw_text_not_primed") from exc
        return quantized_cosine_similarity(left_vector, right_vector)

    def legacy_vectorizer(
        self,
        extraction: NarrativeExtraction,
        feature_ids: tuple[str, ...],
    ) -> tuple[int, ...]:
        self._validate_extraction(extraction)
        if feature_ids != LEGACY_FEATURE_IDS:
            raise IntrinsicScorerError("legacy_feature_registry_drifted")
        try:
            rows = search_structural_patterns(
                None,
                extraction.source.text,
                top_n=len(LEGACY_FEATURE_IDS),
                min_score=0.0,
                include_defaults=True,
            )
        except Exception as exc:
            raise IntrinsicScorerError("legacy_vectorization_failed") from exc
        scores: dict[str, int] = {}
        for row in rows:
            pattern_id = row.get("pattern_id")
            score = row.get("score")
            if (
                pattern_id not in LEGACY_FEATURE_IDS
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not np.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                raise IntrinsicScorerError("legacy_score_invalid")
            scores[str(pattern_id)] = int(
                round(float(score) * LEGACY_SCORE_SCALE)
            )
        if len(scores) != len(LEGACY_FEATURE_IDS):
            raise IntrinsicScorerError("legacy_pattern_set_incomplete")
        return tuple(scores[feature_id] for feature_id in feature_ids)

    def structural_scorer(
        self,
        source: NarrativeExtraction,
        target: NarrativeExtraction,
    ) -> SemanticScoreTable:
        self._validate_extraction(source)
        self._validate_extraction(target)
        source_mentions = {
            mention.mention_id: mention for mention in source.mentions
        }
        target_mentions = {
            mention.mention_id: mention for mention in target.mentions
        }

        def vector(mention_id: str, mentions: Mapping[str, object]) -> np.ndarray:
            mention = mentions.get(mention_id)
            quote_hash = getattr(mention, "quote_sha256", None)
            if not isinstance(quote_hash, str):
                raise IntrinsicScorerError("mention_reference_invalid")
            try:
                return self.mention_vectors[quote_hash]
            except KeyError as exc:
                raise IntrinsicScorerError("mention_not_primed") from exc

        object_scores: dict[tuple[str, str], int] = {}
        for source_id in source.hypergraph.object_mention_ids:
            for target_id in target.hypergraph.object_mention_ids:
                object_scores[(source_id, target_id)] = (
                    quantized_cosine_similarity(
                        vector(source_id, source_mentions),
                        vector(target_id, target_mentions),
                    )
                )
        generator_scores: dict[tuple[str, str], int] = {}
        for source_generator in source.generators:
            for target_generator in target.generators:
                generator_scores[
                    (
                        source_generator.generator_id,
                        target_generator.generator_id,
                    )
                ] = quantized_cosine_similarity(
                    vector(
                        source_generator.anchor_mention_id,
                        source_mentions,
                    ),
                    vector(
                        target_generator.anchor_mention_id,
                        target_mentions,
                    ),
                )
        return SemanticScoreTable.from_mappings(
            object_scores=object_scores,
            generator_scores=generator_scores,
        )


__all__ = [
    "FrozenNarrativeScorers",
    "IntrinsicScorerError",
    "LEGACY_FEATURE_IDS",
    "LEGACY_REGISTRY_SHA256",
    "SCORER_CONTRACT",
    "SCORER_CONTRACT_HASH",
    "SCORER_VERSION",
    "lossless_token_chunks",
]
