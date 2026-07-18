"""FEVEROUS-owned, claim-only offline semantic tensor adapter.

The adapter has no dataset-row, label, family, evidence, gold, HippoRAG, RAW
candidate, filesystem, network, or online-evaluator capability.  Its complete
input is the already-frozen 8192-unit public corpus view followed by claim
strings, and three injected local backends whose immutable asset provenance is
checked against the committed MiniLM, NER, and NLI trust roots.  Corpus MiniLM
embeddings, TARGET-only NER/numeric sidecars, and the typed graph are prepared
exactly once; every subsequent query reuses that immutable asset.

The semantic combination is explicit and deterministic:

* MiniLM cosine is computed for the claim and every facet against all 8192
  units using the Qasper binary64 ``fsum``/Python-round/ties-to-even contract;
* for each facet, NLI is called only on the union of its MiniLM top 32 and its
  exact typed-entity or numeric/date matches;
* semantic coverage is ``max(MiniLM, NLI-if-scored, 1_000_000-if-exact)``;
* direct-anchor strength is ``max(positive NLI-if-scored,
  1_000_000-if-exact)``.  MiniLM alone never creates a direct anchor.

The module does not instantiate heavyweight models.  Production code may wrap
the existing verified Qasper/MultiHopRAG/QASC runtimes; synthetic tests use
explicitly marked test doubles, which are rejected unless the caller opts in.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
import unicodedata
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from replication_runtime.multihoprag_ner_v1 import binding as ner_binding
from replication_runtime.qasc_nli_v1 import binding as nli_binding
from replication_runtime.qasper_minilm_v1 import binding as minilm_binding

from .feverous_atomic_corpus_v1 import (
    FeverousAtomicCorpusError,
    NerSpan,
    _NUMERIC_OR_DATE_RE as _ATOMIC_NUMBER_OR_DATE_RE,
    compile_claim_facets as compile_atomic_claim_facets,
)

from .feverous_p6_query_anchored_operator_v1 import (
    AtomicUnit,
    ClaimFacet,
    EntityKey,
    FeverousP6OperatorError,
    INTEGER_SCALE,
    QuerySemanticTensor,
    TypedCorpusGraph,
    build_typed_graph,
    make_claim_facet,
    make_entity_key,
    make_query_semantic_tensor,
    normalize_key,
    recompute_graph_sha256,
)


VERSION = "feverous_offline_semantic_tensor_v1"
BINDING_VERSION = "feverous_offline_semantic_backend_binding_v1"
DESIGN_SHA256 = "6193646baca9e35820a5d157bc248012fbd478c89a45db7d879295c4d64f0181"
CORPUS_SIZE = 8192
MINILM_TOP_K = 32
NER_CHUNK_SIZE = 4096
NLI_MAXIMUM_PAIRS_PER_CALL = 16_384
MAXIMUM_CLAIM_CHARACTERS = 131_072
ENTITY_TYPES = ("LOC", "MISC", "ORG", "PER")
UNIT_TYPES = ("sentence", "item", "cell", "header_cell", "table_caption")

SEMANTIC_COMBINATION = (
    "coverage=max(full_corpus_quantized_MiniLM,shortlist_NLI_margin,"
    "exact_entity_or_numeric_match_1000000);"
    "direct_anchor=max(shortlist_positive_NLI_margin,"
    "exact_entity_or_numeric_match_1000000);MiniLM_alone_is_not_an_anchor"
)

_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LINE_LABELS = {
    "sentence": ("TARGET", "TITLE", "SECTION_PATH", "TYPE"),
    "item": (
        "TARGET",
        "TITLE",
        "SECTION_PATH",
        "TYPE",
        "LIST_ANCESTOR_PATH",
    ),
    "cell": (
        "TARGET",
        "TITLE",
        "SECTION_PATH",
        "TYPE",
        "TABLE_CAPTION",
        "APPLICABLE_HEADERS",
        "ROW_WITH_TARGET_MARKED",
    ),
    "header_cell": (
        "TARGET",
        "TITLE",
        "SECTION_PATH",
        "TYPE",
        "TABLE_CAPTION",
        "APPLICABLE_HEADERS",
        "ROW_WITH_TARGET_MARKED",
    ),
    "table_caption": (
        "TARGET",
        "TITLE",
        "SECTION_PATH",
        "TYPE",
        "TABLE_KIND",
    ),
}


class FeverousSemanticTensorError(ValueError):
    """A semantic input, backend, or receipt violated the frozen contract."""


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
        raise FeverousSemanticTensorError("value is not canonical JSON") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise FeverousSemanticTensorError(f"{field} is not a lowercase SHA-256")
    return value


def normalize_surface_text(value: str) -> str:
    """Apply the frozen NFKC/whitespace text normalization without casefold."""

    if not isinstance(value, str):
        raise FeverousSemanticTensorError("surface text must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).split()).strip()


def _expected_backend(role: str) -> dict[str, str]:
    if role == "MiniLM":
        return {
            "asset_sha256": minilm_binding.ASSET_SELF_SHA256,
            "model_id": minilm_binding.MODEL_ID,
            "model_revision": minilm_binding.MODEL_REVISION,
            "model_tree_sha256": minilm_binding.MODEL_TREE_SHA256,
            "weights_sha256": minilm_binding.WEIGHTS_SHA256,
        }
    if role == "NER":
        return {
            "asset_sha256": ner_binding.ASSET_SELF_SHA256,
            "model_id": ner_binding.MODEL_ID,
            "model_revision": ner_binding.MODEL_REVISION,
            "model_tree_sha256": ner_binding.MODEL_TREE_SHA256,
            "weights_sha256": ner_binding.WEIGHTS_SHA256,
        }
    if role == "NLI":
        return {
            "asset_sha256": nli_binding.ASSET_SELF_SHA256,
            "model_id": nli_binding.MODEL_ID,
            "model_revision": nli_binding.MODEL_REVISION,
            "model_tree_sha256": nli_binding.MODEL_TREE_SHA256,
            "weights_sha256": nli_binding.WEIGHTS_SHA256,
        }
    raise FeverousSemanticTensorError("backend role is outside the frozen registry")


@dataclass(frozen=True)
class BackendBinding:
    """FEVEROUS-owned binding around one pre-verified local runtime."""

    role: str
    asset_sha256: str
    model_id: str
    model_revision: str
    model_tree_sha256: str
    weights_sha256: str
    runtime_receipt_sha256: str
    canary_receipt_sha256: str
    backend_kind: str
    offline_only: bool = True
    network_calls: int = 0

    def __post_init__(self) -> None:
        expected = _expected_backend(self.role)
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise FeverousSemanticTensorError(
                    f"{self.role} {field} drifted from the frozen asset"
                )
        _require_sha256(self.runtime_receipt_sha256, "runtime receipt hash")
        _require_sha256(self.canary_receipt_sha256, "canary receipt hash")
        if self.backend_kind not in {"verified_local_runtime", "synthetic_test_double"}:
            raise FeverousSemanticTensorError("backend kind is invalid")
        if self.offline_only is not True or self.network_calls != 0:
            raise FeverousSemanticTensorError("backend is not proven offline")

    def payload(self) -> dict[str, object]:
        return {
            "asset_sha256": self.asset_sha256,
            "backend_kind": self.backend_kind,
            "binding_version": BINDING_VERSION,
            "canary_receipt_sha256": self.canary_receipt_sha256,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_tree_sha256": self.model_tree_sha256,
            "network_calls": self.network_calls,
            "offline_only": self.offline_only,
            "role": self.role,
            "runtime_receipt_sha256": self.runtime_receipt_sha256,
            "weights_sha256": self.weights_sha256,
        }


def make_verified_backend_binding(
    *,
    role: str,
    runtime_receipt: Mapping[str, object],
    canary_receipt: Mapping[str, object],
) -> BackendBinding:
    """Bind an already-verified existing runtime without loading a model."""

    expected = _expected_backend(role)
    if not runtime_receipt or not canary_receipt:
        raise FeverousSemanticTensorError("verified backend receipts are empty")
    for field in ("asset_sha256", "model_tree_sha256"):
        if runtime_receipt.get(field) != expected[field]:
            raise FeverousSemanticTensorError(
                f"{role} live runtime receipt {field} drifted"
            )
    if "weights_sha256" in runtime_receipt and (
        runtime_receipt.get("weights_sha256") != expected["weights_sha256"]
    ):
        raise FeverousSemanticTensorError(
            f"{role} live runtime receipt weights_sha256 drifted"
        )
    expected_status = {
        "MiniLM": "verified_offline_immutable_qasper_minilm_runtime",
        "NER": "verified_exact_six_file_offline_ner_runtime",
        "NLI": "verified_offline_immutable_runtime",
    }[role]
    if runtime_receipt.get("status") != expected_status:
        raise FeverousSemanticTensorError(
            f"{role} live runtime receipt status drifted"
        )
    if (
        canary_receipt.get("repeat_exact") is not True
        and canary_receipt.get("per_worker_startup_repeat_exact") is not True
    ):
        raise FeverousSemanticTensorError(f"{role} startup canary is not repeat-exact")
    return BackendBinding(
        role=role,
        **expected,
        runtime_receipt_sha256=stable_hash(dict(runtime_receipt)),
        canary_receipt_sha256=stable_hash(dict(canary_receipt)),
        backend_kind="verified_local_runtime",
    )


def make_synthetic_backend_binding(role: str) -> BackendBinding:
    """Return an unmistakably non-production binding for row-free tests."""

    expected = _expected_backend(role)
    return BackendBinding(
        role=role,
        **expected,
        runtime_receipt_sha256=stable_hash(
            {"role": role, "scope": "synthetic_test_double_runtime"}
        ),
        canary_receipt_sha256=stable_hash(
            {"role": role, "scope": "synthetic_test_double_canary"}
        ),
        backend_kind="synthetic_test_double",
    )


@dataclass(frozen=True)
class DetectedEntity:
    entity_type: str
    start: int
    end: int
    text: str


class MiniLMBackend(Protocol):
    binding: BackendBinding

    def encode(self, texts: Sequence[str]) -> object: ...


class NERBackend(Protocol):
    binding: BackendBinding

    def extract_texts(
        self, texts: Sequence[str]
    ) -> Sequence[Sequence[DetectedEntity]]: ...


class NLIBackend(Protocol):
    binding: BackendBinding

    def score_pairs(
        self, pairs: Sequence[Mapping[str, str]]
    ) -> Sequence[int]: ...


@dataclass(frozen=True)
class BoundMiniLMBackend:
    """Thin FEVEROUS-owned view of the existing Qasper encoder."""

    runtime: object
    binding: BackendBinding

    def __post_init__(self) -> None:
        if self.binding.role != "MiniLM" or not callable(
            getattr(self.runtime, "encode", None)
        ):
            raise FeverousSemanticTensorError("bound MiniLM runtime is malformed")

    def encode(self, texts: Sequence[str]) -> object:
        return self.runtime.encode(texts)


@dataclass(frozen=True)
class BoundNERBackend:
    """Map text-only FEVEROUS calls onto the verified MultiHopRAG NER API."""

    runtime: object
    binding: BackendBinding

    def __post_init__(self) -> None:
        if self.binding.role != "NER" or not callable(
            getattr(self.runtime, "extract_inputs", None)
        ):
            raise FeverousSemanticTensorError("bound NER runtime is malformed")

    def extract_texts(
        self, texts: Sequence[str]
    ) -> tuple[tuple[DetectedEntity, ...], ...]:
        raw = self.runtime.extract_inputs(
            tuple({"kind": "query", "query": text} for text in texts)
        )
        return tuple(
            tuple(
                DetectedEntity(
                    entity_type=span.entity_type,
                    start=span.start,
                    end=span.end,
                    text=span.text,
                )
                for span in row
            )
            for row in raw
        )


@dataclass(frozen=True)
class BoundNLIBackend:
    """Thin FEVEROUS-owned view of a verified QASC NLI scorer or pool."""

    runtime: object
    binding: BackendBinding

    def __post_init__(self) -> None:
        if self.binding.role != "NLI" or not callable(
            getattr(self.runtime, "score_pairs", None)
        ):
            raise FeverousSemanticTensorError("bound NLI runtime is malformed")

    def score_pairs(
        self, pairs: Sequence[Mapping[str, str]]
    ) -> Sequence[int]:
        return self.runtime.score_pairs(pairs)


def _parse_target_first_text(
    *, linearized_text: object, unit_type: str, section_path: tuple[str, ...]
) -> str:
    """Validate the exact multiline atomic serialization and return TARGET only."""

    if (
        not isinstance(linearized_text, str)
        or not linearized_text
        or "\x00" in linearized_text
        or "\r" in linearized_text
        or unicodedata.normalize("NFKC", linearized_text) != linearized_text
    ):
        raise FeverousSemanticTensorError("linearized text is not safe frozen UTF-8 text")
    expected_labels = _LINE_LABELS.get(unit_type)
    if expected_labels is None:
        raise FeverousSemanticTensorError("unit type is outside the registry")
    lines = linearized_text.split("\n")
    if len(lines) != len(expected_labels) or any(not line for line in lines):
        raise FeverousSemanticTensorError(
            "linearized text does not have the frozen multiline schema"
        )
    values: dict[str, str] = {}
    observed_labels: list[str] = []
    for line in lines:
        label, separator, value = line.partition(": ")
        if separator != ": " or not value or normalize_surface_text(value) != value:
            raise FeverousSemanticTensorError(
                "linearized text line is outside frozen normalization"
            )
        observed_labels.append(label)
        values[label] = value
    if tuple(observed_labels) != expected_labels or lines[0] != (
        f"TARGET: {values.get('TARGET', '')}"
    ):
        raise FeverousSemanticTensorError(
            "linearized text is not the frozen TARGET-first serialization"
        )
    if values["TYPE"] != unit_type:
        raise FeverousSemanticTensorError("linearized TYPE disagrees with sidecar")
    expected_section = " > ".join(section_path) or "<ROOT>"
    if values["SECTION_PATH"] != expected_section:
        raise FeverousSemanticTensorError(
            "linearized SECTION_PATH disagrees with sidecar"
        )
    return values["TARGET"]


@dataclass(frozen=True)
class SemanticCorpusUnit:
    """One exact multiline atomic unit plus Agent-only structural sidecars."""

    corpus_ordinal: int
    linearized_text: str
    unit_type: str
    page_key: str
    official_order: int
    section_path: tuple[str, ...] = ()
    table_key: str | None = None
    table_row: int | None = None
    applicable_header_ordinals: tuple[int, ...] = ()
    list_parent_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.corpus_ordinal, bool)
            or not isinstance(self.corpus_ordinal, int)
            or self.corpus_ordinal < 0
        ):
            raise FeverousSemanticTensorError("corpus ordinal is invalid")
        if not isinstance(self.section_path, tuple):
            raise FeverousSemanticTensorError("section path must be an immutable tuple")
        if any(
            not isinstance(value, str)
            or not value
            or normalize_surface_text(value) != value
            or "\x00" in value
            for value in self.section_path
        ):
            raise FeverousSemanticTensorError("section path is invalid")
        _parse_target_first_text(
            linearized_text=self.linearized_text,
            unit_type=self.unit_type,
            section_path=self.section_path,
        )
        # Reuse the operator's exact structural validation with no entities yet.
        self.to_atomic_unit(())

    @property
    def target_text(self) -> str:
        return _parse_target_first_text(
            linearized_text=self.linearized_text,
            unit_type=self.unit_type,
            section_path=self.section_path,
        )

    def to_atomic_unit(self, entities: Sequence[EntityKey]) -> AtomicUnit:
        canonical_entities = tuple(sorted(set(entities)))
        return AtomicUnit(
            corpus_ordinal=self.corpus_ordinal,
            unit_type=self.unit_type,
            page_key=self.page_key,
            official_order=self.official_order,
            section_path=self.section_path,
            table_key=self.table_key,
            table_row=self.table_row,
            applicable_header_ordinals=self.applicable_header_ordinals,
            list_parent_path=self.list_parent_path,
            entities=canonical_entities,
        )


@dataclass(frozen=True)
class PreparedSemanticCorpus:
    """Immutable result of the one-time corpus embedding/NER/graph pass."""

    corpus_units: tuple[SemanticCorpusUnit, ...]
    target_texts: tuple[str, ...]
    typed_entities: tuple[tuple[EntityKey, ...], ...]
    numeric_keys: tuple[frozenset[str], ...]
    atomic_units: tuple[AtomicUnit, ...]
    graph: TypedCorpusGraph
    corpus_embedding_f32_le: bytes
    minilm_backend_binding: BackendBinding
    ner_backend_binding: BackendBinding
    receipt: Mapping[str, object]

    @property
    def preparation_receipt_sha256(self) -> str:
        value = self.receipt.get("preparation_receipt_sha256")
        assert isinstance(value, str)
        return value


@dataclass(frozen=True)
class SemanticTensorBuild:
    atomic_units: tuple[AtomicUnit, ...]
    graph: TypedCorpusGraph
    tensor: QuerySemanticTensor
    receipt: Mapping[str, object]


@dataclass(frozen=True)
class _FacetMeta:
    facet: ClaimFacet
    exact_entity_key: EntityKey | None = None
    exact_numeric_key: str | None = None


def _validate_binding(
    backend: object, role: str, *, allow_synthetic_backends: bool
) -> BackendBinding:
    binding = getattr(backend, "binding", None)
    if not isinstance(binding, BackendBinding) or binding.role != role:
        raise FeverousSemanticTensorError(f"{role} backend binding is missing")
    # __post_init__ already checked every immutable trust-root field.
    if binding.backend_kind == "synthetic_test_double" and not allow_synthetic_backends:
        raise FeverousSemanticTensorError(
            "synthetic backend is forbidden outside explicit row-free tests"
        )
    return binding


def _validate_claim(claim_text: object) -> str:
    if (
        not isinstance(claim_text, str)
        or not claim_text.strip()
        or "\x00" in claim_text
        or len(claim_text) > MAXIMUM_CLAIM_CHARACTERS
    ):
        raise FeverousSemanticTensorError("claim text is outside the frozen bound")
    normalized = normalize_surface_text(claim_text)
    if not normalized:
        raise FeverousSemanticTensorError("claim text is empty after normalization")
    return normalized


def _validate_corpus(
    corpus_units: Sequence[SemanticCorpusUnit],
) -> tuple[SemanticCorpusUnit, ...]:
    if isinstance(corpus_units, (str, bytes)) or not isinstance(corpus_units, Sequence):
        raise FeverousSemanticTensorError("corpus must be a sequence")
    rows = tuple(corpus_units)
    if len(rows) != CORPUS_SIZE or any(
        not isinstance(row, SemanticCorpusUnit) for row in rows
    ):
        raise FeverousSemanticTensorError("corpus must contain exactly 8192 typed units")
    if tuple(row.corpus_ordinal for row in rows) != tuple(range(CORPUS_SIZE)):
        raise FeverousSemanticTensorError("corpus ordinals are not exact source order")
    return rows


def _validate_entity_rows(
    rows: object, texts: Sequence[str]
) -> tuple[tuple[DetectedEntity, ...], ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise FeverousSemanticTensorError("NER output is not a sequence")
    if len(rows) != len(texts):
        raise FeverousSemanticTensorError("NER output count drifted")
    result: list[tuple[DetectedEntity, ...]] = []
    for row, text in zip(rows, texts):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise FeverousSemanticTensorError("NER entity row is malformed")
        normalized: list[DetectedEntity] = []
        previous_end = -1
        for entity in row:
            if not isinstance(entity, DetectedEntity):
                raise FeverousSemanticTensorError("NER emitted a foreign entity type")
            if (
                entity.entity_type not in ENTITY_TYPES
                or isinstance(entity.start, bool)
                or not isinstance(entity.start, int)
                or isinstance(entity.end, bool)
                or not isinstance(entity.end, int)
                or not 0 <= entity.start < entity.end <= len(text)
                or entity.start < previous_end
                or text[entity.start : entity.end] != entity.text
                or not entity.text.strip()
            ):
                raise FeverousSemanticTensorError("NER entity offsets or type drifted")
            normalized.append(entity)
            previous_end = entity.end
        result.append(tuple(normalized))
    return tuple(result)


def _extract_all_entities(
    backend: NERBackend, texts: Sequence[str]
) -> tuple[tuple[DetectedEntity, ...], ...]:
    result: list[tuple[DetectedEntity, ...]] = []
    for offset in range(0, len(texts), NER_CHUNK_SIZE):
        batch = tuple(texts[offset : offset + NER_CHUNK_SIZE])
        try:
            raw = backend.extract_texts(batch)
        except Exception as exc:
            raise FeverousSemanticTensorError("offline NER backend failed") from exc
        result.extend(_validate_entity_rows(raw, batch))
    return tuple(result)


def _numeric_spans(text: str) -> tuple[tuple[int, int, str], ...]:
    return tuple(
        (match.start(), match.end(), normalize_key(match.group(0)))
        for match in _ATOMIC_NUMBER_OR_DATE_RE.finditer(text)
    )


def extract_claim_facets(
    *, claim_text: str, claim_entities: Sequence[DetectedEntity]
) -> tuple[_FacetMeta, ...]:
    """Adapt the single frozen atomic claim-facet compiler to operator facets."""

    claim = _validate_claim(claim_text)
    entities = _validate_entity_rows((tuple(claim_entities),), (claim,))[0]
    try:
        compiled = compile_atomic_claim_facets(
            claim,
            tuple(NerSpan(entity.start, entity.end) for entity in entities),
        )
    except FeverousAtomicCorpusError as exc:
        raise FeverousSemanticTensorError(
            "atomic claim facet compilation failed"
        ) from exc
    typed_by_span = {
        (entity.start, entity.end): make_entity_key(
            entity.entity_type, entity.text
        )
        for entity in entities
    }
    result: list[_FacetMeta] = []
    for source in compiled.facets:
        exact_entity_key = None
        exact_numeric_key = None
        if source.kind == "entity":
            exact_entity_key = typed_by_span.get(
                (source.source_start, source.source_end)
            )
            if exact_entity_key is None:
                raise FeverousSemanticTensorError(
                    "atomic entity facet lost its typed NER span"
                )
        elif source.kind == "numeric_or_date":
            exact_numeric_key = normalize_key(source.text)
        result.append(
            _FacetMeta(
                facet=make_claim_facet(len(result), source.kind, source.text),
                exact_entity_key=exact_entity_key,
                exact_numeric_key=exact_numeric_key,
            )
        )
    return tuple(result)


def _validate_embeddings(values: object, expected_rows: int) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise FeverousSemanticTensorError("MiniLM output is not float32") from exc
    if matrix.shape != (expected_rows, minilm_binding.EMBEDDING_DIMENSION):
        raise FeverousSemanticTensorError("MiniLM output shape drifted")
    if not np.isfinite(matrix).all():
        raise FeverousSemanticTensorError("MiniLM output is non-finite")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-6):
        raise FeverousSemanticTensorError("MiniLM output is not normalized")
    return np.ascontiguousarray(matrix, dtype="<f4")


def _embedding_bytes(matrix: np.ndarray) -> bytes:
    return np.ascontiguousarray(matrix, dtype="<f4").tobytes(order="C")


def _prepared_embedding_matrix(prepared: PreparedSemanticCorpus) -> np.ndarray:
    expected_bytes = (
        CORPUS_SIZE * minilm_binding.EMBEDDING_DIMENSION * np.dtype("<f4").itemsize
    )
    if len(prepared.corpus_embedding_f32_le) != expected_bytes:
        raise FeverousSemanticTensorError("prepared corpus embedding byte count drifted")
    matrix = np.frombuffer(prepared.corpus_embedding_f32_le, dtype="<f4").reshape(
        CORPUS_SIZE, minilm_binding.EMBEDDING_DIMENSION
    )
    if matrix.flags.writeable:
        raise FeverousSemanticTensorError("prepared corpus embeddings are mutable")
    return matrix


_FLOAT64_UNIT_ROUNDOFF = 2.0**-53
_DOT_ERROR_TERM_COUNT = minilm_binding.EMBEDDING_DIMENSION + 2
_DOT_GAMMA = (
    _DOT_ERROR_TERM_COUNT * _FLOAT64_UNIT_ROUNDOFF
) / (1.0 - _DOT_ERROR_TERM_COUNT * _FLOAT64_UNIT_ROUNDOFF)
_FSUM_ULP_ALLOWANCE = 8.0
_QUANTIZATION_CORPUS_BATCH_SIZE = 1024


def _exact_quantized_similarity(left: np.ndarray, right: np.ndarray) -> int:
    """The frozen Qasper scalar expression, after exact float32 coercion."""

    cosine = math.fsum(
        float(left[index]) * float(right[index])
        for index in range(minilm_binding.EMBEDDING_DIMENSION)
    )
    return int(round(cosine * INTEGER_SCALE))


def _quantized_matrix(
    queries: np.ndarray, corpus: np.ndarray
) -> tuple[tuple[int, ...], ...]:
    """Batch exact integer cosines with a proven scalar fallback boundary.

    A binary32 value is represented exactly as binary64, and a product of two
    binary32 significands needs at most 48 bits, so every individual product is
    exact in binary64.  The single-threaded ``einsum`` result can therefore be
    bounded against the exact sum by ``gamma_n * sum(abs(products))``.  We use
    the conservative, exactly representable upper bound
    ``384 * max(abs(query)) * max(abs(row))``, plus an eight-ULP allowance for
    the frozen ``math.fsum`` and scale operations.  A fast result is accepted
    only when that complete interval cannot touch a half-integer rounding
    boundary; every ambiguous cell executes the literal frozen scalar formula.
    """

    left = np.asarray(queries, dtype=np.float32)
    rows = np.asarray(corpus, dtype=np.float32)
    if (
        left.ndim != 2
        or left.shape[1] != minilm_binding.EMBEDDING_DIMENSION
        or rows.ndim != 2
        or rows.shape[1] != minilm_binding.EMBEDDING_DIMENSION
        or not np.isfinite(left).all()
        or not np.isfinite(rows).all()
    ):
        raise FeverousSemanticTensorError("MiniLM cosine matrix shape drifted")

    left64 = left.astype(np.float64)
    left_maximum = np.max(np.abs(left64), axis=1)[:, None]
    result: list[list[int]] = [[] for _query in left]
    for offset in range(0, len(rows), _QUANTIZATION_CORPUS_BATCH_SIZE):
        stop = min(offset + _QUANTIZATION_CORPUS_BATCH_SIZE, len(rows))
        row_batch = rows[offset:stop]
        rows64 = row_batch.astype(np.float64)
        # optimize=False stays in NumPy's single-threaded C iterator instead of
        # entering a many-threaded BLAS pool for each concurrently evaluated item.
        approximate = np.einsum(
            "qd,nd->qn", left64, rows64, optimize=False
        )
        absolute_sum_upper = (
            minilm_binding.EMBEDDING_DIMENSION
            * left_maximum
            * np.max(np.abs(rows64), axis=1)[None, :]
        )
        cosine_error = np.nextafter(
            _DOT_GAMMA * absolute_sum_upper
            + _FSUM_ULP_ALLOWANCE
            * np.spacing(np.maximum(1.0, absolute_sum_upper)),
            np.inf,
        )
        scaled = approximate * INTEGER_SCALE
        scaled_error = np.nextafter(
            cosine_error * INTEGER_SCALE
            + _FSUM_ULP_ALLOWANCE
            * np.spacing(np.maximum(1.0, np.abs(scaled))),
            np.inf,
        )
        fractional = scaled - np.floor(scaled)
        safe = (
            np.isfinite(scaled)
            & (np.abs(scaled) < 2.0**50)
            & (np.abs(fractional - 0.5) > scaled_error)
        )

        # Unsafe values are replaced before int64 conversion; their exact Python
        # integers may exceed int64 for non-normalized synthetic input.
        fast_scaled = np.where(safe, scaled, 0.0)
        fast = np.rint(fast_scaled).astype(np.int64)
        batch_result = [[int(value) for value in row] for row in fast]
        for query_i, batch_i in zip(*np.nonzero(~safe)):
            batch_result[int(query_i)][int(batch_i)] = (
                _exact_quantized_similarity(
                    left[int(query_i)], row_batch[int(batch_i)]
                )
            )
        for query_i, row in enumerate(batch_result):
            result[query_i].extend(row)
    return tuple(tuple(row) for row in result)


def _quantized_vector(query: np.ndarray, corpus: np.ndarray) -> tuple[int, ...]:
    """Compatibility seam for one exact query-by-corpus similarity vector."""

    left = np.asarray(query, dtype=np.float32)
    if left.shape != (minilm_binding.EMBEDDING_DIMENSION,):
        raise FeverousSemanticTensorError("MiniLM cosine query shape drifted")
    return _quantized_matrix(left[None, :], corpus)[0]


def quantized_minilm_similarity(left: object, right: object) -> int:
    """Expose the exact bound Qasper cosine quantizer under FEVEROUS ownership."""

    try:
        return minilm_binding.quantized_cosine_similarity(left, right)
    except Exception as exc:
        raise FeverousSemanticTensorError("quantized MiniLM similarity failed") from exc


def _score_nli(
    backend: NLIBackend, pairs: Sequence[Mapping[str, str]]
) -> tuple[int, ...]:
    if len(pairs) > NLI_MAXIMUM_PAIRS_PER_CALL:
        raise FeverousSemanticTensorError("NLI shortlist escaped the frozen call bound")
    if not pairs:
        return ()
    try:
        raw = backend.score_pairs(tuple(pairs))
    except Exception as exc:
        raise FeverousSemanticTensorError("offline NLI backend failed") from exc
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) != len(pairs):
        raise FeverousSemanticTensorError("NLI output count drifted")
    scores: list[int] = []
    for value in raw:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not -(2**63) <= value <= 2**63 - 1
        ):
            raise FeverousSemanticTensorError("NLI margin is not a frozen integer")
        scores.append(value)
    return tuple(scores)


def _unit_receipt(unit: SemanticCorpusUnit) -> list[object]:
    return [
        unit.corpus_ordinal,
        stable_hash(unit.linearized_text),
        unit.unit_type,
        stable_hash(unit.page_key),
        unit.official_order,
        [stable_hash(value) for value in unit.section_path],
        None if unit.table_key is None else stable_hash(unit.table_key),
        unit.table_row,
        list(unit.applicable_header_ordinals),
        [stable_hash(value) for value in unit.list_parent_path],
    ]


def _self_hashed(
    body: Mapping[str, object], *, field: str = "semantic_receipt_sha256"
) -> dict[str, object]:
    value = dict(body)
    if field in value:
        raise FeverousSemanticTensorError("self-hash field already exists")
    value[field] = stable_hash(value)
    return value


def _binding_sha256(binding: BackendBinding) -> str:
    return stable_hash(binding.payload())


_PREPARATION_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "version",
        "design_sha256",
        "corpus_size",
        "corpus_commitment_sha256",
        "target_text_commitment_sha256",
        "typed_entity_sidecar_sha256",
        "numeric_target_sidecar_sha256",
        "corpus_embedding_sha256",
        "corpus_embedding_dtype",
        "corpus_embedding_shape",
        "graph_sha256",
        "MiniLM_binding_sha256",
        "MiniLM_asset_sha256",
        "MiniLM_backend_kind",
        "NER_binding_sha256",
        "NER_asset_sha256",
        "NER_backend_kind",
        "corpus_MiniLM_call_count",
        "corpus_MiniLM_encoded_text_count",
        "corpus_NER_call_count",
        "corpus_NER_target_text_count",
        "unit_NER_and_numeric_scope",
        "full_target_first_text_embedded",
        "graph_built_once",
        "labels_family_gold_evidence_or_Hippo_accessed",
        "network_calls",
        "online_evaluator_calls",
        "preparation_receipt_sha256",
    }
)

_QUERY_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "version",
        "design_sha256",
        "query_sha256",
        "corpus_size",
        "preparation_receipt_sha256",
        "preparation_mode",
        "prepared_corpus_reused",
        "corpus_commitment_sha256",
        "corpus_embedding_sha256",
        "graph_sha256",
        "facet_count",
        "facet_schema_sha256",
        "MiniLM_binding_sha256",
        "MiniLM_asset_sha256",
        "NER_binding_sha256",
        "NER_asset_sha256",
        "NLI_binding_sha256",
        "NLI_asset_sha256",
        "MiniLM_call_count",
        "MiniLM_encoded_text_count",
        "MiniLM_similarity_count",
        "MiniLM_quantization",
        "full_corpus_scan",
        "NLI_shortlist_policy",
        "NLI_pair_count",
        "shortlists",
        "semantic_combination",
        "claim_NER_text_count",
        "claim_NER_call_count",
        "corpus_MiniLM_calls_in_query",
        "corpus_NER_calls_in_query",
        "tensor_sha256",
        "atomic_unit_count",
        "input_capability",
        "labels_family_gold_evidence_or_Hippo_accessed",
        "RAW_or_Hippo_candidates_consumed",
        "network_calls",
        "online_evaluator_calls",
        "raw_claim_or_corpus_text_persisted",
        "semantic_receipt_sha256",
    }
)


def verify_preparation_receipt(receipt: Mapping[str, object]) -> str:
    """Verify the independent one-time corpus-preparation commitment."""

    if not isinstance(receipt, Mapping) or set(receipt) != _PREPARATION_RECEIPT_KEYS:
        raise FeverousSemanticTensorError("preparation receipt schema drifted")
    body = dict(receipt)
    declared = _require_sha256(
        body.pop("preparation_receipt_sha256", None),
        "preparation receipt hash",
    )
    hashes = (
        "corpus_commitment_sha256",
        "target_text_commitment_sha256",
        "typed_entity_sidecar_sha256",
        "numeric_target_sidecar_sha256",
        "corpus_embedding_sha256",
        "graph_sha256",
        "MiniLM_binding_sha256",
        "NER_binding_sha256",
    )
    for field in hashes:
        _require_sha256(receipt.get(field), field)
    if (
        stable_hash(body) != declared
        or receipt.get("schema") != f"{VERSION}_prepared_corpus_receipt"
        or receipt.get("version") != VERSION
        or receipt.get("design_sha256") != DESIGN_SHA256
        or receipt.get("corpus_size") != CORPUS_SIZE
        or receipt.get("corpus_embedding_dtype") != "little_endian_float32"
        or receipt.get("corpus_embedding_shape")
        != f"{CORPUS_SIZE}x{minilm_binding.EMBEDDING_DIMENSION}"
        or receipt.get("MiniLM_asset_sha256") != minilm_binding.ASSET_SELF_SHA256
        or receipt.get("NER_asset_sha256") != ner_binding.ASSET_SELF_SHA256
        or receipt.get("MiniLM_backend_kind")
        not in {"verified_local_runtime", "synthetic_test_double"}
        or receipt.get("NER_backend_kind")
        not in {"verified_local_runtime", "synthetic_test_double"}
        or receipt.get("corpus_MiniLM_call_count") != 1
        or receipt.get("corpus_MiniLM_encoded_text_count") != CORPUS_SIZE
        or receipt.get("corpus_NER_call_count")
        != (CORPUS_SIZE + NER_CHUNK_SIZE - 1) // NER_CHUNK_SIZE
        or receipt.get("corpus_NER_target_text_count") != CORPUS_SIZE
        or receipt.get("unit_NER_and_numeric_scope") != "TARGET_only"
        or receipt.get("full_target_first_text_embedded") is not True
        or receipt.get("graph_built_once") is not True
        or receipt.get("labels_family_gold_evidence_or_Hippo_accessed") is not False
        or receipt.get("network_calls") != 0
        or receipt.get("online_evaluator_calls") != 0
    ):
        raise FeverousSemanticTensorError("preparation receipt drifted")
    return declared


def _typed_entity_receipt(
    rows: Sequence[Sequence[EntityKey]],
) -> list[list[list[str]]]:
    return [
        [[entity.entity_type, entity.normalized_span] for entity in row]
        for row in rows
    ]


def _numeric_receipt(rows: Sequence[frozenset[str]]) -> list[list[str]]:
    return [sorted(row) for row in rows]


def verify_prepared_semantic_corpus(
    prepared: PreparedSemanticCorpus,
) -> str:
    """Perform a full one-time integrity audit of an immutable preparation."""

    if not isinstance(prepared, PreparedSemanticCorpus):
        raise FeverousSemanticTensorError("prepared corpus has the wrong type")
    receipt_sha256 = verify_preparation_receipt(prepared.receipt)
    corpus = _validate_corpus(prepared.corpus_units)
    targets = tuple(unit.target_text for unit in corpus)
    if targets != prepared.target_texts:
        raise FeverousSemanticTensorError("prepared TARGET projection drifted")
    if (
        len(prepared.typed_entities) != CORPUS_SIZE
        or len(prepared.numeric_keys) != CORPUS_SIZE
        or len(prepared.atomic_units) != CORPUS_SIZE
        or any(not isinstance(row, tuple) for row in prepared.typed_entities)
        or any(not isinstance(row, frozenset) for row in prepared.numeric_keys)
    ):
        raise FeverousSemanticTensorError("prepared corpus sidecar shape drifted")
    expected_atomic = tuple(
        unit.to_atomic_unit(prepared.typed_entities[unit.corpus_ordinal])
        for unit in corpus
    )
    if expected_atomic != prepared.atomic_units:
        raise FeverousSemanticTensorError("prepared atomic units drifted")
    if (
        not isinstance(prepared.graph, TypedCorpusGraph)
        or prepared.graph.units != prepared.atomic_units
        or recompute_graph_sha256(prepared.graph) != prepared.graph.graph_sha256
    ):
        raise FeverousSemanticTensorError("prepared typed graph drifted")
    matrix = _prepared_embedding_matrix(prepared)
    if not np.isfinite(matrix).all():
        raise FeverousSemanticTensorError("prepared corpus embeddings are non-finite")
    if prepared.minilm_backend_binding.role != "MiniLM" or (
        prepared.ner_backend_binding.role != "NER"
    ):
        raise FeverousSemanticTensorError("prepared backend binding role drifted")
    expected = {
        "corpus_commitment_sha256": stable_hash(
            [_unit_receipt(unit) for unit in corpus]
        ),
        "target_text_commitment_sha256": stable_hash(
            [stable_hash(value) for value in targets]
        ),
        "typed_entity_sidecar_sha256": stable_hash(
            _typed_entity_receipt(prepared.typed_entities)
        ),
        "numeric_target_sidecar_sha256": stable_hash(
            _numeric_receipt(prepared.numeric_keys)
        ),
        "corpus_embedding_sha256": hashlib.sha256(
            prepared.corpus_embedding_f32_le
        ).hexdigest(),
        "graph_sha256": prepared.graph.graph_sha256,
        "MiniLM_binding_sha256": _binding_sha256(
            prepared.minilm_backend_binding
        ),
        "NER_binding_sha256": _binding_sha256(prepared.ner_backend_binding),
    }
    if any(prepared.receipt.get(field) != value for field, value in expected.items()):
        raise FeverousSemanticTensorError("prepared corpus commitment drifted")
    return receipt_sha256


def _require_prepared_for_query(
    prepared: PreparedSemanticCorpus,
    *,
    minilm_binding_value: BackendBinding,
    ner_binding_value: BackendBinding,
) -> None:
    """Constant-size verification used by every query after full preparation QA."""

    if not isinstance(prepared, PreparedSemanticCorpus):
        raise FeverousSemanticTensorError("prepared corpus has the wrong type")
    verify_preparation_receipt(prepared.receipt)
    if (
        prepared.minilm_backend_binding != minilm_binding_value
        or prepared.ner_backend_binding != ner_binding_value
        or prepared.receipt.get("MiniLM_binding_sha256")
        != _binding_sha256(minilm_binding_value)
        or prepared.receipt.get("NER_binding_sha256")
        != _binding_sha256(ner_binding_value)
        or prepared.graph.units is not prepared.atomic_units
        or len(prepared.target_texts) != CORPUS_SIZE
        or len(prepared.typed_entities) != CORPUS_SIZE
        or len(prepared.numeric_keys) != CORPUS_SIZE
    ):
        raise FeverousSemanticTensorError(
            "query backend or immutable preparation binding drifted"
        )
    _prepared_embedding_matrix(prepared)


def prepare_semantic_corpus(
    *,
    corpus_units: Sequence[SemanticCorpusUnit],
    minilm_backend: MiniLMBackend,
    ner_backend: NERBackend,
    allow_synthetic_backends: bool = False,
) -> PreparedSemanticCorpus:
    """Run the sole corpus-wide NER, MiniLM, and typed-graph preparation pass."""

    corpus = _validate_corpus(corpus_units)
    minilm_backend_binding = _validate_binding(
        minilm_backend,
        "MiniLM",
        allow_synthetic_backends=allow_synthetic_backends,
    )
    ner_backend_binding = _validate_binding(
        ner_backend,
        "NER",
        allow_synthetic_backends=allow_synthetic_backends,
    )
    targets = tuple(unit.target_text for unit in corpus)
    entity_rows = _extract_all_entities(ner_backend, targets)
    typed_entities = tuple(
        tuple(
            sorted(
                {
                    make_entity_key(entity.entity_type, entity.text)
                    for entity in row
                }
            )
        )
        for row in entity_rows
    )
    numeric_keys = tuple(
        frozenset(value for _start, _end, value in _numeric_spans(target))
        for target in targets
    )
    atomic_units = tuple(
        unit.to_atomic_unit(typed_entities[unit.corpus_ordinal])
        for unit in corpus
    )
    try:
        graph = build_typed_graph(atomic_units)
    except FeverousP6OperatorError as exc:
        raise FeverousSemanticTensorError("typed corpus graph preparation failed") from exc
    embedding_texts = tuple(unit.linearized_text for unit in corpus)
    try:
        raw_embeddings = minilm_backend.encode(embedding_texts)
    except Exception as exc:
        raise FeverousSemanticTensorError("offline MiniLM backend failed") from exc
    corpus_embeddings = _validate_embeddings(raw_embeddings, CORPUS_SIZE)
    corpus_embedding_f32_le = _embedding_bytes(corpus_embeddings)
    receipt = _self_hashed(
        {
            "schema": f"{VERSION}_prepared_corpus_receipt",
            "version": VERSION,
            "design_sha256": DESIGN_SHA256,
            "corpus_size": CORPUS_SIZE,
            "corpus_commitment_sha256": stable_hash(
                [_unit_receipt(unit) for unit in corpus]
            ),
            "target_text_commitment_sha256": stable_hash(
                [stable_hash(value) for value in targets]
            ),
            "typed_entity_sidecar_sha256": stable_hash(
                _typed_entity_receipt(typed_entities)
            ),
            "numeric_target_sidecar_sha256": stable_hash(
                _numeric_receipt(numeric_keys)
            ),
            "corpus_embedding_sha256": hashlib.sha256(
                corpus_embedding_f32_le
            ).hexdigest(),
            "corpus_embedding_dtype": "little_endian_float32",
            "corpus_embedding_shape": (
                f"{CORPUS_SIZE}x{minilm_binding.EMBEDDING_DIMENSION}"
            ),
            "graph_sha256": graph.graph_sha256,
            "MiniLM_binding_sha256": _binding_sha256(
                minilm_backend_binding
            ),
            "MiniLM_asset_sha256": minilm_backend_binding.asset_sha256,
            "MiniLM_backend_kind": minilm_backend_binding.backend_kind,
            "NER_binding_sha256": _binding_sha256(ner_backend_binding),
            "NER_asset_sha256": ner_backend_binding.asset_sha256,
            "NER_backend_kind": ner_backend_binding.backend_kind,
            "corpus_MiniLM_call_count": 1,
            "corpus_MiniLM_encoded_text_count": CORPUS_SIZE,
            "corpus_NER_call_count": (
                CORPUS_SIZE + NER_CHUNK_SIZE - 1
            )
            // NER_CHUNK_SIZE,
            "corpus_NER_target_text_count": CORPUS_SIZE,
            "unit_NER_and_numeric_scope": "TARGET_only",
            "full_target_first_text_embedded": True,
            "graph_built_once": True,
            "labels_family_gold_evidence_or_Hippo_accessed": False,
            "network_calls": 0,
            "online_evaluator_calls": 0,
        },
        field="preparation_receipt_sha256",
    )
    prepared = PreparedSemanticCorpus(
        corpus_units=corpus,
        target_texts=targets,
        typed_entities=typed_entities,
        numeric_keys=numeric_keys,
        atomic_units=atomic_units,
        graph=graph,
        corpus_embedding_f32_le=corpus_embedding_f32_le,
        minilm_backend_binding=minilm_backend_binding,
        ner_backend_binding=ner_backend_binding,
        receipt=MappingProxyType(receipt),
    )
    verify_prepared_semantic_corpus(prepared)
    return prepared


def verify_semantic_receipt(receipt: Mapping[str, object]) -> str:
    if not isinstance(receipt, Mapping) or set(receipt) != _QUERY_RECEIPT_KEYS:
        raise FeverousSemanticTensorError("semantic receipt schema drifted")
    body = dict(receipt)
    declared = _require_sha256(
        body.pop("semantic_receipt_sha256", None), "semantic receipt hash"
    )
    if (
        receipt.get("schema") != f"{VERSION}_query_receipt"
        or stable_hash(body) != declared
        or receipt.get("version") != VERSION
        or receipt.get("design_sha256") != DESIGN_SHA256
        or receipt.get("corpus_size") != CORPUS_SIZE
        or receipt.get("full_corpus_scan") is not True
        or receipt.get("semantic_combination") != SEMANTIC_COMBINATION
        or receipt.get("preparation_mode")
        not in {"precomputed_formal_path", "inline_compatibility_wrapper"}
        or receipt.get("prepared_corpus_reused") is not True
        or type(receipt.get("facet_count")) is not int
        or not 1 <= receipt.get("facet_count", 0) <= 8
        or receipt.get("MiniLM_call_count") != 1
        or receipt.get("MiniLM_encoded_text_count")
        != 1 + receipt.get("facet_count", 0)
        or receipt.get("MiniLM_similarity_count")
        != (1 + receipt.get("facet_count", 0)) * CORPUS_SIZE
        or receipt.get("NLI_shortlist_policy")
        != "per_facet_MiniLM_top32_union_exact_entity_or_numeric"
        or type(receipt.get("NLI_pair_count")) is not int
        or not 0 <= receipt.get("NLI_pair_count", -1) <= 8 * CORPUS_SIZE
        or not isinstance(receipt.get("shortlists"), list)
        or len(receipt.get("shortlists", ())) != receipt.get("facet_count")
        or receipt.get("claim_NER_call_count") != 1
        or receipt.get("corpus_MiniLM_calls_in_query") != 0
        or receipt.get("corpus_NER_calls_in_query") != 0
        or receipt.get("claim_NER_text_count") != 1
        or receipt.get("atomic_unit_count") != CORPUS_SIZE
        or receipt.get("input_capability")
        != "claim_text_plus_prepared_fixed_linearized_corpus_only"
        or receipt.get("RAW_or_Hippo_candidates_consumed") != 0
        or receipt.get("labels_family_gold_evidence_or_Hippo_accessed") is not False
        or receipt.get("network_calls") != 0
        or receipt.get("online_evaluator_calls") != 0
        or receipt.get("raw_claim_or_corpus_text_persisted") is not False
    ):
        raise FeverousSemanticTensorError("semantic receipt drifted")
    for field in (
        "query_sha256",
        "preparation_receipt_sha256",
        "corpus_commitment_sha256",
        "corpus_embedding_sha256",
        "graph_sha256",
        "MiniLM_binding_sha256",
        "NER_binding_sha256",
        "NLI_binding_sha256",
        "facet_schema_sha256",
        "tensor_sha256",
    ):
        _require_sha256(receipt.get(field), field)
    if (
        receipt.get("MiniLM_asset_sha256") != minilm_binding.ASSET_SELF_SHA256
        or receipt.get("NER_asset_sha256") != ner_binding.ASSET_SELF_SHA256
        or receipt.get("NLI_asset_sha256") != nli_binding.ASSET_SELF_SHA256
    ):
        raise FeverousSemanticTensorError("semantic backend asset drifted")
    return declared


def _build_from_prepared(
    *,
    claim_text: str,
    prepared_corpus: PreparedSemanticCorpus,
    minilm_backend: MiniLMBackend,
    ner_backend: NERBackend,
    nli_backend: NLIBackend,
    allow_synthetic_backends: bool,
    preparation_mode: str,
) -> SemanticTensorBuild:
    """Internal query-only path over an already-prepared immutable corpus."""

    claim = _validate_claim(claim_text)
    bindings = {
        "MiniLM": _validate_binding(
            minilm_backend, "MiniLM", allow_synthetic_backends=allow_synthetic_backends
        ),
        "NER": _validate_binding(
            ner_backend, "NER", allow_synthetic_backends=allow_synthetic_backends
        ),
        "NLI": _validate_binding(
            nli_backend, "NLI", allow_synthetic_backends=allow_synthetic_backends
        ),
    }
    _require_prepared_for_query(
        prepared_corpus,
        minilm_binding_value=bindings["MiniLM"],
        ner_binding_value=bindings["NER"],
    )

    entity_rows = _extract_all_entities(ner_backend, (claim,))
    facet_meta = extract_claim_facets(
        claim_text=claim, claim_entities=entity_rows[0]
    )
    facets = tuple(row.facet for row in facet_meta)
    embedding_texts = (
        claim,
        *(row.facet.normalized_text for row in facet_meta),
    )
    try:
        raw_embeddings = minilm_backend.encode(embedding_texts)
    except Exception as exc:
        raise FeverousSemanticTensorError("offline MiniLM backend failed") from exc
    embeddings = _validate_embeddings(raw_embeddings, len(embedding_texts))
    corpus_embeddings = _prepared_embedding_matrix(prepared_corpus)
    quantized_vectors = _quantized_matrix(embeddings, corpus_embeddings)
    dense_relevance = quantized_vectors[0]

    semantic_rows: list[tuple[int, ...]] = []
    direct_rows: list[tuple[int, ...]] = []
    shortlist_receipts: list[dict[str, object]] = []
    total_nli_pairs = 0
    for facet_index, meta in enumerate(facet_meta):
        similarities = quantized_vectors[1 + facet_index]
        top32 = tuple(
            sorted(
                range(CORPUS_SIZE),
                key=lambda ordinal: (-similarities[ordinal], ordinal),
            )[:MINILM_TOP_K]
        )
        exact: list[int] = []
        for ordinal in range(CORPUS_SIZE):
            if (
                meta.exact_entity_key is not None
                and meta.exact_entity_key
                in prepared_corpus.typed_entities[ordinal]
            ) or (
                meta.exact_numeric_key is not None
                and meta.exact_numeric_key
                in prepared_corpus.numeric_keys[ordinal]
            ):
                exact.append(ordinal)
        nli_ordinals = tuple(sorted(set(top32).union(exact)))
        pairs = tuple(
            {
                "hypothesis": meta.facet.normalized_text,
                "premise": prepared_corpus.corpus_units[
                    ordinal
                ].linearized_text,
            }
            for ordinal in nli_ordinals
        )
        scores = _score_nli(nli_backend, pairs)
        score_by_ordinal = dict(zip(nli_ordinals, scores))
        exact_set = set(exact)
        semantic: list[int] = []
        direct: list[int] = []
        for ordinal, similarity in enumerate(similarities):
            exact_strength = INTEGER_SCALE if ordinal in exact_set else 0
            nli_margin = score_by_ordinal.get(ordinal)
            semantic.append(
                max(
                    similarity,
                    exact_strength,
                    nli_margin if nli_margin is not None else -(2**63),
                )
            )
            direct.append(
                max(
                    exact_strength,
                    max(0, nli_margin) if nli_margin is not None else 0,
                )
            )
        semantic_rows.append(tuple(semantic))
        direct_rows.append(tuple(direct))
        total_nli_pairs += len(pairs)
        shortlist_receipts.append(
            {
                "facet_i": facet_index,
                "facet_type": meta.facet.facet_type,
                "MiniLM_top32_count": len(top32),
                "MiniLM_top32_ordinals_sha256": stable_hash(list(top32)),
                "exact_match_count": len(exact),
                "exact_match_ordinals_sha256": stable_hash(exact),
                "NLI_union_count": len(nli_ordinals),
                "NLI_union_ordinals_sha256": stable_hash(list(nli_ordinals)),
                "NLI_margin_vector_sha256": stable_hash(list(scores)),
            }
        )

    query_sha256 = hashlib.sha256(claim.encode("utf-8")).hexdigest()
    tensor = make_query_semantic_tensor(
        query_sha256=query_sha256,
        facets=facets,
        semantic_coverage_ints=tuple(semantic_rows),
        direct_anchor_strength_ints=tuple(direct_rows),
        dense_relevance_ints=dense_relevance,
    )
    receipt = _self_hashed(
        {
            "schema": f"{VERSION}_query_receipt",
            "version": VERSION,
            "design_sha256": DESIGN_SHA256,
            "query_sha256": query_sha256,
            "corpus_size": CORPUS_SIZE,
            "preparation_receipt_sha256": (
                prepared_corpus.preparation_receipt_sha256
            ),
            "preparation_mode": preparation_mode,
            "prepared_corpus_reused": True,
            "corpus_commitment_sha256": prepared_corpus.receipt[
                "corpus_commitment_sha256"
            ],
            "corpus_embedding_sha256": prepared_corpus.receipt[
                "corpus_embedding_sha256"
            ],
            "graph_sha256": prepared_corpus.graph.graph_sha256,
            "facet_count": len(facets),
            "facet_schema_sha256": stable_hash(
                [
                    [facet.facet_i, facet.facet_type, facet.normalized_text]
                    for facet in facets
                ]
            ),
            "MiniLM_binding_sha256": _binding_sha256(bindings["MiniLM"]),
            "MiniLM_asset_sha256": bindings["MiniLM"].asset_sha256,
            "NER_binding_sha256": _binding_sha256(bindings["NER"]),
            "NER_asset_sha256": bindings["NER"].asset_sha256,
            "NLI_binding_sha256": _binding_sha256(bindings["NLI"]),
            "NLI_asset_sha256": bindings["NLI"].asset_sha256,
            "MiniLM_call_count": 1,
            "MiniLM_encoded_text_count": len(embedding_texts),
            "MiniLM_similarity_count": (1 + len(facets)) * CORPUS_SIZE,
            "MiniLM_quantization": (
                "Qasper_binary64_products_math_fsum_Python_round_"
                "ties_to_even_scale_1000000"
            ),
            "full_corpus_scan": True,
            "NLI_shortlist_policy": "per_facet_MiniLM_top32_union_exact_entity_or_numeric",
            "NLI_pair_count": total_nli_pairs,
            "shortlists": shortlist_receipts,
            "semantic_combination": SEMANTIC_COMBINATION,
            "claim_NER_text_count": 1,
            "claim_NER_call_count": 1,
            "corpus_MiniLM_calls_in_query": 0,
            "corpus_NER_calls_in_query": 0,
            "tensor_sha256": tensor.tensor_sha256,
            "atomic_unit_count": len(prepared_corpus.atomic_units),
            "input_capability": (
                "claim_text_plus_prepared_fixed_linearized_corpus_only"
            ),
            "labels_family_gold_evidence_or_Hippo_accessed": False,
            "RAW_or_Hippo_candidates_consumed": 0,
            "network_calls": 0,
            "online_evaluator_calls": 0,
            "raw_claim_or_corpus_text_persisted": False,
        }
    )
    verify_semantic_receipt(receipt)
    return SemanticTensorBuild(
        atomic_units=prepared_corpus.atomic_units,
        graph=prepared_corpus.graph,
        tensor=tensor,
        receipt=MappingProxyType(receipt),
    )


def build_prepared_offline_semantic_tensor(
    *,
    claim_text: str,
    prepared_corpus: PreparedSemanticCorpus,
    minilm_backend: MiniLMBackend,
    ner_backend: NERBackend,
    nli_backend: NLIBackend,
    allow_synthetic_backends: bool = False,
) -> SemanticTensorBuild:
    """Formal per-query path; corpus NER, embedding, and graph are reused."""

    return _build_from_prepared(
        claim_text=claim_text,
        prepared_corpus=prepared_corpus,
        minilm_backend=minilm_backend,
        ner_backend=ner_backend,
        nli_backend=nli_backend,
        allow_synthetic_backends=allow_synthetic_backends,
        preparation_mode="precomputed_formal_path",
    )


def build_offline_semantic_tensor(
    *,
    claim_text: str,
    corpus_units: Sequence[SemanticCorpusUnit],
    minilm_backend: MiniLMBackend,
    ner_backend: NERBackend,
    nli_backend: NLIBackend,
    allow_synthetic_backends: bool = False,
) -> SemanticTensorBuild:
    """Compatibility wrapper that prepares inline; formal runs use two phases."""

    prepared = prepare_semantic_corpus(
        corpus_units=corpus_units,
        minilm_backend=minilm_backend,
        ner_backend=ner_backend,
        allow_synthetic_backends=allow_synthetic_backends,
    )
    return _build_from_prepared(
        claim_text=claim_text,
        prepared_corpus=prepared,
        minilm_backend=minilm_backend,
        ner_backend=ner_backend,
        nli_backend=nli_backend,
        allow_synthetic_backends=allow_synthetic_backends,
        preparation_mode="inline_compatibility_wrapper",
    )


__all__ = [
    "BINDING_VERSION",
    "BackendBinding",
    "BoundMiniLMBackend",
    "BoundNERBackend",
    "BoundNLIBackend",
    "CORPUS_SIZE",
    "DESIGN_SHA256",
    "DetectedEntity",
    "FeverousSemanticTensorError",
    "MINILM_TOP_K",
    "MiniLMBackend",
    "NERBackend",
    "NLIBackend",
    "PreparedSemanticCorpus",
    "SEMANTIC_COMBINATION",
    "SemanticCorpusUnit",
    "SemanticTensorBuild",
    "VERSION",
    "build_offline_semantic_tensor",
    "build_prepared_offline_semantic_tensor",
    "extract_claim_facets",
    "make_synthetic_backend_binding",
    "make_verified_backend_binding",
    "normalize_surface_text",
    "prepare_semantic_corpus",
    "quantized_minilm_similarity",
    "stable_hash",
    "verify_semantic_receipt",
    "verify_preparation_receipt",
    "verify_prepared_semantic_corpus",
]
