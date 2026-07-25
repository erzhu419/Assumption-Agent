"""Source-free local model execution and HippoRAG contracts for MMQA P1.

The only benchmark-bearing input accepted by :func:`execute_local_actions` is
an already validated :class:`mmqa_p1_action_integration_v1.AnonymousWorkItem`.
Model paths and verified identities are infrastructure bindings; local batch
functions are injected by the caller and receive explicit offline-only flags.
This module performs no source read, filesystem discovery, network/API call,
retry, qid/family access, or gold/answer/support handling.

One MiniLM batch produces the question and unit embeddings.  Cosines are
computed here and mapped from [-1, 1] to [0, 1].  One cross-encoder batch
produces unit logits, transformed by a stable sigmoid.  A minimal frozen text
parser supplies deterministic entity, relation-bigram, and numeric/temporal
anchor flags before delegating action formation to the integration layer.

The candidate-restricted HippoRAG surface only builds and validates an
anonymous worker payload/terminal.  It thinly reuses the ERASER exact-text
quotient and canonical ordinal-only terminal contract and never launches the
official model itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import re
from typing import Callable, Mapping, Sequence
import unicodedata

from assumption_agent.benchmarks.eraser_evidence_inference_official_hipporag_v1 import (
    contract as eraser_hippo,
)

from . import mmqa_p1_action_integration_v1 as integration
from . import mmqa_p1_typed_proof_e5_core_v1 as core


VERSION = "mmqa_p1_local_action_executor_v1"
STUDY_ID = core.STUDY_ID
STUDY_DESIGN_SELF_SHA256 = integration.STUDY_DESIGN_SELF_SHA256

MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MINILM_REQUIRED_TREE_SHA256 = (
    "1514beb65d2d3a2824a93f133a300cc60d5b437ccd6ea1e622eb4cd9881dcfdb"
)
CROSS_ENCODER_REQUIRED_TREE_SHA256 = (
    "923d4371d5fe13534d7431895890c2142a8552a441f09ec7b28d035aaae9120c"
)
MINILM_EMBEDDING_DIMENSION = 384
MINILM_BATCH_SIZE = 32
MINILM_MAX_LENGTH = 256
CROSS_ENCODER_BATCH_SIZE = 64
CROSS_ENCODER_MAX_LENGTH = 512
ANCHOR_PARSER_VERSION = "mmqa_exact_surface_entity_relation_numeric_v1"

HIPPORAG_PAYLOAD_SCHEMA = f"{VERSION}_candidate_restricted_hipporag_payload"
HIPPORAG_TERMINAL_SCHEMA = f"{VERSION}_candidate_restricted_hipporag_terminal"
EXECUTION_RECEIPT_SCHEMA = f"{VERSION}_receipt"

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_WORD = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", flags=re.UNICODE)
_NUMBER = re.compile(
    r"(?<!\w)[+-]?\d+(?:[.,:/-]\d+)*(?:%|[A-Za-z]+)?(?!\w)",
    flags=re.UNICODE,
)


class MmqaP1LocalActionExecutorError(RuntimeError):
    """A frozen local model, feature, payload, or terminal contract drifted."""


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise MmqaP1LocalActionExecutorError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _absolute_lexical_path(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or value == "/"
        or value.endswith("/")
        or "\x00" in value
        or "//" in value
        or "/./" in value
        or "/../" in value
        or value.endswith("/.")
        or value.endswith("/..")
    ):
        raise MmqaP1LocalActionExecutorError(
            f"{field} must be one normalized absolute lexical path"
        )
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MmqaP1LocalActionExecutorError(
            "executor value is not canonical JSON"
        ) from exc


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class FrozenLocalModelBinding:
    """Caller-verified local paths and identities; no discovery occurs here."""

    minilm_model_path: str
    minilm_required_tree_sha256: str
    cross_encoder_model_path: str
    cross_encoder_required_tree_sha256: str
    local_runtime_identity_sha256: str
    asset_identity_verified: bool = True
    local_files_only: bool = True
    trust_remote_code: bool = False
    network_disabled: bool = True
    retry_count: int = 0

    def __post_init__(self) -> None:
        minilm_path = _absolute_lexical_path(
            self.minilm_model_path, "MiniLM model path"
        )
        ce_path = _absolute_lexical_path(
            self.cross_encoder_model_path, "cross-encoder model path"
        )
        if self.minilm_required_tree_sha256 != MINILM_REQUIRED_TREE_SHA256:
            raise MmqaP1LocalActionExecutorError(
                "MiniLM required tree identity drifted"
            )
        if (
            self.cross_encoder_required_tree_sha256
            != CROSS_ENCODER_REQUIRED_TREE_SHA256
        ):
            raise MmqaP1LocalActionExecutorError(
                "cross-encoder required tree identity drifted"
            )
        _sha256(self.local_runtime_identity_sha256, "local runtime identity")
        if (
            self.asset_identity_verified is not True
            or self.local_files_only is not True
            or self.trust_remote_code is not False
            or self.network_disabled is not True
            or type(self.retry_count) is not int
            or self.retry_count != 0
        ):
            raise MmqaP1LocalActionExecutorError(
                "local/offline model binding policy drifted"
            )
        object.__setattr__(self, "minilm_model_path", minilm_path)
        object.__setattr__(self, "cross_encoder_model_path", ce_path)

    def public_binding(self) -> dict[str, object]:
        return {
            "minilm_model_id": MINILM_MODEL_ID,
            "minilm_model_path_sha256": hashlib.sha256(
                self.minilm_model_path.encode("utf-8")
            ).hexdigest(),
            "minilm_required_tree_sha256": self.minilm_required_tree_sha256,
            "cross_encoder_model_id": CROSS_ENCODER_MODEL_ID,
            "cross_encoder_model_path_sha256": hashlib.sha256(
                self.cross_encoder_model_path.encode("utf-8")
            ).hexdigest(),
            "cross_encoder_required_tree_sha256": (
                self.cross_encoder_required_tree_sha256
            ),
            "local_runtime_identity_sha256": self.local_runtime_identity_sha256,
            "asset_identity_verified": True,
            "local_files_only": True,
            "trust_remote_code": False,
            "network_disabled": True,
            "retry_count": 0,
        }


@dataclass(frozen=True)
class LocalBatchFunctions:
    """Injected local-only batch calls; neither function may perform retries."""

    encode_minilm: Callable[..., object]
    score_cross_encoder: Callable[..., object]

    def __post_init__(self) -> None:
        if not callable(self.encode_minilm) or not callable(
            self.score_cross_encoder
        ):
            raise MmqaP1LocalActionExecutorError(
                "local batch functions must be callable"
            )


def _rows(value: object, field: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)):
        raise MmqaP1LocalActionExecutorError(f"{field} is not an array")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise MmqaP1LocalActionExecutorError(f"{field} is not iterable") from exc


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MmqaP1LocalActionExecutorError(f"{field} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result):
        raise MmqaP1LocalActionExecutorError(f"{field} must be a finite scalar")
    return 0.0 if result == 0.0 else result


def _validated_embedding_matrix(
    value: object, expected_rows: int
) -> tuple[tuple[float, ...], ...]:
    raw_rows = _rows(value, "MiniLM embedding matrix")
    if len(raw_rows) != expected_rows:
        raise MmqaP1LocalActionExecutorError(
            "MiniLM embedding row count drifted"
        )
    matrix = []
    for row_index, raw in enumerate(raw_rows):
        coordinates = _rows(raw, f"MiniLM row {row_index}")
        if len(coordinates) != MINILM_EMBEDDING_DIMENSION:
            raise MmqaP1LocalActionExecutorError(
                "MiniLM embedding dimension drifted"
            )
        vector = tuple(
            _finite(value, f"MiniLM row {row_index} coordinate")
            for value in coordinates
        )
        norm = math.sqrt(math.fsum(value * value for value in vector))
        if not math.isfinite(norm) or norm <= 0.0:
            raise MmqaP1LocalActionExecutorError(
                "MiniLM embedding has zero or nonfinite norm"
            )
        matrix.append(vector)
    return tuple(matrix)


def _cosine_scores(
    matrix: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    question = tuple(matrix[0])
    question_norm = math.sqrt(math.fsum(value * value for value in question))
    output = []
    for row in matrix[1:]:
        row_tuple = tuple(row)
        row_norm = math.sqrt(math.fsum(value * value for value in row_tuple))
        cosine = math.fsum(
            left * right for left, right in zip(question, row_tuple, strict=True)
        ) / (question_norm * row_norm)
        if not math.isfinite(cosine) or not -1.0 - 1.0e-12 <= cosine <= 1.0 + 1.0e-12:
            raise MmqaP1LocalActionExecutorError(
                "MiniLM cosine escaped its mathematical range"
            )
        cosine = min(1.0, max(-1.0, cosine))
        output.append((cosine + 1.0) / 2.0)
    return tuple(output)


def _stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _validated_ce_scores(value: object, expected_rows: int) -> tuple[float, ...]:
    logits = _rows(value, "cross-encoder logits")
    if len(logits) != expected_rows:
        raise MmqaP1LocalActionExecutorError(
            "cross-encoder logit count drifted"
        )
    return tuple(
        _stable_sigmoid(_finite(row, "cross-encoder logit")) for row in logits
    )


def _surface_words(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value)
    return tuple(match.group(0) for match in _WORD.finditer(normalized))


def deterministic_anchor_flags(
    question: str, serialized_content: str
) -> tuple[int, int, int]:
    """Return fixed surface-only entity, relation-bigram, and numeric flags.

    Entity evidence is exact case-insensitive overlap with a question token
    containing an uppercase character.  Relation evidence is exact overlap of
    an adjacent casefolded, nonnumeric question/unit token bigram.  Numeric or
    temporal evidence is the presence of a structured number/date token in the
    unit.  No learned model, family registry, answer, support, threshold search,
    or per-source keyword list participates.
    """

    if not isinstance(question, str) or not isinstance(serialized_content, str):
        raise MmqaP1LocalActionExecutorError(
            "anchor parser accepts exact question/content text only"
        )
    question_words = _surface_words(question)
    unit_words = _surface_words(serialized_content)
    question_entities = {
        word.casefold()
        for word in question_words
        if any(character.isupper() for character in word)
    }
    unit_folded = {word.casefold() for word in unit_words}
    entity = int(bool(question_entities.intersection(unit_folded)))

    question_relation_words = tuple(
        word.casefold()
        for word in question_words
        if _NUMBER.fullmatch(word) is None
    )
    unit_relation_words = tuple(
        word.casefold() for word in unit_words if _NUMBER.fullmatch(word) is None
    )
    question_bigrams = set(
        zip(question_relation_words, question_relation_words[1:])
    )
    unit_bigrams = set(zip(unit_relation_words, unit_relation_words[1:]))
    relation = int(bool(question_bigrams.intersection(unit_bigrams)))
    numeric_or_temporal = int(_NUMBER.search(serialized_content) is not None)
    return entity, relation, numeric_or_temporal


@dataclass(frozen=True)
class LocalActionExecutionReceipt:
    anonymous_projection_sha256: str
    local_model_binding_sha256: str
    minilm_score_vector_sha256: str
    cross_encoder_score_vector_sha256: str
    anchor_flag_vector_sha256: str
    unit_count: int

    def payload(self) -> dict[str, object]:
        return {
            "schema": EXECUTION_RECEIPT_SCHEMA,
            "study_id": STUDY_ID,
            "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
            "anonymous_projection_sha256": self.anonymous_projection_sha256,
            "local_model_binding_sha256": self.local_model_binding_sha256,
            "minilm_score_vector_sha256": self.minilm_score_vector_sha256,
            "cross_encoder_score_vector_sha256": (
                self.cross_encoder_score_vector_sha256
            ),
            "anchor_flag_vector_sha256": self.anchor_flag_vector_sha256,
            "anchor_parser_version": ANCHOR_PARSER_VERSION,
            "unit_count": self.unit_count,
            "minilm_batch_call_count": 1,
            "cross_encoder_batch_call_count": 1,
            "model_call_count": 2,
            "source_reader_call_count": 0,
            "gold_answer_support_family_qid_read_count": 0,
            "network_or_api_call_count": 0,
            "retry_replay_resample_count": 0,
        }


@dataclass(frozen=True)
class LocalActionExecution:
    actions: integration.IntegratedActions
    receipt: LocalActionExecutionReceipt


def _validated_unit_score_vector(
    value: object, expected_rows: int, field: str
) -> tuple[float, ...]:
    rows = _rows(value, field)
    if len(rows) != expected_rows:
        raise MmqaP1LocalActionExecutorError(f"{field} count drifted")
    output = tuple(_finite(row, field) for row in rows)
    if any(not 0.0 <= row <= 1.0 for row in output):
        raise MmqaP1LocalActionExecutorError(f"{field} escaped [0, 1]")
    return output


def form_actions_from_local_coordinate_vectors(
    work_item: integration.AnonymousWorkItem,
    minilm_scores: object,
    cross_encoder_scores: object,
    *,
    e5_model: core.E5Model | None = None,
) -> integration.IntegratedActions:
    """Form actions from complete, already-computed anonymous score vectors."""

    if not isinstance(work_item, integration.AnonymousWorkItem):
        raise MmqaP1LocalActionExecutorError(
            "coordinate merge benchmark input must be AnonymousWorkItem only"
        )
    units = work_item.units
    minilm = _validated_unit_score_vector(
        minilm_scores, len(units), "MiniLM score vector"
    )
    cross_encoder = _validated_unit_score_vector(
        cross_encoder_scores, len(units), "cross-encoder score vector"
    )
    anchor_flags = tuple(
        deterministic_anchor_flags(work_item.question, unit.serialized_content)
        for unit in units
    )
    coordinates = tuple(
        integration.UnitCoordinates(
            ordinal=unit.ordinal,
            minilm_similarity=minilm[index],
            cross_encoder_relevance=cross_encoder[index],
            entity_anchor=anchor_flags[index][0],
            relation_anchor=anchor_flags[index][1],
            numeric_or_temporal_anchor=anchor_flags[index][2],
        )
        for index, unit in enumerate(units)
    )
    try:
        return integration.form_actions(
            work_item, coordinates, e5_model=e5_model
        )
    except (integration.MmqaP1ActionIntegrationError, core.MmqaP1CoreError) as exc:
        raise MmqaP1LocalActionExecutorError(
            "local coordinates could not form the frozen actions"
        ) from exc


def execute_local_actions(
    work_item: integration.AnonymousWorkItem,
    *,
    model_binding: FrozenLocalModelBinding,
    batch_functions: LocalBatchFunctions,
    e5_model: core.E5Model | None = None,
) -> LocalActionExecution:
    """Run exactly one local MiniLM batch and one local CE batch, then form actions."""

    if not isinstance(work_item, integration.AnonymousWorkItem):
        raise MmqaP1LocalActionExecutorError(
            "executor benchmark input must be AnonymousWorkItem only"
        )
    if not isinstance(model_binding, FrozenLocalModelBinding):
        raise MmqaP1LocalActionExecutorError(
            "executor requires a frozen local model binding"
        )
    if not isinstance(batch_functions, LocalBatchFunctions):
        raise MmqaP1LocalActionExecutorError(
            "executor requires injected local batch functions"
        )
    units = work_item.units
    unit_texts = tuple(unit.serialized_content for unit in units)
    embedding_texts = (work_item.question, *unit_texts)

    try:
        raw_embeddings = batch_functions.encode_minilm(
            model_path=model_binding.minilm_model_path,
            texts=embedding_texts,
            batch_size=MINILM_BATCH_SIZE,
            max_length=MINILM_MAX_LENGTH,
            normalize_embeddings=True,
            local_files_only=True,
            trust_remote_code=False,
            network_disabled=True,
            deterministic=True,
        )
    except Exception as exc:
        raise MmqaP1LocalActionExecutorError(
            "the single MiniLM batch failed"
        ) from exc
    matrix = _validated_embedding_matrix(raw_embeddings, 1 + len(units))
    minilm_scores = _cosine_scores(matrix)

    try:
        raw_logits = batch_functions.score_cross_encoder(
            model_path=model_binding.cross_encoder_model_path,
            question=work_item.question,
            documents=unit_texts,
            batch_size=CROSS_ENCODER_BATCH_SIZE,
            max_length=CROSS_ENCODER_MAX_LENGTH,
            local_files_only=True,
            trust_remote_code=False,
            network_disabled=True,
            deterministic=True,
        )
    except Exception as exc:
        raise MmqaP1LocalActionExecutorError(
            "the single cross-encoder batch failed"
        ) from exc
    ce_scores = _validated_ce_scores(raw_logits, len(units))
    anchor_flags = tuple(
        deterministic_anchor_flags(work_item.question, unit.serialized_content)
        for unit in units
    )
    actions = form_actions_from_local_coordinate_vectors(
        work_item,
        minilm_scores,
        ce_scores,
        e5_model=e5_model,
    )

    binding_payload = model_binding.public_binding()
    receipt = LocalActionExecutionReceipt(
        anonymous_projection_sha256=work_item.anonymous_projection_sha256,
        local_model_binding_sha256=_semantic_hash(binding_payload),
        minilm_score_vector_sha256=_semantic_hash(
            [value.hex() for value in minilm_scores]
        ),
        cross_encoder_score_vector_sha256=_semantic_hash(
            [value.hex() for value in ce_scores]
        ),
        anchor_flag_vector_sha256=_semantic_hash(
            [list(value) for value in anchor_flags]
        ),
        unit_count=len(units),
    )
    return LocalActionExecution(actions=actions, receipt=receipt)


@dataclass(frozen=True)
class CandidateRestrictedHippoRAGPayload:
    query: str
    logical_source_ordinals: tuple[int, ...]
    exact_sentence_texts: tuple[str, ...]
    closure_ordinal_bytes_sha256: str
    exact_text_quotient_count: int

    def __post_init__(self) -> None:
        try:
            query, texts = eraser_hippo.validate_single_item(
                self.query, self.exact_sentence_texts
            )
            quotient, _mapping = eraser_hippo.exact_text_quotient(texts)
        except eraser_hippo.EraserEvidenceInferenceOfficialHippoRAGError as exc:
            raise MmqaP1LocalActionExecutorError(
                "candidate-restricted HippoRAG payload drifted"
            ) from exc
        ordinals = tuple(self.logical_source_ordinals)
        if (
            len(ordinals) != len(texts)
            or tuple(sorted(ordinals)) != ordinals
            or len(set(ordinals)) != len(ordinals)
            or any(type(value) is not int or value < 0 for value in ordinals)
        ):
            raise MmqaP1LocalActionExecutorError(
                "HippoRAG logical source ordinals drifted"
            )
        _sha256(
            self.closure_ordinal_bytes_sha256,
            "HippoRAG closure ordinal bytes identity",
        )
        if self.exact_text_quotient_count != len(quotient):
            raise MmqaP1LocalActionExecutorError(
                "HippoRAG exact-text quotient count drifted"
            )
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "exact_sentence_texts", texts)

    def worker_payload(self) -> dict[str, object]:
        # Exact ERASER worker envelope: no ordinal, title, ID, label, or prefix
        # is injected into the documents indexed by the official core.
        return {
            "query": self.query,
            "schema": eraser_hippo.INPUT_SCHEMA,
            "sentence_texts": list(self.exact_sentence_texts),
        }

    def canonical_worker_bytes(self) -> bytes:
        return eraser_hippo.canonical_json_bytes(self.worker_payload())

    def anonymous_binding(self) -> dict[str, object]:
        return {
            "schema": HIPPORAG_PAYLOAD_SCHEMA,
            "study_id": STUDY_ID,
            "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
            "closure_ordinal_bytes_sha256": self.closure_ordinal_bytes_sha256,
            "logical_source_ordinals": list(self.logical_source_ordinals),
            "logical_document_count": len(self.exact_sentence_texts),
            "exact_text_quotient_count": self.exact_text_quotient_count,
            "worker_payload_sha256": hashlib.sha256(
                self.canonical_worker_bytes()
            ).hexdigest(),
            "fresh_isolated_index_required": True,
            "retrieve_all_exact_text_quotient_members_required": True,
            "network_disabled": True,
            "model_run_count_in_this_adapter": 0,
            "source_reader_call_count": 0,
            "retry_replay_resample_count": 0,
        }


def build_candidate_restricted_hipporag_payload(
    actions: integration.IntegratedActions,
) -> CandidateRestrictedHippoRAGPayload:
    """Build the exact anonymous official-worker input without running it."""

    if not isinstance(actions, integration.IntegratedActions):
        raise MmqaP1LocalActionExecutorError(
            "HippoRAG payload requires completed integrated actions"
        )
    shared = actions.shared_closure
    texts = tuple(unit.serialized_content for unit in shared.units)
    try:
        _query, validated = eraser_hippo.validate_single_item(
            actions.work_item.question, texts
        )
        quotient, _mapping = eraser_hippo.exact_text_quotient(validated)
    except eraser_hippo.EraserEvidenceInferenceOfficialHippoRAGError as exc:
        raise MmqaP1LocalActionExecutorError(
            "common closure cannot form the official HippoRAG payload"
        ) from exc
    return CandidateRestrictedHippoRAGPayload(
        query=actions.work_item.question,
        logical_source_ordinals=shared.ordinals,
        exact_sentence_texts=validated,
        closure_ordinal_bytes_sha256=shared.ordinal_bytes_sha256,
        exact_text_quotient_count=len(quotient),
    )


@dataclass(frozen=True)
class CandidateRestrictedHippoRAGTerminal:
    top5_source_ordinals: tuple[int, ...]
    worker_output_sha256: str
    closure_ordinal_bytes_sha256: str

    def payload(self) -> dict[str, object]:
        return {
            "schema": HIPPORAG_TERMINAL_SCHEMA,
            "study_id": STUDY_ID,
            "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
            "top5_source_ordinals": list(self.top5_source_ordinals),
            "worker_output_sha256": self.worker_output_sha256,
            "closure_ordinal_bytes_sha256": self.closure_ordinal_bytes_sha256,
            "worker_output_contract": eraser_hippo.OUTPUT_SCHEMA,
            "model_run_count_in_this_adapter": 0,
            "source_reader_call_count": 0,
            "network_or_api_call_count": 0,
            "retry_replay_resample_count": 0,
        }


def parse_candidate_restricted_hipporag_terminal(
    payload: CandidateRestrictedHippoRAGPayload,
    raw_worker_output: bytes,
) -> CandidateRestrictedHippoRAGTerminal:
    """Validate canonical logical positions and map them to source ordinals."""

    if not isinstance(payload, CandidateRestrictedHippoRAGPayload):
        raise MmqaP1LocalActionExecutorError(
            "HippoRAG terminal requires its frozen anonymous payload"
        )
    if not isinstance(raw_worker_output, bytes):
        raise MmqaP1LocalActionExecutorError(
            "HippoRAG worker terminal must be exact bytes"
        )
    try:
        logical_positions = eraser_hippo.parse_ordinals_only_output(
            raw_worker_output,
            logical_sentence_count=len(payload.logical_source_ordinals),
        )
    except eraser_hippo.EraserEvidenceInferenceOfficialHippoRAGError as exc:
        raise MmqaP1LocalActionExecutorError(
            "HippoRAG worker terminal drifted"
        ) from exc
    top5 = tuple(
        payload.logical_source_ordinals[position] for position in logical_positions
    )
    if len(top5) != core.TOP_K or len(set(top5)) != core.TOP_K:
        raise MmqaP1LocalActionExecutorError(
            "HippoRAG terminal did not map to five unique closure ordinals"
        )
    return CandidateRestrictedHippoRAGTerminal(
        top5_source_ordinals=top5,
        worker_output_sha256=hashlib.sha256(raw_worker_output).hexdigest(),
        closure_ordinal_bytes_sha256=payload.closure_ordinal_bytes_sha256,
    )


__all__ = [
    "VERSION",
    "STUDY_ID",
    "STUDY_DESIGN_SELF_SHA256",
    "MINILM_MODEL_ID",
    "CROSS_ENCODER_MODEL_ID",
    "MINILM_REQUIRED_TREE_SHA256",
    "CROSS_ENCODER_REQUIRED_TREE_SHA256",
    "MINILM_EMBEDDING_DIMENSION",
    "MINILM_BATCH_SIZE",
    "MINILM_MAX_LENGTH",
    "CROSS_ENCODER_BATCH_SIZE",
    "CROSS_ENCODER_MAX_LENGTH",
    "ANCHOR_PARSER_VERSION",
    "HIPPORAG_PAYLOAD_SCHEMA",
    "HIPPORAG_TERMINAL_SCHEMA",
    "EXECUTION_RECEIPT_SCHEMA",
    "MmqaP1LocalActionExecutorError",
    "FrozenLocalModelBinding",
    "LocalBatchFunctions",
    "LocalActionExecutionReceipt",
    "LocalActionExecution",
    "CandidateRestrictedHippoRAGPayload",
    "CandidateRestrictedHippoRAGTerminal",
    "deterministic_anchor_flags",
    "form_actions_from_local_coordinate_vectors",
    "execute_local_actions",
    "build_candidate_restricted_hipporag_payload",
    "parse_candidate_restricted_hipporag_terminal",
]
