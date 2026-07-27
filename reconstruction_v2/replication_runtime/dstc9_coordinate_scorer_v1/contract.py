"""Source-free contract for the DSTC9 six-coordinate scorer.

The public input contains only the frozen typed-core projection: exactly 2,900
``ordinal/entity_name/title/body`` snippets and 1..256 ordered histories made
from ``U``/``S`` text plus an opaque work id.  There is no field for domain,
family, entity/document ids, qrels, labels, responses, scores, or evaluators.

The output is deliberately private.  It contains six bounded integer vectors
of width 2,900 per history plus content-free hashes and execution counts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks import dstc9_p1_typed_core_v1 as typed_core


VERSION = "dstc9_coordinate_scorer_v1"
STUDY_ID = "DSTC9_P1_HIERARCHICAL_KNOWLEDGE_EVALUATOR_L5_V1"
TYPED_CORE_SHA256 = (
    "a8290586595922e074e0a1aff52fd0d3eee396d0f1d366ccfc8407a5db65aa32"
)
TYPED_CORE_RELATIVE_PATH = (
    "assumption_agent/benchmarks/dstc9_p1_typed_core_v1.py"
)

CORPUS_SIZE = 2900
MIN_QUERY_COUNT = 1
MAX_QUERY_COUNT = 256
SCORE_SCALE = typed_core.SCALE
MAX_SCORE_ABS = typed_core.MAX_SCORE_ABS
SCORE_NAMES = typed_core.SCORE_NAMES
PHYSICAL_GPU = 1
CUDA_VISIBLE_DEVICES = "1"
LOGICAL_CUDA_DEVICE = "cuda:0"

INPUT_SCHEMA = f"{VERSION}_source_free_input_v1"
OUTPUT_SCHEMA = f"{VERSION}_private_output_v1"
RECEIPT_SCHEMA = f"{VERSION}_private_receipt_v1"
MODEL_BINDING_SCHEMA = f"{VERSION}_model_binding_v1"
SNIPPET_SERIALIZATION = "frozen_dstc9_typed_core_serialize_passage_v1"
QUERY_SERIALIZATION = "frozen_dstc9_typed_core_serialize_model_query_v1"
ENTITY_NONE_SERIALIZATION = "ENTITY: <NONE>"
QUANTIZATION = "finite_float64_rint_times_1000000_bounded_int_v1"
TRANSPORT = "systemd_run_user_transient_service_v1"
SYSTEMD_NETWORK_PROPERTIES = (
    "IPAddressDeny=any",
    "RestrictAddressFamilies=AF_UNIX",
)

SNIPPET_KEYS = frozenset({"body", "entity_name", "ordinal", "title"})
HISTORY_KEYS = frozenset({"turns", "work_id"})
TURN_KEYS = frozenset({"speaker", "text"})
INPUT_KEYS = frozenset(
    {
        "histories",
        "history_projection_sha256",
        "model_query_sha256",
        "passage_serialization_sha256",
        "schema",
        "self_sha256",
        "snippet_projection_sha256",
        "snippets",
        "study_id",
        "typed_core_sha256",
    }
)
OUTPUT_ROW_KEYS = frozenset({"query_ordinal", "vectors", "work_id"})
OUTPUT_KEYS = frozenset(
    {
        "corpus_count",
        "input_self_sha256",
        "query_count",
        "receipt",
        "rows",
        "schema",
        "self_sha256",
        "study_id",
        "typed_core_sha256",
    }
)
RECEIPT_KEYS = frozenset(
    {
        "corpus_count",
        "cross_encoder_call_count",
        "cross_encoder_model_load_count",
        "cross_encoder_pair_count",
        "cuda_visible_devices",
        "dynamic_resize_count",
        "history_projection_sha256",
        "input_self_sha256",
        "logical_cuda_device",
        "minilm_encode_call_count",
        "minilm_model_load_count",
        "minilm_text_count",
        "model_binding_sha256",
        "model_query_sha256",
        "network_access",
        "passage_serialization_sha256",
        "physical_gpu",
        "quantization",
        "query_count",
        "query_serialization",
        "receipt_sha256",
        "retry_count",
        "schema",
        "score_bundle_sha256",
        "score_names",
        "snippet_projection_sha256",
        "snippet_serialization",
        "status",
        "study_id",
        "typed_core_sha256",
        "work_id_order_sha256",
    }
)

FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "answer",
        "answers",
        "doc_id",
        "document_id",
        "domain",
        "domain_id",
        "entity_id",
        "evaluator",
        "families",
        "family",
        "gold",
        "label",
        "labels",
        "metric",
        "qrel",
        "qrels",
        "relevance",
        "response",
        "score",
        "scores",
        "split",
        "target",
        "utility",
    }
)

WORKER_ENVIRONMENT_KEYS = frozenset(
    {
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "LANG",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TOKENIZERS_PARALLELISM",
        "TRANSFORMERS_OFFLINE",
    }
)
WORKER_FIXED_ENVIRONMENT_VALUES = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
    "HF_HUB_OFFLINE": "1",
    "LANG": "C.UTF-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
}

_HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")
_OPAQUE_WORK_ID_RE = re.compile(r"[^\x00\r\n]{1,512}\Z")


class Dstc9CoordinateScorerError(RuntimeError):
    """The source-free scorer contract or private output drifted."""


@dataclass(frozen=True, slots=True)
class HistoryItem:
    work_id: str
    turns: tuple[typed_core.DialogueTurn, ...]


@dataclass(frozen=True, slots=True)
class ScorerInput:
    snippets: tuple[typed_core.KnowledgeSnippet, ...]
    histories: tuple[HistoryItem, ...]
    snippet_projection_sha256: str
    history_projection_sha256: str
    model_query_sha256: str
    passage_serialization_sha256: str
    self_sha256: str


def canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    """Delegate canonical JSON encoding to the frozen typed core."""

    try:
        return typed_core.canonical_bytes(value, newline=newline)
    except typed_core.Dstc9P1TypedCoreError as exc:
        raise Dstc9CoordinateScorerError(
            "value is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    """Delegate SHA-256 projection hashing to the frozen typed core."""

    try:
        return typed_core.stable_hash(value)
    except typed_core.Dstc9P1TypedCoreError as exc:
        raise Dstc9CoordinateScorerError("value cannot be hashed") from exc


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise Dstc9CoordinateScorerError(
            f"{field} is not a lowercase SHA-256"
        )
    return value


def _work_id(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or _OPAQUE_WORK_ID_RE.fullmatch(value) is None
        or not value.strip()
    ):
        raise Dstc9CoordinateScorerError(f"{field} is not opaque text")
    return value


def _reject_forbidden_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if (
                not isinstance(key, str)
                or key.casefold() in FORBIDDEN_INPUT_KEYS
            ):
                raise Dstc9CoordinateScorerError(
                    "input contains a forbidden source/label/score field"
                )
            _reject_forbidden_keys(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _reject_forbidden_keys(nested)


def _with_self_hash(body: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in body:
        raise Dstc9CoordinateScorerError("self hash was supplied twice")
    result = dict(body)
    result["self_sha256"] = stable_hash(result)
    return result


def _verify_self_hash(value: Mapping[str, object], field: str) -> str:
    body = dict(value)
    claimed = _sha256(body.pop("self_sha256", None), f"{field} self hash")
    if stable_hash(body) != claimed:
        raise Dstc9CoordinateScorerError(f"{field} self hash drifted")
    return claimed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_typed_core_binding(project_root: Path) -> Path:
    """Verify both the frozen file bytes and imported semantic identity."""

    if not isinstance(project_root, Path) or not project_root.is_absolute():
        raise Dstc9CoordinateScorerError(
            "project root must be an absolute path"
        )
    path = project_root / TYPED_CORE_RELATIVE_PATH
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
        or _sha256_file(path) != TYPED_CORE_SHA256
        or typed_core.STUDY_ID != STUDY_ID
        or typed_core.VERSION != "dstc9_p1_typed_core_v1"
        or typed_core.SCORE_NAMES != SCORE_NAMES
    ):
        raise Dstc9CoordinateScorerError(
            "frozen typed-core binding drifted"
        )
    return path


def _snippet_projection(
    snippets: Sequence[typed_core.KnowledgeSnippet],
) -> list[dict[str, object]]:
    return [typed_core.snippet_public_payload(row) for row in snippets]


def _history_projection(
    histories: Sequence[HistoryItem],
) -> list[dict[str, object]]:
    return [
        {
            "turns": [
                typed_core.turn_public_payload(turn) for turn in row.turns
            ],
            "work_id": row.work_id,
        }
        for row in histories
    ]


def serialize_model_queries(histories: Sequence[HistoryItem]) -> tuple[str, ...]:
    try:
        return tuple(
            typed_core.serialize_model_query(row.turns) for row in histories
        )
    except typed_core.Dstc9P1TypedCoreError as exc:
        raise Dstc9CoordinateScorerError(
            "model-query serialization failed"
        ) from exc


def serialize_passages(
    snippets: Sequence[typed_core.KnowledgeSnippet],
) -> tuple[str, ...]:
    try:
        return tuple(typed_core.serialize_passage(row) for row in snippets)
    except typed_core.Dstc9P1TypedCoreError as exc:
        raise Dstc9CoordinateScorerError(
            "passage serialization failed"
        ) from exc


def serialize_entity_fields(
    snippets: Sequence[typed_core.KnowledgeSnippet],
) -> tuple[str, ...]:
    return tuple(
        row.entity_name
        if row.entity_name is not None
        else ENTITY_NONE_SERIALIZATION
        for row in snippets
    )


def _parse_snippets(value: object) -> tuple[typed_core.KnowledgeSnippet, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != CORPUS_SIZE
    ):
        raise Dstc9CoordinateScorerError(
            "snippet corpus must contain exactly 2900 rows"
        )
    rows: list[typed_core.KnowledgeSnippet] = []
    for position, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != SNIPPET_KEYS:
            raise Dstc9CoordinateScorerError(
                "snippet must contain exact public fields"
            )
        try:
            row = typed_core.snippet_from_public_fields(raw)
        except typed_core.Dstc9P1TypedCoreError as exc:
            raise Dstc9CoordinateScorerError(
                "typed-core snippet validation failed"
            ) from exc
        if row.ordinal != position:
            raise Dstc9CoordinateScorerError(
                "snippet ordinals must be contiguous zero-based order"
            )
        if dict(raw) != typed_core.snippet_public_payload(row):
            raise Dstc9CoordinateScorerError(
                "snippet is not the canonical typed-core projection"
            )
        rows.append(row)
    return tuple(rows)


def _parse_histories(value: object) -> tuple[HistoryItem, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not MIN_QUERY_COUNT <= len(value) <= MAX_QUERY_COUNT
    ):
        raise Dstc9CoordinateScorerError(
            "history count is outside the frozen 1..256 bound"
        )
    rows: list[HistoryItem] = []
    seen: set[str] = set()
    for position, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != HISTORY_KEYS:
            raise Dstc9CoordinateScorerError(
                "history must contain only turns and opaque work_id"
            )
        work_id = _work_id(raw.get("work_id"), f"histories[{position}].work_id")
        if work_id in seen:
            raise Dstc9CoordinateScorerError("work_id values must be unique")
        seen.add(work_id)
        raw_turns = raw.get("turns")
        if (
            isinstance(raw_turns, (str, bytes))
            or not isinstance(raw_turns, Sequence)
            or not raw_turns
        ):
            raise Dstc9CoordinateScorerError("history turns are malformed")
        turns: list[typed_core.DialogueTurn] = []
        for raw_turn in raw_turns:
            if not isinstance(raw_turn, Mapping) or set(raw_turn) != TURN_KEYS:
                raise Dstc9CoordinateScorerError(
                    "turn must contain only U/S speaker and text"
                )
            try:
                turn = typed_core.turn_from_public_fields(raw_turn)
            except typed_core.Dstc9P1TypedCoreError as exc:
                raise Dstc9CoordinateScorerError(
                    "typed-core turn validation failed"
                ) from exc
            if dict(raw_turn) != typed_core.turn_public_payload(turn):
                raise Dstc9CoordinateScorerError(
                    "turn is not the canonical typed-core projection"
                )
            turns.append(turn)
        try:
            typed_core.serialize_model_query(tuple(turns))
        except typed_core.Dstc9P1TypedCoreError as exc:
            raise Dstc9CoordinateScorerError(
                "typed-core history validation failed"
            ) from exc
        rows.append(HistoryItem(work_id=work_id, turns=tuple(turns)))
    return tuple(rows)


def input_payload(
    *,
    snippets: Sequence[Mapping[str, object]],
    histories: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Normalize through the frozen typed core and make a self-hashed input."""

    parsed_snippets: list[typed_core.KnowledgeSnippet] = []
    for raw in snippets:
        try:
            parsed_snippets.append(typed_core.snippet_from_public_fields(raw))
        except typed_core.Dstc9P1TypedCoreError as exc:
            raise Dstc9CoordinateScorerError(
                "typed-core snippet validation failed"
            ) from exc
    parsed_histories: list[HistoryItem] = []
    for position, raw in enumerate(histories):
        if not isinstance(raw, Mapping) or set(raw) != HISTORY_KEYS:
            raise Dstc9CoordinateScorerError(
                "history must contain only turns and opaque work_id"
            )
        raw_turns = raw.get("turns")
        if isinstance(raw_turns, (str, bytes)) or not isinstance(
            raw_turns, Sequence
        ):
            raise Dstc9CoordinateScorerError("history turns are malformed")
        try:
            turns = tuple(
                typed_core.turn_from_public_fields(turn) for turn in raw_turns
            )
            typed_core.serialize_model_query(turns)
        except typed_core.Dstc9P1TypedCoreError as exc:
            raise Dstc9CoordinateScorerError(
                "typed-core history validation failed"
            ) from exc
        parsed_histories.append(
            HistoryItem(
                work_id=_work_id(
                    raw.get("work_id"), f"histories[{position}].work_id"
                ),
                turns=turns,
            )
        )
    snippet_rows = tuple(parsed_snippets)
    history_rows = tuple(parsed_histories)
    snippets_projection = _snippet_projection(snippet_rows)
    histories_projection = _history_projection(history_rows)
    model_queries = serialize_model_queries(history_rows)
    passages = serialize_passages(snippet_rows)
    body = {
        "histories": histories_projection,
        "history_projection_sha256": stable_hash(histories_projection),
        "model_query_sha256": stable_hash(list(model_queries)),
        "passage_serialization_sha256": stable_hash(list(passages)),
        "schema": INPUT_SCHEMA,
        "snippet_projection_sha256": stable_hash(snippets_projection),
        "snippets": snippets_projection,
        "study_id": STUDY_ID,
        "typed_core_sha256": TYPED_CORE_SHA256,
    }
    payload = _with_self_hash(body)
    validate_input(payload)
    return payload


def validate_input(value: object) -> ScorerInput:
    _reject_forbidden_keys(value)
    if not isinstance(value, Mapping) or set(value) != INPUT_KEYS:
        raise Dstc9CoordinateScorerError("source-free input schema drifted")
    if (
        value.get("schema") != INPUT_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("typed_core_sha256") != TYPED_CORE_SHA256
    ):
        raise Dstc9CoordinateScorerError("source-free input identity drifted")
    snippets = _parse_snippets(value.get("snippets"))
    histories = _parse_histories(value.get("histories"))
    snippet_projection = _snippet_projection(snippets)
    history_projection = _history_projection(histories)
    model_queries = serialize_model_queries(histories)
    passages = serialize_passages(snippets)
    expected_hashes = {
        "history_projection_sha256": stable_hash(history_projection),
        "model_query_sha256": stable_hash(list(model_queries)),
        "passage_serialization_sha256": stable_hash(list(passages)),
        "snippet_projection_sha256": stable_hash(snippet_projection),
    }
    for field, expected in expected_hashes.items():
        if _sha256(value.get(field), field) != expected:
            raise Dstc9CoordinateScorerError(f"{field} drifted")
    self_sha256 = _verify_self_hash(value, "source-free input")
    return ScorerInput(
        snippets=snippets,
        histories=histories,
        snippet_projection_sha256=expected_hashes[
            "snippet_projection_sha256"
        ],
        history_projection_sha256=expected_hashes[
            "history_projection_sha256"
        ],
        model_query_sha256=expected_hashes["model_query_sha256"],
        passage_serialization_sha256=expected_hashes[
            "passage_serialization_sha256"
        ],
        self_sha256=self_sha256,
    )


def input_projection(value: ScorerInput) -> dict[str, object]:
    if not isinstance(value, ScorerInput):
        raise Dstc9CoordinateScorerError("validated scorer input is invalid")
    return {
        "histories": _history_projection(value.histories),
        "history_projection_sha256": value.history_projection_sha256,
        "model_query_sha256": value.model_query_sha256,
        "passage_serialization_sha256": value.passage_serialization_sha256,
        "schema": INPUT_SCHEMA,
        "self_sha256": value.self_sha256,
        "snippet_projection_sha256": value.snippet_projection_sha256,
        "snippets": _snippet_projection(value.snippets),
        "study_id": STUDY_ID,
        "typed_core_sha256": TYPED_CORE_SHA256,
    }


def _validated_vectors(
    vectors: object,
) -> dict[str, tuple[int, ...]]:
    if not isinstance(vectors, Mapping) or set(vectors) != set(SCORE_NAMES):
        raise Dstc9CoordinateScorerError("score-vector registry drifted")
    output: dict[str, tuple[int, ...]] = {}
    for name in SCORE_NAMES:
        raw = vectors.get(name)
        if (
            isinstance(raw, (str, bytes))
            or not isinstance(raw, Sequence)
            or len(raw) != CORPUS_SIZE
        ):
            raise Dstc9CoordinateScorerError(
                f"{name} vector width drifted"
            )
        values = tuple(raw)
        if any(
            type(score) is not int or abs(score) > MAX_SCORE_ABS
            for score in values
        ):
            raise Dstc9CoordinateScorerError(
                f"{name} vector is not bounded integer data"
            )
        output[name] = values
    return output


def _output_rows(
    scorer_input: ScorerInput,
    score_rows: Sequence[Mapping[str, Sequence[int]]],
) -> list[dict[str, object]]:
    if (
        isinstance(score_rows, (str, bytes))
        or not isinstance(score_rows, Sequence)
        or len(score_rows) != len(scorer_input.histories)
    ):
        raise Dstc9CoordinateScorerError("score row count drifted")
    rows: list[dict[str, object]] = []
    for ordinal, (history, raw_vectors) in enumerate(
        zip(scorer_input.histories, score_rows)
    ):
        vectors = _validated_vectors(raw_vectors)
        rows.append(
            {
                "query_ordinal": ordinal,
                "vectors": {
                    name: list(vectors[name]) for name in SCORE_NAMES
                },
                "work_id": history.work_id,
            }
        )
    return rows


def make_output(
    *,
    scorer_input: ScorerInput,
    score_rows: Sequence[Mapping[str, Sequence[int]]],
    model_binding_sha256: str,
) -> dict[str, object]:
    """Create the canonical self-hashed private score artifact."""

    scorer_input = validate_input(input_projection(scorer_input))
    binding_hash = _sha256(model_binding_sha256, "model binding SHA-256")
    rows = _output_rows(scorer_input, score_rows)
    query_count = len(scorer_input.histories)
    receipt_body = {
        "corpus_count": CORPUS_SIZE,
        "cross_encoder_call_count": 2 * query_count,
        "cross_encoder_model_load_count": 1,
        "cross_encoder_pair_count": 2 * query_count * CORPUS_SIZE,
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "dynamic_resize_count": 0,
        "history_projection_sha256": (
            scorer_input.history_projection_sha256
        ),
        "input_self_sha256": scorer_input.self_sha256,
        "logical_cuda_device": LOGICAL_CUDA_DEVICE,
        "minilm_encode_call_count": 5,
        "minilm_model_load_count": 1,
        "minilm_text_count": query_count + 4 * CORPUS_SIZE,
        "model_binding_sha256": binding_hash,
        "model_query_sha256": scorer_input.model_query_sha256,
        "network_access": "denied",
        "passage_serialization_sha256": (
            scorer_input.passage_serialization_sha256
        ),
        "physical_gpu": PHYSICAL_GPU,
        "quantization": QUANTIZATION,
        "query_count": query_count,
        "query_serialization": QUERY_SERIALIZATION,
        "retry_count": 0,
        "schema": RECEIPT_SCHEMA,
        "score_bundle_sha256": stable_hash(rows),
        "score_names": list(SCORE_NAMES),
        "snippet_projection_sha256": (
            scorer_input.snippet_projection_sha256
        ),
        "snippet_serialization": SNIPPET_SERIALIZATION,
        "status": "passed_private_coordinate_scoring_once",
        "study_id": STUDY_ID,
        "typed_core_sha256": TYPED_CORE_SHA256,
        "work_id_order_sha256": stable_hash(
            [row.work_id for row in scorer_input.histories]
        ),
    }
    receipt = {
        **receipt_body,
        "receipt_sha256": stable_hash(receipt_body),
    }
    body = {
        "corpus_count": CORPUS_SIZE,
        "input_self_sha256": scorer_input.self_sha256,
        "query_count": query_count,
        "receipt": receipt,
        "rows": rows,
        "schema": OUTPUT_SCHEMA,
        "study_id": STUDY_ID,
        "typed_core_sha256": TYPED_CORE_SHA256,
    }
    return _with_self_hash(body)


def validate_output(
    value: object,
    *,
    expected_input: ScorerInput,
    expected_model_binding_sha256: str,
) -> dict[str, object]:
    """Validate a private output without accepting alternative score channels."""

    expected_input = validate_input(input_projection(expected_input))
    if not isinstance(value, Mapping) or set(value) != OUTPUT_KEYS:
        raise Dstc9CoordinateScorerError("private output schema drifted")
    if (
        value.get("schema") != OUTPUT_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("typed_core_sha256") != TYPED_CORE_SHA256
        or value.get("corpus_count") != CORPUS_SIZE
        or value.get("query_count") != len(expected_input.histories)
        or value.get("input_self_sha256") != expected_input.self_sha256
    ):
        raise Dstc9CoordinateScorerError("private output identity drifted")
    raw_rows = value.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(
        expected_input.histories
    ):
        raise Dstc9CoordinateScorerError("private output rows drifted")
    score_rows: list[Mapping[str, Sequence[int]]] = []
    for ordinal, (raw, history) in enumerate(
        zip(raw_rows, expected_input.histories)
    ):
        if (
            not isinstance(raw, Mapping)
            or set(raw) != OUTPUT_ROW_KEYS
            or raw.get("query_ordinal") != ordinal
            or raw.get("work_id") != history.work_id
        ):
            raise Dstc9CoordinateScorerError(
                "private output row binding drifted"
            )
        vectors = _validated_vectors(raw.get("vectors"))
        score_rows.append(vectors)
    expected = make_output(
        scorer_input=expected_input,
        score_rows=score_rows,
        model_binding_sha256=expected_model_binding_sha256,
    )
    if dict(value) != expected:
        raise Dstc9CoordinateScorerError(
            "private output receipt or self hash drifted"
        )
    return expected


def parse_output_bytes(
    raw: bytes,
    *,
    expected_input: ScorerInput,
    expected_model_binding_sha256: str,
) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Dstc9CoordinateScorerError(
            "private output is invalid JSON"
        ) from exc
    if raw != canonical_bytes(value):
        raise Dstc9CoordinateScorerError(
            "private output is not canonical JSON"
        )
    return validate_output(
        value,
        expected_input=expected_input,
        expected_model_binding_sha256=expected_model_binding_sha256,
    )


__all__ = [
    "CORPUS_SIZE",
    "CUDA_VISIBLE_DEVICES",
    "Dstc9CoordinateScorerError",
    "ENTITY_NONE_SERIALIZATION",
    "HistoryItem",
    "INPUT_SCHEMA",
    "LOGICAL_CUDA_DEVICE",
    "MAX_QUERY_COUNT",
    "MAX_SCORE_ABS",
    "MODEL_BINDING_SCHEMA",
    "OUTPUT_SCHEMA",
    "PHYSICAL_GPU",
    "QUANTIZATION",
    "SCORE_NAMES",
    "SCORE_SCALE",
    "STUDY_ID",
    "SYSTEMD_NETWORK_PROPERTIES",
    "ScorerInput",
    "TYPED_CORE_RELATIVE_PATH",
    "TYPED_CORE_SHA256",
    "VERSION",
    "WORKER_ENVIRONMENT_KEYS",
    "WORKER_FIXED_ENVIRONMENT_VALUES",
    "canonical_bytes",
    "input_payload",
    "input_projection",
    "make_output",
    "parse_output_bytes",
    "serialize_entity_fields",
    "serialize_model_queries",
    "serialize_passages",
    "stable_hash",
    "validate_input",
    "validate_output",
    "verify_typed_core_binding",
]
