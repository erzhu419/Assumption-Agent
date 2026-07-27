"""Source-free contract for the BioASQ six-coordinate private scorer.

The input is deliberately narrower than the formal source: exactly 2,900
ordered ``ordinal/text`` passages and 1..256 question texts.  It cannot carry
question family/type, qrels, document identifiers, answers, labels, scores, or
evaluator decisions.  Query variants are formed only by the frozen typed core.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from assumption_agent.benchmarks import bioasq_p1_typed_core_v1 as typed_core


VERSION = "bioasq_coordinate_scorer_v1"
STUDY_ID = "BIOASQ_P1_TYPED_QUESTION_EVIDENCE_EVALUATOR_L5_V1"
TYPED_CORE_SHA256 = (
    "6bfd386431b977043f43eac0984a67b688fad9def276d37902b2fb3c4cff9342"
)
TYPED_CORE_RELATIVE_PATH = (
    "assumption_agent/benchmarks/bioasq_p1_typed_core_v1.py"
)

CORPUS_SIZE = 2900
MIN_QUERY_COUNT = 1
MAX_QUERY_COUNT = 256
# Coordinate quantization is independent of the typed-core utility scale.
SCORE_SCALE = 1_000_000
MAX_SCORE_ABS = typed_core.MAX_SCORE_ABS
SCORE_NAMES = (
    "raw_ce",
    "focus_ce",
    "dense_base",
    "dense_support",
    "dense_contrast",
    "dense_coverage",
)
DENSE_SCORE_NAMES = SCORE_NAMES[2:]
PHYSICAL_GPU = 1
CUDA_VISIBLE_DEVICES = "1"
LOGICAL_CUDA_DEVICE = "cuda:0"

INPUT_SCHEMA = f"{VERSION}_source_free_input_v1"
OUTPUT_SCHEMA = f"{VERSION}_private_output_v1"
RECEIPT_SCHEMA = f"{VERSION}_private_receipt_v1"
MODEL_BINDING_SCHEMA = f"{VERSION}_model_binding_v1"
PASSAGE_SERIALIZATION = "frozen_bioasq_typed_core_serialize_passage_v1"
QUERY_SERIALIZATION = "frozen_bioasq_typed_core_score_query_bundle_v1"
QUANTIZATION = "finite_float64_rint_times_1000000_bounded_int_v1"
TRANSPORT = "systemd_run_user_transient_service_v1"
SYSTEMD_NETWORK_PROPERTIES = (
    "IPAddressDeny=any",
    "RestrictAddressFamilies=AF_UNIX",
)

PASSAGE_KEYS = frozenset({"ordinal", "text"})
QUERY_KEYS = frozenset({"text"})
INPUT_KEYS = frozenset(
    {
        "passage_projection_sha256",
        "passage_serialization_sha256",
        "passages",
        "queries",
        "query_projection_sha256",
        "query_variant_serialization_sha256",
        "schema",
        "self_sha256",
        "study_id",
        "typed_core_sha256",
    }
)
OUTPUT_ROW_KEYS = frozenset({"query_ordinal", "vectors"})
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
        "input_self_sha256",
        "logical_cuda_device",
        "minilm_constructor_canary_encode_call_count",
        "minilm_formal_batch_encode_call_count",
        "minilm_model_load_count",
        "minilm_passage_count",
        "minilm_query_variant_count",
        "minilm_text_count",
        "minilm_total_encode_call_count",
        "model_binding_sha256",
        "network_access",
        "passage_projection_sha256",
        "passage_serialization",
        "passage_serialization_sha256",
        "physical_gpu",
        "quantization",
        "query_count",
        "query_projection_sha256",
        "query_serialization",
        "query_variant_serialization_sha256",
        "receipt_sha256",
        "retry_count",
        "schema",
        "score_bundle_sha256",
        "score_names",
        "status",
        "study_id",
        "typed_core_sha256",
    }
)

FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "answer",
        "answers",
        "document_id",
        "doc_id",
        "evaluator",
        "families",
        "family",
        "gold",
        "id",
        "item_id",
        "label",
        "labels",
        "metric",
        "pmid",
        "qrel",
        "qrels",
        "question_id",
        "question_type",
        "relevance",
        "score",
        "scores",
        "split",
        "target",
        "type",
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


class BioasqCoordinateScorerError(RuntimeError):
    """The source-free scorer contract or private output drifted."""


@dataclass(frozen=True, slots=True)
class QueryItem:
    text: str


@dataclass(frozen=True, slots=True)
class ScorerInput:
    passages: tuple[typed_core.Passage, ...]
    queries: tuple[QueryItem, ...]
    passage_projection_sha256: str
    query_projection_sha256: str
    query_variant_serialization_sha256: str
    passage_serialization_sha256: str
    self_sha256: str


def canonical_bytes(value: object, *, newline: bool = True) -> bytes:
    try:
        return typed_core.canonical_bytes(value, newline=newline)
    except Exception as exc:
        raise BioasqCoordinateScorerError(
            "value is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    try:
        return typed_core.stable_hash(value)
    except Exception as exc:
        raise BioasqCoordinateScorerError("value cannot be hashed") from exc


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise BioasqCoordinateScorerError(
            f"{field} is not a lowercase SHA-256"
        )
    return value


def _reject_forbidden_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if (
                not isinstance(key, str)
                or key.casefold() in FORBIDDEN_INPUT_KEYS
            ):
                raise BioasqCoordinateScorerError(
                    "input contains a forbidden source/family/id/label/score field"
                )
            _reject_forbidden_keys(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _reject_forbidden_keys(nested)


def _with_self_hash(body: Mapping[str, object]) -> dict[str, object]:
    if "self_sha256" in body:
        raise BioasqCoordinateScorerError("self hash was supplied twice")
    result = dict(body)
    result["self_sha256"] = stable_hash(result)
    return result


def _verify_self_hash(value: Mapping[str, object], field: str) -> str:
    body = dict(value)
    claimed = _sha256(body.pop("self_sha256", None), f"{field} self hash")
    if stable_hash(body) != claimed:
        raise BioasqCoordinateScorerError(f"{field} self hash drifted")
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
        raise BioasqCoordinateScorerError(
            "project root must be an absolute path"
        )
    path = project_root / TYPED_CORE_RELATIVE_PATH
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
        or _sha256_file(path) != TYPED_CORE_SHA256
        or typed_core.STUDY_ID != STUDY_ID
        or typed_core.VERSION != "bioasq_p1_typed_core_v1"
        or tuple(typed_core.SCORE_NAMES) != SCORE_NAMES
    ):
        raise BioasqCoordinateScorerError(
            "frozen typed-core binding drifted"
        )
    return path


def _passage_projection(
    passages: Sequence[typed_core.Passage],
) -> list[dict[str, object]]:
    return [typed_core.passage_public_payload(row) for row in passages]


def _query_projection(queries: Sequence[QueryItem]) -> list[dict[str, str]]:
    return [{"text": row.text} for row in queries]


def _bundle_payload(bundle: object) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in SCORE_NAMES:
        value = getattr(bundle, name, None)
        if (
            not isinstance(value, str)
            or not value.strip()
            or "\x00" in value
        ):
            raise BioasqCoordinateScorerError(
                "typed-core score-query bundle drifted"
            )
        result[name] = value
    return result


def serialize_query_variants(
    queries: Sequence[QueryItem],
) -> tuple[dict[str, str], ...]:
    try:
        return tuple(
            _bundle_payload(typed_core.serialize_score_queries(row.text))
            for row in queries
        )
    except BioasqCoordinateScorerError:
        raise
    except Exception as exc:
        raise BioasqCoordinateScorerError(
            "query-variant serialization failed"
        ) from exc


def serialize_passages(
    passages: Sequence[typed_core.Passage],
) -> tuple[str, ...]:
    try:
        result = tuple(typed_core.serialize_passage(row) for row in passages)
    except Exception as exc:
        raise BioasqCoordinateScorerError(
            "passage serialization failed"
        ) from exc
    if any(
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        for value in result
    ):
        raise BioasqCoordinateScorerError(
            "typed-core passage serialization drifted"
        )
    return result


def _parse_passages(value: object) -> tuple[typed_core.Passage, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != CORPUS_SIZE
    ):
        raise BioasqCoordinateScorerError(
            "passage corpus must contain exactly 2900 rows"
        )
    rows: list[typed_core.Passage] = []
    for position, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != PASSAGE_KEYS:
            raise BioasqCoordinateScorerError(
                "passage must contain exact public fields"
            )
        try:
            row = typed_core.passage_from_public_fields(raw)
        except Exception as exc:
            raise BioasqCoordinateScorerError(
                "typed-core passage validation failed"
            ) from exc
        if row.ordinal != position:
            raise BioasqCoordinateScorerError(
                "passage ordinals must be contiguous zero-based order"
            )
        if dict(raw) != typed_core.passage_public_payload(row):
            raise BioasqCoordinateScorerError(
                "passage is not the canonical typed-core projection"
            )
        rows.append(row)
    return tuple(rows)


def _query_text(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > 24_000
    ):
        raise BioasqCoordinateScorerError(f"{field} is invalid")
    try:
        typed_core.serialize_score_queries(value)
    except Exception as exc:
        raise BioasqCoordinateScorerError(
            f"{field} failed typed-core validation"
        ) from exc
    return value


def _parse_queries(value: object) -> tuple[QueryItem, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not MIN_QUERY_COUNT <= len(value) <= MAX_QUERY_COUNT
    ):
        raise BioasqCoordinateScorerError(
            "query count is outside the frozen 1..256 bound"
        )
    rows: list[QueryItem] = []
    for position, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != QUERY_KEYS:
            raise BioasqCoordinateScorerError(
                "query must contain only question text"
            )
        rows.append(
            QueryItem(text=_query_text(raw.get("text"), f"queries[{position}]"))
        )
    return tuple(rows)


def input_payload(
    *,
    passages: Sequence[Mapping[str, object]],
    queries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Normalize through the typed core and create a self-hashed input."""

    passage_rows = _parse_passages(passages)
    query_rows = _parse_queries(queries)
    passage_projection = _passage_projection(passage_rows)
    query_projection = _query_projection(query_rows)
    query_variants = serialize_query_variants(query_rows)
    passage_serialization = serialize_passages(passage_rows)
    body = {
        "passage_projection_sha256": stable_hash(passage_projection),
        "passage_serialization_sha256": stable_hash(
            [
                hashlib.sha256(value.encode("utf-8")).hexdigest()
                for value in passage_serialization
            ]
        ),
        "passages": passage_projection,
        "queries": query_projection,
        "query_projection_sha256": stable_hash(query_projection),
        "query_variant_serialization_sha256": stable_hash(
            list(query_variants)
        ),
        "schema": INPUT_SCHEMA,
        "study_id": STUDY_ID,
        "typed_core_sha256": TYPED_CORE_SHA256,
    }
    payload = _with_self_hash(body)
    validate_input(payload)
    return payload


def validate_input(value: object) -> ScorerInput:
    _reject_forbidden_keys(value)
    if not isinstance(value, Mapping) or set(value) != INPUT_KEYS:
        raise BioasqCoordinateScorerError("source-free input schema drifted")
    if (
        value.get("schema") != INPUT_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("typed_core_sha256") != TYPED_CORE_SHA256
    ):
        raise BioasqCoordinateScorerError("source-free input identity drifted")
    passages = _parse_passages(value.get("passages"))
    queries = _parse_queries(value.get("queries"))
    passage_projection = _passage_projection(passages)
    query_projection = _query_projection(queries)
    query_variants = serialize_query_variants(queries)
    passage_serialization = serialize_passages(passages)
    expected_hashes = {
        "passage_projection_sha256": stable_hash(passage_projection),
        "passage_serialization_sha256": stable_hash(
            [
                hashlib.sha256(value.encode("utf-8")).hexdigest()
                for value in passage_serialization
            ]
        ),
        "query_projection_sha256": stable_hash(query_projection),
        "query_variant_serialization_sha256": stable_hash(
            list(query_variants)
        ),
    }
    for field, expected in expected_hashes.items():
        if _sha256(value.get(field), field) != expected:
            raise BioasqCoordinateScorerError(f"{field} drifted")
    self_sha256 = _verify_self_hash(value, "source-free input")
    return ScorerInput(
        passages=passages,
        queries=queries,
        passage_projection_sha256=expected_hashes[
            "passage_projection_sha256"
        ],
        query_projection_sha256=expected_hashes["query_projection_sha256"],
        query_variant_serialization_sha256=expected_hashes[
            "query_variant_serialization_sha256"
        ],
        passage_serialization_sha256=expected_hashes[
            "passage_serialization_sha256"
        ],
        self_sha256=self_sha256,
    )


def input_projection(value: ScorerInput) -> dict[str, object]:
    if not isinstance(value, ScorerInput):
        raise BioasqCoordinateScorerError("validated scorer input is invalid")
    return {
        "passage_projection_sha256": value.passage_projection_sha256,
        "passage_serialization_sha256": value.passage_serialization_sha256,
        "passages": _passage_projection(value.passages),
        "queries": _query_projection(value.queries),
        "query_projection_sha256": value.query_projection_sha256,
        "query_variant_serialization_sha256": (
            value.query_variant_serialization_sha256
        ),
        "schema": INPUT_SCHEMA,
        "self_sha256": value.self_sha256,
        "study_id": STUDY_ID,
        "typed_core_sha256": TYPED_CORE_SHA256,
    }


def _validated_vectors(vectors: object) -> dict[str, tuple[int, ...]]:
    if not isinstance(vectors, Mapping) or set(vectors) != set(SCORE_NAMES):
        raise BioasqCoordinateScorerError("score-vector registry drifted")
    output: dict[str, tuple[int, ...]] = {}
    for name in SCORE_NAMES:
        raw = vectors.get(name)
        if (
            isinstance(raw, (str, bytes))
            or not isinstance(raw, Sequence)
            or len(raw) != CORPUS_SIZE
        ):
            raise BioasqCoordinateScorerError(f"{name} vector width drifted")
        values = tuple(raw)
        if any(
            type(score) is not int or abs(score) > MAX_SCORE_ABS
            for score in values
        ):
            raise BioasqCoordinateScorerError(
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
        or len(score_rows) != len(scorer_input.queries)
    ):
        raise BioasqCoordinateScorerError("score row count drifted")
    rows: list[dict[str, object]] = []
    for ordinal, raw_vectors in enumerate(score_rows):
        vectors = _validated_vectors(raw_vectors)
        rows.append(
            {
                "query_ordinal": ordinal,
                "vectors": {
                    name: list(vectors[name]) for name in SCORE_NAMES
                },
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
    query_count = len(scorer_input.queries)
    receipt_body = {
        "corpus_count": CORPUS_SIZE,
        "cross_encoder_call_count": 2,
        "cross_encoder_model_load_count": 1,
        "cross_encoder_pair_count": 2 * query_count * CORPUS_SIZE,
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        "dynamic_resize_count": 0,
        "input_self_sha256": scorer_input.self_sha256,
        "logical_cuda_device": LOGICAL_CUDA_DEVICE,
        # BrightMiniLMEncoder performs two repeat-exact synthetic canary
        # encodes during its single constructor, followed by exactly one
        # content-bearing formal batch encode in this scorer lifecycle.
        "minilm_constructor_canary_encode_call_count": 2,
        "minilm_formal_batch_encode_call_count": 1,
        "minilm_model_load_count": 1,
        "minilm_passage_count": CORPUS_SIZE,
        "minilm_query_variant_count": len(DENSE_SCORE_NAMES) * query_count,
        "minilm_text_count": (
            CORPUS_SIZE + len(DENSE_SCORE_NAMES) * query_count
        ),
        "minilm_total_encode_call_count": 3,
        "model_binding_sha256": binding_hash,
        "network_access": "denied",
        "passage_projection_sha256": (
            scorer_input.passage_projection_sha256
        ),
        "passage_serialization": PASSAGE_SERIALIZATION,
        "passage_serialization_sha256": (
            scorer_input.passage_serialization_sha256
        ),
        "physical_gpu": PHYSICAL_GPU,
        "quantization": QUANTIZATION,
        "query_count": query_count,
        "query_projection_sha256": scorer_input.query_projection_sha256,
        "query_serialization": QUERY_SERIALIZATION,
        "query_variant_serialization_sha256": (
            scorer_input.query_variant_serialization_sha256
        ),
        "retry_count": 0,
        "schema": RECEIPT_SCHEMA,
        "score_bundle_sha256": stable_hash(rows),
        "score_names": list(SCORE_NAMES),
        "status": "passed_private_coordinate_scoring_once",
        "study_id": STUDY_ID,
        "typed_core_sha256": TYPED_CORE_SHA256,
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
    expected_input = validate_input(input_projection(expected_input))
    if not isinstance(value, Mapping) or set(value) != OUTPUT_KEYS:
        raise BioasqCoordinateScorerError("private output schema drifted")
    if (
        value.get("schema") != OUTPUT_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("typed_core_sha256") != TYPED_CORE_SHA256
        or value.get("corpus_count") != CORPUS_SIZE
        or value.get("query_count") != len(expected_input.queries)
        or value.get("input_self_sha256") != expected_input.self_sha256
    ):
        raise BioasqCoordinateScorerError("private output identity drifted")
    raw_rows = value.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(
        expected_input.queries
    ):
        raise BioasqCoordinateScorerError("private output rows drifted")
    score_rows: list[Mapping[str, Sequence[int]]] = []
    for ordinal, raw in enumerate(raw_rows):
        if (
            not isinstance(raw, Mapping)
            or set(raw) != OUTPUT_ROW_KEYS
            or raw.get("query_ordinal") != ordinal
        ):
            raise BioasqCoordinateScorerError(
                "private output row binding drifted"
            )
        score_rows.append(_validated_vectors(raw.get("vectors")))
    expected = make_output(
        scorer_input=expected_input,
        score_rows=score_rows,
        model_binding_sha256=expected_model_binding_sha256,
    )
    if dict(value) != expected:
        raise BioasqCoordinateScorerError(
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
        raise BioasqCoordinateScorerError(
            "private output is invalid JSON"
        ) from exc
    if raw != canonical_bytes(value):
        raise BioasqCoordinateScorerError(
            "private output is not canonical JSON"
        )
    return validate_output(
        value,
        expected_input=expected_input,
        expected_model_binding_sha256=expected_model_binding_sha256,
    )


__all__ = [
    "BioasqCoordinateScorerError",
    "CORPUS_SIZE",
    "CUDA_VISIBLE_DEVICES",
    "DENSE_SCORE_NAMES",
    "INPUT_SCHEMA",
    "LOGICAL_CUDA_DEVICE",
    "MAX_QUERY_COUNT",
    "MAX_SCORE_ABS",
    "MODEL_BINDING_SCHEMA",
    "OUTPUT_SCHEMA",
    "PHYSICAL_GPU",
    "QUANTIZATION",
    "QueryItem",
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
    "serialize_passages",
    "serialize_query_variants",
    "stable_hash",
    "validate_input",
    "validate_output",
    "verify_typed_core_binding",
]
