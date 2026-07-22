"""Canonical offline Qwen typed-plan contract for TAT-QA P18."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks import tatqa_p18_label_free_runtime_v1 as features
from assumption_agent.benchmarks import tatqa_p18_typed_evaluator_core_v1 as core


VERSION = "tatqa_p18_typed_plan_runtime_v1"
INPUT_SCHEMA = f"{VERSION}_input"
OUTPUT_SCHEMA = f"{VERSION}_output"
MAXIMUM_ITEM_COUNT = 64
MAXIMUM_QUESTION_CHARACTERS = 4_096
MAXIMUM_TABLE_HEADER_CHARACTERS = 4_096
MAXIMUM_PARAGRAPH_LEADS = 8
MAXIMUM_PARAGRAPH_LEAD_CHARACTERS = 512
MAXIMUM_COMPLETION_TOKENS = 256

_INPUT_KEYS = frozenset({"items", "schema"})
_INPUT_ITEM_KEYS = frozenset(
    {"ordinal", "paragraph_leads", "question", "table_header"}
)
_OUTPUT_KEYS = frozenset({"items", "schema"})
_OUTPUT_ITEM_KEYS = frozenset(
    {
        "completion_sha256",
        "completion_token_count",
        "generation_valid",
        "ordinal",
        "plan",
        "prompt_projection_sha256",
        "prompt_sha256",
        "prompt_token_count",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class TatqaP18TypedPlanRuntimeError(RuntimeError):
    """The frozen label-free typed-plan envelope drifted."""


@dataclass(frozen=True)
class PlanInput:
    ordinal: int
    question: str
    table_header: str
    paragraph_leads: tuple[str, ...]


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
        raise TatqaP18TypedPlanRuntimeError("value is not canonical JSON") from exc


def _text(value: object, *, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not value.strip()
        or "\x00" in value
        or len(value) > maximum
        or "  " in value
        or value != value.strip()
    ):
        raise TatqaP18TypedPlanRuntimeError(f"{field} is not canonical text")
    return value


def _truncate(value: str, maximum: int) -> str:
    truncated = value[:maximum].rstrip()
    if not truncated:
        raise TatqaP18TypedPlanRuntimeError("projected text became empty")
    return truncated


def project_item(item: features.LabelFreeRuntimeItem, ordinal: int) -> PlanInput:
    """Project one canonical item into the only Qwen-visible fields.

    Paragraphs are retained in canonical order; at most the first eight and the
    first 512 Unicode code points of each are visible.  No item identity is
    included in either the dataclass or serialized prompt input.
    """

    if not isinstance(item, features.LabelFreeRuntimeItem):
        raise TatqaP18TypedPlanRuntimeError("label-free item type drifted")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise TatqaP18TypedPlanRuntimeError("ordinal is invalid")
    table_header = next(row.text for row in item.units if row.unit_id == "T:0")
    paragraph_leads = tuple(
        _truncate(row.text, MAXIMUM_PARAGRAPH_LEAD_CHARACTERS)
        for row in item.units
        if row.unit_id.startswith("P:")
    )[:MAXIMUM_PARAGRAPH_LEADS]
    return PlanInput(
        ordinal=ordinal,
        question=_truncate(item.question, MAXIMUM_QUESTION_CHARACTERS),
        table_header=_truncate(table_header, MAXIMUM_TABLE_HEADER_CHARACTERS),
        paragraph_leads=paragraph_leads,
    )


def _validated_items(value: object) -> tuple[PlanInput, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TatqaP18TypedPlanRuntimeError("typed-plan items are not a sequence")
    if not 1 <= len(value) <= MAXIMUM_ITEM_COUNT:
        raise TatqaP18TypedPlanRuntimeError("typed-plan item count is outside the bound")
    rows: list[PlanInput] = []
    for position, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != _INPUT_ITEM_KEYS:
            raise TatqaP18TypedPlanRuntimeError("typed-plan input shape drifted")
        ordinal = raw.get("ordinal")
        leads = raw.get("paragraph_leads")
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal != position
            or isinstance(leads, (str, bytes))
            or not isinstance(leads, Sequence)
            or len(leads) > MAXIMUM_PARAGRAPH_LEADS
        ):
            raise TatqaP18TypedPlanRuntimeError("typed-plan input values drifted")
        rows.append(
            PlanInput(
                ordinal=ordinal,
                question=_text(
                    raw.get("question"),
                    field="question",
                    maximum=MAXIMUM_QUESTION_CHARACTERS,
                ),
                table_header=_text(
                    raw.get("table_header"),
                    field="table header",
                    maximum=MAXIMUM_TABLE_HEADER_CHARACTERS,
                ),
                paragraph_leads=tuple(
                    _text(
                        lead,
                        field="paragraph lead",
                        maximum=MAXIMUM_PARAGRAPH_LEAD_CHARACTERS,
                    )
                    for lead in leads
                ),
            )
        )
    return tuple(rows)


def input_payload(items: Sequence[PlanInput]) -> dict[str, Any]:
    rows = tuple(items)
    validated = _validated_items(
        [
            {
                "ordinal": row.ordinal,
                "paragraph_leads": list(row.paragraph_leads),
                "question": row.question,
                "table_header": row.table_header,
            }
            for row in rows
        ]
    )
    return {
        "items": [
            {
                "ordinal": row.ordinal,
                "paragraph_leads": list(row.paragraph_leads),
                "question": row.question,
                "table_header": row.table_header,
            }
            for row in validated
        ],
        "schema": INPUT_SCHEMA,
    }


def parse_input(raw: bytes) -> tuple[PlanInput, ...]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TatqaP18TypedPlanRuntimeError("typed-plan input is invalid JSON") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != _INPUT_KEYS
        or value.get("schema") != INPUT_SCHEMA
        or canonical_json_bytes(value) != raw
    ):
        raise TatqaP18TypedPlanRuntimeError("typed-plan input envelope drifted")
    return _validated_items(value.get("items"))


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("non-finite JSON constant")


def _candidate_from_completion(completion: str) -> object:
    text = completion.strip()
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError):
        return None


def build_output_item(
    *,
    item: PlanInput,
    completion: str,
    completion_token_count: int,
    prompt_sha256: str,
    prompt_token_count: int,
    prompt_projection_sha256: str,
) -> dict[str, Any]:
    if not isinstance(item, PlanInput):
        raise TatqaP18TypedPlanRuntimeError("typed-plan item type drifted")
    if not isinstance(completion, str) or "\x00" in completion:
        raise TatqaP18TypedPlanRuntimeError("completion is invalid text")
    if (
        isinstance(completion_token_count, bool)
        or not isinstance(completion_token_count, int)
        or not 0 <= completion_token_count <= MAXIMUM_COMPLETION_TOKENS
    ):
        raise TatqaP18TypedPlanRuntimeError("completion token count drifted")
    if (
        not isinstance(prompt_sha256, str)
        or _SHA256.fullmatch(prompt_sha256) is None
        or not isinstance(prompt_projection_sha256, str)
        or _SHA256.fullmatch(prompt_projection_sha256) is None
        or isinstance(prompt_token_count, bool)
        or not isinstance(prompt_token_count, int)
        or prompt_token_count < 1
    ):
        raise TatqaP18TypedPlanRuntimeError("prompt projection receipt drifted")
    candidate = _candidate_from_completion(completion)
    try:
        plan = core.validate_typed_plan(candidate)  # type: ignore[arg-type]
        valid = True
    except (core.TatqaP18TypedEvaluatorError, TypeError):
        plan = core.totalize_typed_plan(
            candidate,
            fallback_relation_query=item.question,
            fallback_entity_facets=(item.question,),
            fallback_metric_facets=(item.table_header,),
            fallback_time_facets=(),
        )
        valid = False
    return {
        "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
        "completion_token_count": completion_token_count,
        "generation_valid": valid,
        "ordinal": item.ordinal,
        "plan": plan.payload(),
        "prompt_projection_sha256": prompt_projection_sha256,
        "prompt_sha256": prompt_sha256,
        "prompt_token_count": prompt_token_count,
    }


def output_payload(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(items)
    if not 1 <= len(rows) <= MAXIMUM_ITEM_COUNT:
        raise TatqaP18TypedPlanRuntimeError("typed-plan output count drifted")
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _OUTPUT_ITEM_KEYS:
            raise TatqaP18TypedPlanRuntimeError("typed-plan output shape drifted")
        ordinal = row.get("ordinal")
        tokens = row.get("completion_token_count")
        digest = row.get("completion_sha256")
        valid = row.get("generation_valid")
        prompt_tokens = row.get("prompt_token_count")
        if (
            isinstance(ordinal, bool)
            or not isinstance(ordinal, int)
            or ordinal != position
            or isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or not 0 <= tokens <= MAXIMUM_COMPLETION_TOKENS
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or not isinstance(valid, bool)
            or isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens < 1
            or not isinstance(row.get("prompt_sha256"), str)
            or _SHA256.fullmatch(row["prompt_sha256"]) is None
            or not isinstance(row.get("prompt_projection_sha256"), str)
            or _SHA256.fullmatch(row["prompt_projection_sha256"]) is None
        ):
            raise TatqaP18TypedPlanRuntimeError("typed-plan output values drifted")
        try:
            core.validate_typed_plan(row.get("plan"))  # type: ignore[arg-type]
        except (core.TatqaP18TypedEvaluatorError, TypeError) as exc:
            raise TatqaP18TypedPlanRuntimeError("totalized typed plan drifted") from exc
    return {"items": rows, "schema": OUTPUT_SCHEMA}


def parse_output(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TatqaP18TypedPlanRuntimeError("typed-plan output is invalid JSON") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != _OUTPUT_KEYS
        or value.get("schema") != OUTPUT_SCHEMA
        or canonical_json_bytes(value) != raw
    ):
        raise TatqaP18TypedPlanRuntimeError("typed-plan output envelope drifted")
    return output_payload(value.get("items", ()))


__all__ = [
    "INPUT_SCHEMA",
    "MAXIMUM_COMPLETION_TOKENS",
    "MAXIMUM_ITEM_COUNT",
    "OUTPUT_SCHEMA",
    "PlanInput",
    "TatqaP18TypedPlanRuntimeError",
    "VERSION",
    "build_output_item",
    "canonical_json_bytes",
    "input_payload",
    "output_payload",
    "parse_input",
    "parse_output",
    "project_item",
]
