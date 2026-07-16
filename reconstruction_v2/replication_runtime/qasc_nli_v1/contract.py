"""Exact, label-free request and response contract for the local NLI worker.

Only premise/hypothesis pairs cross the process boundary.  Benchmark item
identifiers, answer keys, option labels, support labels, and evaluator outcomes
are intentionally not representable in this schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence


REQUEST_SCHEMA = "qasc_nli_pair_score_request_v1"
RESPONSE_SCHEMA = "qasc_nli_integer_margin_response_v1"
PAIR_KEYS = frozenset({"hypothesis", "premise"})
REQUEST_KEYS = frozenset({"pairs", "schema"})
RESPONSE_KEYS = frozenset({"schema", "scores"})

BATCH_SIZE = 64
MAXIMUM_SEQUENCE_LENGTH = 256
MAXIMUM_PAIRS_PER_REQUEST = 16_384
MAXIMUM_TEXT_CHARACTERS = 32_768
MAXIMUM_TEXT_UTF8_BYTES = 131_072
MAXIMUM_REQUEST_BYTES = 32 * 1024 * 1024
MAXIMUM_RESPONSE_BYTES = 2 * 1024 * 1024
MINIMUM_SCORE = -(2**63)
MAXIMUM_SCORE = 2**63 - 1


class QASCNLIError(RuntimeError):
    """Raised when the frozen NLI boundary cannot be proven."""


@dataclass(frozen=True)
class NLIPair:
    premise: str
    hypothesis: str

    def as_payload(self) -> dict[str, str]:
        return {"hypothesis": self.hypothesis, "premise": self.premise}


def canonical_json_line(value: object) -> bytes:
    """Return the only accepted wire representation for one JSON value."""

    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise QASCNLIError("value cannot be represented as canonical JSON") from exc


def _reject_json_constant(value: str) -> None:
    raise QASCNLIError(f"non-finite JSON constant is forbidden: {value}")


def _decode_canonical_json_line(raw: bytes, *, maximum_bytes: int, field: str) -> Any:
    if not isinstance(raw, bytes) or not raw or len(raw) > maximum_bytes:
        raise QASCNLIError(f"{field} byte length is outside the frozen bound")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise QASCNLIError(f"{field} is not one canonical JSON line")
    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QASCNLIError(f"{field} is not valid canonical JSON") from exc
    if canonical_json_line(value) != raw:
        raise QASCNLIError(f"{field} is not canonical JSON")
    return value


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QASCNLIError(f"{field} must be non-empty text")
    if "\x00" in value or len(value) > MAXIMUM_TEXT_CHARACTERS:
        raise QASCNLIError(f"{field} is outside the frozen text bound")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise QASCNLIError(f"{field} contains invalid Unicode") from exc
    if len(encoded) > MAXIMUM_TEXT_UTF8_BYTES:
        raise QASCNLIError(f"{field} is outside the frozen UTF-8 byte bound")
    return value


def validate_pairs(pairs: object) -> tuple[NLIPair, ...]:
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
        raise QASCNLIError("pairs must be a sequence")
    if not 1 <= len(pairs) <= MAXIMUM_PAIRS_PER_REQUEST:
        raise QASCNLIError("pair count is outside the frozen request bound")
    normalized: list[NLIPair] = []
    for index, raw in enumerate(pairs):
        if not isinstance(raw, Mapping) or set(raw) != PAIR_KEYS:
            raise QASCNLIError(
                "each pair must contain only premise and hypothesis"
            )
        normalized.append(
            NLIPair(
                premise=_required_text(raw.get("premise"), f"pairs[{index}].premise"),
                hypothesis=_required_text(
                    raw.get("hypothesis"), f"pairs[{index}].hypothesis"
                ),
            )
        )
    return tuple(normalized)


def request_payload(pairs: Sequence[Mapping[str, object] | NLIPair]) -> dict[str, object]:
    raw_pairs = [pair.as_payload() if isinstance(pair, NLIPair) else pair for pair in pairs]
    normalized = validate_pairs(raw_pairs)
    return {
        "pairs": [pair.as_payload() for pair in normalized],
        "schema": REQUEST_SCHEMA,
    }


def encode_request(pairs: Sequence[Mapping[str, object] | NLIPair]) -> bytes:
    return canonical_json_line(request_payload(pairs))


def decode_request(raw: bytes) -> tuple[NLIPair, ...]:
    value = _decode_canonical_json_line(
        raw,
        maximum_bytes=MAXIMUM_REQUEST_BYTES,
        field="NLI request",
    )
    if not isinstance(value, Mapping) or set(value) != REQUEST_KEYS:
        raise QASCNLIError("NLI request envelope is not exact")
    if value.get("schema") != REQUEST_SCHEMA:
        raise QASCNLIError("NLI request schema mismatch")
    return validate_pairs(value.get("pairs"))


def validate_scores(scores: object, *, expected_count: int | None = None) -> tuple[int, ...]:
    if isinstance(scores, (str, bytes)) or not isinstance(scores, Sequence):
        raise QASCNLIError("scores must be a sequence")
    if expected_count is not None and len(scores) != expected_count:
        raise QASCNLIError("NLI response score count mismatch")
    if not 1 <= len(scores) <= MAXIMUM_PAIRS_PER_REQUEST:
        raise QASCNLIError("score count is outside the frozen response bound")
    normalized: list[int] = []
    for value in scores:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not MINIMUM_SCORE <= value <= MAXIMUM_SCORE
        ):
            raise QASCNLIError("NLI response contains a malformed score")
        normalized.append(value)
    return tuple(normalized)


def encode_response(scores: Sequence[int]) -> bytes:
    normalized = validate_scores(scores)
    return canonical_json_line(
        {"schema": RESPONSE_SCHEMA, "scores": list(normalized)}
    )


def decode_response(raw: bytes, *, expected_count: int) -> tuple[int, ...]:
    value = _decode_canonical_json_line(
        raw,
        maximum_bytes=MAXIMUM_RESPONSE_BYTES,
        field="NLI response",
    )
    if not isinstance(value, Mapping) or set(value) != RESPONSE_KEYS:
        raise QASCNLIError("NLI response envelope is not exact")
    if value.get("schema") != RESPONSE_SCHEMA:
        raise QASCNLIError("NLI response schema mismatch")
    return validate_scores(value.get("scores"), expected_count=expected_count)
