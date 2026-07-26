"""Private source boundary for the frozen MAUD extraction P2 study.

The parser intentionally does not use the official preprocessing pipeline.
It streams the top-level SQuAD2 ``data`` array one contract at a time, applies
the frozen whole-contract exposure exclusion before block assignment, and
projects separate label-free action views and private gold packs.

Most importantly, TRAIN contracts assigned to ``F_search`` never have their
``answers`` values semantically decoded.  Those values are only traversed by
the JSON syntax scanner, so no F_search gold object can be returned or written.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import codecs
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, BinaryIO, Iterator, Mapping, Sequence, TextIO
import unicodedata


VERSION = "maud_extraction_p2_source_v1"
STUDY_ID = "MAUD_EXTRACTION_P2_CGROUP_BOUNDED_EVALUATOR_V1"
SPLITS = ("TRAIN", "DEV", "TEST")
BLOCKS = ("A_form", "F_search", "A_hold", "M_search")

DEFINITION_REFERENCE_TYPES = (
    "Intervening Event Definition",
    "Knowledge Definition",
    "MAE Definition",
    "Superior Offer Definition",
    "Type of Consideration",
)
CONDITION_OBLIGATION_TYPES = (
    "Absence of Litigation Closing Condition",
    "Accuracy of Target R&W Closing Condition",
    "Compliance with Covenant Closing Condition",
    "General Antitrust Efforts Standard",
    "Negative interim operating covenant",
    "Ordinary course covenant",
)
PROTECTION_EXCEPTION_REMEDY_TYPES = (
    "Agreement provides for matching rights in connection with COR",
    "Agreement provides for matching rights in connection with FTR",
    "Breach of Meeting Covenant",
    "Breach of No Shop",
    "FTR Triggers",
    "Fiduciary exception to COR covenant",
    "Fiduciary exception: Board determination (no-shop)",
    "Limitations on FTR Exercise",
    "No-Shop",
    "Specific Performance",
    "Tail Period & Acquisition Proposal Details",
)
DEAL_POINT_TYPES = (
    *DEFINITION_REFERENCE_TYPES,
    *CONDITION_OBLIGATION_TYPES,
    *PROTECTION_EXCEPTION_REMEDY_TYPES,
)
TYPE_ALIASES = {
    "Fiduciary exception to COR convent": (
        "Fiduciary exception to COR covenant"
    ),
    "Negative interim operating convenant": (
        "Negative interim operating covenant"
    ),
}
QUESTION_PREFIX = (
    "Highlight the parts of the text (if any) related to "
)
QUESTION_SUFFIX = " that should be reviewed by a lawyer"

EXCLUDED_NORMALIZED_TITLE_SHA256S = frozenset(
    {
        "fcf2822d878e9b74a8fba51c92e5326ca152989cad7e2239654a462658be08a1",
        "567690b1766043b436952371cf33efb3bf522fd055845eb8739e594742688b47",
        "8160234bb08577cf04d1fb7bda6cc1615cfd1cba250b6cc88be3ee8550629180",
        "ef52a774eb5a2d331d8dc46758e844a6caa3e6db4efbb0cc4b626f604d789381",
        "996cc530c89795b4723b53bf87d0d9f851977b0423d1682bf251dec060989661",
        "96078da0eed1df9855965842f7ffa00a5265753759fb3a546ed1811ea9bb2bea",
        "5727988807294281cfe0f71bf13d459bcba46969e991e53cabdcaae544324c90",
        "6f00c3543ce52ed75eca239c6d0f7c2c0ecb2024fc66d29326b59aa05542072f",
        "33fb16324b766093529c8de60b90f4b85d971a3c7ce71dd0b06ce34f73f32f3c",
        "75f267f1bfc603ead771b03bc66956c2b86d024a71b082c3fa585b67d715d7ec",
        "45308e3b4f3a4f972894fb0d5f076678a6902e008e90bc41d15bd71e678c56df",
        "f220358f37c9e7b6df383a1241ec9361c2ae10e8521b76b963374fa8e09a328e",
        "404eee073f1409341a500835030fd0d62615a6ffc3dc65068782f3fa9edfe4d2",
        "37a2db6efc97af85b99f792aa1219e22587323030d42b1d0ee918e471a6b5339",
        "1c6bbcaf5c84ed3cdb9db1493d2252f0e79b6af8ca170271884c4c9430413074",
        "5fc8e7df4255a454801a75631cd6706c0ec9facda3192f711caf5d76bf326ae6",
    }
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_NUMBER = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z"
)
_TYPE_TO_FAMILY = {
    **{value: "definition_reference" for value in DEFINITION_REFERENCE_TYPES},
    **{value: "condition_obligation" for value in CONDITION_OBLIGATION_TYPES},
    **{
        value: "protection_exception_remedy"
        for value in PROTECTION_EXCEPTION_REMEDY_TYPES
    },
}
_TYPE_ORDER = {value: index for index, value in enumerate(DEAL_POINT_TYPES)}


class MaudSourceError(RuntimeError):
    """The source schema, privacy boundary, or frozen split contract drifted."""


@dataclass(frozen=True)
class TestParseCapability:
    """Explicit authority to semantically parse the untouched TEST source."""

    a_hold_promotion_receipt_sha256: str
    study_id: str = STUDY_ID
    promoted: bool = True

    def validate(self) -> None:
        if self.study_id != STUDY_ID or self.promoted is not True:
            raise MaudSourceError("TEST parse capability is not authorized")
        if (
            not isinstance(self.a_hold_promotion_receipt_sha256, str)
            or _HEX64.fullmatch(
                self.a_hold_promotion_receipt_sha256
            )
            is None
        ):
            raise MaudSourceError("TEST parse capability receipt is invalid")


@dataclass(frozen=True, order=True)
class GoldSpan:
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or self.start < 0
            or self.end <= self.start
            or not isinstance(self.text, str)
            or not self.text
        ):
            raise MaudSourceError("gold span is invalid")


@dataclass(frozen=True)
class PreparedItem:
    work_id: str
    question: str
    deal_point_type: str
    family: str
    spans: tuple[GoldSpan, ...] | None
    merged_intervals: tuple[tuple[int, int], ...] | None

    @property
    def gold_semantically_opened(self) -> bool:
        return self.spans is not None


@dataclass(frozen=True)
class PreparedContract:
    split: str
    block: str
    work_id: str
    normalized_title_sha256: str
    context: str
    context_sha256: str
    items: tuple[PreparedItem, ...]


@dataclass(frozen=True)
class PreparedSplit:
    split: str
    contracts: tuple[PreparedContract, ...]
    excluded_contract_count: int
    source_contract_count: int

    def contracts_for(self, block: str) -> tuple[PreparedContract, ...]:
        if block not in BLOCKS:
            raise MaudSourceError("unknown block")
        permitted = {
            "TRAIN": {"A_form", "F_search"},
            "DEV": {"A_hold"},
            "TEST": {"M_search"},
        }[self.split]
        if block not in permitted:
            raise MaudSourceError("block is incompatible with official split")
        return tuple(value for value in self.contracts if value.block == block)

    def action_view(self, block: str) -> dict[str, Any]:
        contracts = self.contracts_for(block)
        return {
            "schema": f"{VERSION}_label_free_action_view",
            "study_id": STUDY_ID,
            "split": self.split,
            "block": block,
            "contract_count": len(contracts),
            "item_count": sum(len(value.items) for value in contracts),
            "contracts": [
                {
                    "contract_work_id": contract.work_id,
                    "context": contract.context,
                    "context_sha256": contract.context_sha256,
                    "items": [
                        {
                            "work_id": item.work_id,
                            "question": item.question,
                            "deal_point_type": item.deal_point_type,
                            "family": item.family,
                        }
                        for item in contract.items
                    ],
                }
                for contract in contracts
            ],
            "answerability_gold_text_offset_or_span_included": False,
        }

    def gold_pack(self, block: str) -> dict[str, Any]:
        if block == "F_search":
            raise MaudSourceError("F_search gold pack is forbidden")
        contracts = self.contracts_for(block)
        if any(
            item.spans is None or item.merged_intervals is None
            for contract in contracts
            for item in contract.items
        ):
            raise MaudSourceError("gold was not semantically opened")
        body: dict[str, Any] = {
            "schema": f"{VERSION}_private_gold_pack",
            "study_id": STUDY_ID,
            "split": self.split,
            "block": block,
            "contract_count": len(contracts),
            "item_count": sum(len(value.items) for value in contracts),
            "contracts": [
                {
                    "contract_work_id": contract.work_id,
                    "items": [
                        {
                            "work_id": item.work_id,
                            "spans": [
                                {
                                    "start": span.start,
                                    "end": span.end,
                                    "text": span.text,
                                }
                                for span in item.spans or ()
                            ],
                            "merged_intervals": [
                                [start, end]
                                for start, end in (
                                    item.merged_intervals or ()
                                )
                            ],
                        }
                        for item in contract.items
                    ],
                }
                for contract in contracts
            ],
        }
        body["gold_pack_sha256"] = stable_hash(body)
        return body


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
        raise MaudSourceError("value is not canonical JSON") from exc
    return raw + (b"\n" if newline else b"")


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def normalize_contract_title(value: object) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise MaudSourceError("contract title is invalid")
    normalized = " ".join(
        unicodedata.normalize("NFKC", value).casefold().split()
    ).strip()
    if not normalized:
        raise MaudSourceError("contract title is empty")
    return normalized


def contract_title_sha256(value: object) -> str:
    return hashlib.sha256(
        normalize_contract_title(value).encode("utf-8")
    ).hexdigest()


def _normalize_type(value: str) -> str:
    normalized = " ".join(
        unicodedata.normalize("NFKC", value).casefold().split()
    ).strip()
    if normalized.endswith("."):
        normalized = normalized[:-1].rstrip()
    return normalized


_NORMALIZED_CANONICAL_TYPES = {
    _normalize_type(value): value for value in DEAL_POINT_TYPES
}
_NORMALIZED_ALIASES = {
    _normalize_type(alias): canonical
    for alias, canonical in TYPE_ALIASES.items()
}


def deal_point_type_and_family(question: object) -> tuple[str, str]:
    if not isinstance(question, str) or "\x00" in question:
        raise MaudSourceError("question is invalid")
    collapsed = " ".join(unicodedata.normalize("NFKC", question).split())
    if collapsed.endswith("."):
        collapsed = collapsed[:-1]
    if not (
        collapsed.startswith(QUESTION_PREFIX)
        and collapsed.endswith(QUESTION_SUFFIX)
    ):
        raise MaudSourceError("question does not match the frozen template")
    quoted = collapsed[
        len(QUESTION_PREFIX) : -len(QUESTION_SUFFIX)
    ].strip()
    quote_pairs = {'"': '"', "“": "”"}
    if (
        len(quoted) < 3
        or quoted[0] not in quote_pairs
        or quoted[-1] != quote_pairs[quoted[0]]
    ):
        raise MaudSourceError("question type is not quoted")
    normalized_type = _normalize_type(quoted[1:-1])
    canonical = _NORMALIZED_CANONICAL_TYPES.get(normalized_type)
    if canonical is None:
        canonical = _NORMALIZED_ALIASES.get(normalized_type)
    if canonical is None:
        raise MaudSourceError("question has an unknown deal-point type")
    return canonical, _TYPE_TO_FAMILY[canonical]


def question_for_type(deal_point_type: str) -> str:
    """Return the frozen public template, primarily for source-free fixtures."""

    if deal_point_type not in DEAL_POINT_TYPES and deal_point_type not in TYPE_ALIASES:
        raise MaudSourceError("unknown public deal-point type")
    return f'{QUESTION_PREFIX}"{deal_point_type}"{QUESTION_SUFFIX}.'


def _selection_secret(secret: object) -> bytes:
    if not isinstance(secret, bytes) or len(secret) != 32:
        raise MaudSourceError("selection secret must be exactly 32 bytes")
    return secret


def _hmac_hex(secret: bytes, purpose: str, *parts: str) -> str:
    message = "\0".join((VERSION, purpose, *parts)).encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def assign_block(split: str, normalized_title: str, secret: bytes) -> str:
    """Assign a nonexcluded contract to its one frozen contract-level block."""

    if split not in SPLITS:
        raise MaudSourceError("unknown official split")
    key = _selection_secret(secret)
    if split == "DEV":
        return "A_hold"
    if split == "TEST":
        return "M_search"
    digest = hmac.new(
        key,
        "\0".join(
            (VERSION, "TRAIN_contract_partition", normalized_title)
        ).encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return "A_form" if int.from_bytes(digest, "big") % 5 < 4 else "F_search"


class _RawCursor:
    """Non-materializing JSON syntax cursor over one in-memory raw value."""

    def __init__(self, text: str, *, label: str) -> None:
        self.text = text
        self.position = 0
        self.label = label

    def _peek(self) -> str | None:
        if self.position >= len(self.text):
            return None
        return self.text[self.position]

    def _take(self) -> str:
        value = self._peek()
        if value is None:
            raise MaudSourceError(f"{self.label} ended unexpectedly")
        self.position += 1
        return value

    def _skip_ws(self) -> None:
        while (value := self._peek()) is not None and value in " \t\r\n":
            self.position += 1

    def _expect(self, expected: str) -> None:
        self._skip_ws()
        if self._take() != expected:
            raise MaudSourceError(f"{self.label} JSON syntax drifted")

    def _skip_string(self) -> None:
        if self._take() != '"':
            raise MaudSourceError(f"{self.label} expected a JSON string")
        while True:
            value = self._take()
            if value == '"':
                return
            if ord(value) < 0x20:
                raise MaudSourceError(
                    f"{self.label} has an unescaped control character"
                )
            if value != "\\":
                continue
            escape = self._take()
            if escape in '"\\/bfnrt':
                continue
            if escape != "u":
                raise MaudSourceError(f"{self.label} has an invalid escape")
            for _ in range(4):
                if self._take() not in "0123456789abcdefABCDEF":
                    raise MaudSourceError(
                        f"{self.label} has an invalid Unicode escape"
                    )

    def _skip_primitive(self) -> None:
        start = self.position
        while (
            (value := self._peek()) is not None
            and value not in " \t\r\n,]}"
        ):
            self.position += 1
        token = self.text[start : self.position]
        if token not in {"true", "false", "null"} and _NUMBER.fullmatch(
            token
        ) is None:
            raise MaudSourceError(f"{self.label} has an invalid primitive")

    def skip_value(self) -> None:
        self._skip_ws()
        value = self._peek()
        if value is None:
            raise MaudSourceError(f"{self.label} has no JSON value")
        if value == '"':
            self._skip_string()
            return
        if value == "{":
            self.position += 1
            self._skip_ws()
            if self._peek() == "}":
                self.position += 1
                return
            while True:
                self._skip_ws()
                self._skip_string()
                self._expect(":")
                self.skip_value()
                self._skip_ws()
                delimiter = self._take()
                if delimiter == "}":
                    return
                if delimiter != ",":
                    raise MaudSourceError(
                        f"{self.label} object delimiter drifted"
                    )
        if value == "[":
            self.position += 1
            self._skip_ws()
            if self._peek() == "]":
                self.position += 1
                return
            while True:
                self.skip_value()
                self._skip_ws()
                delimiter = self._take()
                if delimiter == "]":
                    return
                if delimiter != ",":
                    raise MaudSourceError(
                        f"{self.label} array delimiter drifted"
                    )
        self._skip_primitive()

    def take_value(self) -> str:
        self._skip_ws()
        start = self.position
        self.skip_value()
        return self.text[start : self.position]

    def finish(self) -> None:
        self._skip_ws()
        if self.position != len(self.text):
            raise MaudSourceError(f"{self.label} has trailing JSON bytes")


class _StreamingCursor:
    """Incremental UTF-8 JSON syntax cursor used only for the top-level file."""

    def __init__(
        self,
        reader: BinaryIO | TextIO,
        *,
        chunk_size: int,
    ) -> None:
        if type(chunk_size) is not int or chunk_size < 1:
            raise MaudSourceError("stream chunk size is invalid")
        self.reader = reader
        self.chunk_size = chunk_size
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.mode: str | None = None
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.first_text = True

    def _read_more(self) -> None:
        if self.eof:
            return
        try:
            chunk = self.reader.read(self.chunk_size)
        except Exception as exc:
            raise MaudSourceError("source stream read failed") from exc
        if chunk in (b"", ""):
            self.eof = True
            if self.mode == "bytes":
                try:
                    tail = self.decoder.decode(b"", final=True)
                except UnicodeDecodeError as exc:
                    raise MaudSourceError("source is not strict UTF-8") from exc
                self._append_text(tail)
            return
        if isinstance(chunk, bytes):
            if self.mode not in (None, "bytes"):
                raise MaudSourceError("source stream changed read type")
            self.mode = "bytes"
            try:
                text = self.decoder.decode(chunk, final=False)
            except UnicodeDecodeError as exc:
                raise MaudSourceError("source is not strict UTF-8") from exc
        elif isinstance(chunk, str):
            if self.mode not in (None, "text"):
                raise MaudSourceError("source stream changed read type")
            self.mode = "text"
            text = chunk
        else:
            raise MaudSourceError("source stream returned a non-text value")
        self._append_text(text)

    def _append_text(self, text: str) -> None:
        if self.first_text and text:
            self.first_text = False
            if text.startswith("\ufeff"):
                raise MaudSourceError("source UTF-8 BOM is forbidden")
        self.buffer += text

    def _peek(self) -> str | None:
        while self.position >= len(self.buffer) and not self.eof:
            self._read_more()
        if self.position >= len(self.buffer):
            return None
        return self.buffer[self.position]

    def _take(self) -> str:
        value = self._peek()
        if value is None:
            raise MaudSourceError("source JSON ended unexpectedly")
        self.position += 1
        return value

    def _skip_ws(self) -> None:
        while (value := self._peek()) is not None and value in " \t\r\n":
            self.position += 1

    def _expect(self, expected: str) -> None:
        self._skip_ws()
        if self._take() != expected:
            raise MaudSourceError("source JSON syntax drifted")

    def _skip_string(self) -> None:
        if self._take() != '"':
            raise MaudSourceError("source expected a JSON string")
        while True:
            value = self._take()
            if value == '"':
                return
            if ord(value) < 0x20:
                raise MaudSourceError(
                    "source has an unescaped control character"
                )
            if value != "\\":
                continue
            escape = self._take()
            if escape in '"\\/bfnrt':
                continue
            if escape != "u":
                raise MaudSourceError("source has an invalid escape")
            for _ in range(4):
                if self._take() not in "0123456789abcdefABCDEF":
                    raise MaudSourceError(
                        "source has an invalid Unicode escape"
                    )

    def _skip_primitive(self) -> None:
        start = self.position
        while (
            (value := self._peek()) is not None
            and value not in " \t\r\n,]}"
        ):
            self.position += 1
        token = self.buffer[start : self.position]
        if token not in {"true", "false", "null"} and _NUMBER.fullmatch(
            token
        ) is None:
            raise MaudSourceError("source has an invalid primitive")

    def skip_value(self) -> None:
        self._skip_ws()
        value = self._peek()
        if value is None:
            raise MaudSourceError("source has no JSON value")
        if value == '"':
            self._skip_string()
            return
        if value == "{":
            self.position += 1
            self._skip_ws()
            if self._peek() == "}":
                self.position += 1
                return
            while True:
                self._skip_ws()
                self._skip_string()
                self._expect(":")
                self.skip_value()
                self._skip_ws()
                delimiter = self._take()
                if delimiter == "}":
                    return
                if delimiter != ",":
                    raise MaudSourceError(
                        "source object delimiter drifted"
                    )
        if value == "[":
            self.position += 1
            self._skip_ws()
            if self._peek() == "]":
                self.position += 1
                return
            while True:
                self.skip_value()
                self._skip_ws()
                delimiter = self._take()
                if delimiter == "]":
                    return
                if delimiter != ",":
                    raise MaudSourceError(
                        "source array delimiter drifted"
                    )
        self._skip_primitive()

    def take_value(self) -> str:
        self._skip_ws()
        start = self.position
        self.skip_value()
        return self.buffer[start : self.position]

    def parse_string(self) -> str:
        self._skip_ws()
        start = self.position
        self._skip_string()
        return _decode_json_string(self.buffer[start : self.position])

    def compact(self) -> None:
        if self.position:
            self.buffer = self.buffer[self.position :]
            self.position = 0

    def finish(self) -> None:
        self._skip_ws()
        if self._peek() is not None:
            raise MaudSourceError("source has trailing JSON bytes")


def _decode_json_string(raw: str) -> str:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise MaudSourceError("JSON string failed to decode") from exc
    if not isinstance(value, str):
        raise MaudSourceError("JSON value is not a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise MaudSourceError("JSON string contains an unpaired surrogate") from exc
    return value


def _decode_json_integer(raw: str, *, label: str) -> int:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise MaudSourceError(f"{label} failed to decode") from exc
    if type(value) is not int:
        raise MaudSourceError(f"{label} is not an integer")
    return value


def _decode_json_boolean(raw: str, *, label: str) -> bool:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise MaudSourceError(f"{label} failed to decode") from exc
    if type(value) is not bool:
        raise MaudSourceError(f"{label} is not a boolean")
    return value


def _object_fields(raw: str, *, label: str) -> dict[str, str]:
    cursor = _RawCursor(raw, label=label)
    cursor._expect("{")
    fields: dict[str, str] = {}
    cursor._skip_ws()
    if cursor._peek() == "}":
        cursor.position += 1
        cursor.finish()
        return fields
    while True:
        cursor._skip_ws()
        start = cursor.position
        cursor._skip_string()
        key = _decode_json_string(raw[start : cursor.position])
        if key in fields:
            raise MaudSourceError(f"{label} has duplicate field {key!r}")
        cursor._expect(":")
        fields[key] = cursor.take_value()
        cursor._skip_ws()
        delimiter = cursor._take()
        if delimiter == "}":
            cursor.finish()
            return fields
        if delimiter != ",":
            raise MaudSourceError(f"{label} object delimiter drifted")


def _array_values(raw: str, *, label: str) -> tuple[str, ...]:
    cursor = _RawCursor(raw, label=label)
    cursor._expect("[")
    values: list[str] = []
    cursor._skip_ws()
    if cursor._peek() == "]":
        cursor.position += 1
        cursor.finish()
        return ()
    while True:
        values.append(cursor.take_value())
        cursor._skip_ws()
        delimiter = cursor._take()
        if delimiter == "]":
            cursor.finish()
            return tuple(values)
        if delimiter != ",":
            raise MaudSourceError(f"{label} array delimiter drifted")


def _iter_raw_contracts(
    reader: BinaryIO | TextIO,
    *,
    chunk_size: int,
) -> Iterator[str]:
    cursor = _StreamingCursor(reader, chunk_size=chunk_size)
    cursor._expect("{")
    seen: set[str] = set()
    found_data = False
    cursor._skip_ws()
    if cursor._peek() == "}":
        raise MaudSourceError("source top-level object is empty")
    while True:
        key = cursor.parse_string()
        if key in seen:
            raise MaudSourceError(
                f"source top-level has duplicate field {key!r}"
            )
        seen.add(key)
        cursor._expect(":")
        if key != "data":
            cursor.skip_value()
        else:
            found_data = True
            cursor._expect("[")
            cursor._skip_ws()
            if cursor._peek() != "]":
                while True:
                    raw = cursor.take_value()
                    yield raw
                    cursor.compact()
                    cursor._skip_ws()
                    delimiter = cursor._take()
                    if delimiter == "]":
                        break
                    if delimiter != ",":
                        raise MaudSourceError(
                            "source data array delimiter drifted"
                        )
            else:
                cursor.position += 1
        cursor._skip_ws()
        delimiter = cursor._take()
        if delimiter == "}":
            break
        if delimiter != ",":
            raise MaudSourceError("source top-level delimiter drifted")
        cursor.compact()
    if not found_data:
        raise MaudSourceError("source top-level data array is absent")
    cursor.finish()


def _merge_intervals(
    spans: Sequence[GoldSpan],
) -> tuple[tuple[int, int], ...]:
    merged: list[list[int]] = []
    for span in spans:
        if not merged or span.start > merged[-1][1]:
            merged.append([span.start, span.end])
        else:
            merged[-1][1] = max(merged[-1][1], span.end)
    return tuple((start, end) for start, end in merged)


def _parse_answers(
    raw: str,
    *,
    context: str,
    item_label: str,
) -> tuple[GoldSpan, ...]:
    observed: set[tuple[int, int, str]] = set()
    for index, answer_raw in enumerate(
        _array_values(raw, label=f"{item_label} answers")
    ):
        fields = _object_fields(
            answer_raw, label=f"{item_label} answer {index}"
        )
        if "text" not in fields or "answer_start" not in fields:
            raise MaudSourceError(
                f"{item_label} answer lacks text or answer_start"
            )
        text = _decode_json_string(fields["text"])
        start = _decode_json_integer(
            fields["answer_start"], label=f"{item_label} answer_start"
        )
        if not text or start < 0:
            raise MaudSourceError(f"{item_label} answer span is empty")
        end = start + len(text)
        if end > len(context) or context[start:end] != text:
            raise MaudSourceError(
                f"{item_label} answer does not match exact context offsets"
            )
        observed.add((start, end, text))
    return tuple(GoldSpan(*value) for value in sorted(observed))


def _parse_item(
    raw: str,
    *,
    context: str,
    include_gold: bool,
    contract_work_id: str,
    secret: bytes,
    ordinal: int,
) -> tuple[str, PreparedItem]:
    fields = _object_fields(raw, label=f"QA {ordinal}")
    for required in ("id", "question", "answers", "is_impossible"):
        if required not in fields:
            raise MaudSourceError(f"QA {ordinal} lacks {required}")
    source_id = _decode_json_string(fields["id"])
    question = _decode_json_string(fields["question"])
    if not source_id or "\x00" in source_id:
        raise MaudSourceError(f"QA {ordinal} ID is invalid")
    deal_point_type, family = deal_point_type_and_family(question)
    work_id = _hmac_hex(
        secret,
        "item_work_id",
        contract_work_id,
        source_id,
        deal_point_type,
    )
    if not include_gold:
        return (
            source_id,
            PreparedItem(
                work_id=work_id,
                question=question,
                deal_point_type=deal_point_type,
                family=family,
                spans=None,
                merged_intervals=None,
            ),
        )
    spans = _parse_answers(
        fields["answers"], context=context, item_label=f"QA {ordinal}"
    )
    impossible = _decode_json_boolean(
        fields["is_impossible"], label=f"QA {ordinal} is_impossible"
    )
    if impossible != (not spans):
        raise MaudSourceError(
            f"QA {ordinal} is_impossible disagrees with answers"
        )
    return (
        source_id,
        PreparedItem(
            work_id=work_id,
            question=question,
            deal_point_type=deal_point_type,
            family=family,
            spans=spans,
            merged_intervals=_merge_intervals(spans),
        ),
    )


def _parse_contract(
    raw: str,
    *,
    split: str,
    secret: bytes,
) -> tuple[str, PreparedContract | None]:
    fields = _object_fields(raw, label="contract")
    if "title" not in fields:
        raise MaudSourceError("contract lacks title")
    title = _decode_json_string(fields["title"])
    normalized_title = normalize_contract_title(title)
    title_sha256 = contract_title_sha256(title)
    if title_sha256 in EXCLUDED_NORMALIZED_TITLE_SHA256S:
        return title_sha256, None
    if "paragraphs" not in fields:
        raise MaudSourceError("contract lacks paragraphs")
    block = assign_block(split, normalized_title, secret)
    paragraph_values = _array_values(
        fields["paragraphs"], label="contract paragraphs"
    )
    if len(paragraph_values) != 1:
        raise MaudSourceError("contract must contain exactly one paragraph")
    paragraph = _object_fields(
        paragraph_values[0], label="contract paragraph"
    )
    if "context" not in paragraph or "qas" not in paragraph:
        raise MaudSourceError("contract paragraph lacks context or qas")
    context = _decode_json_string(paragraph["context"])
    if not context:
        raise MaudSourceError("contract context is empty")
    context_sha256 = hashlib.sha256(context.encode("utf-8")).hexdigest()
    contract_work_id = _hmac_hex(
        secret, "contract_work_id", split, normalized_title
    )
    include_gold = block != "F_search"
    parsed_items = [
        _parse_item(
            qa_raw,
            context=context,
            include_gold=include_gold,
            contract_work_id=contract_work_id,
            secret=secret,
            ordinal=index,
        )
        for index, qa_raw in enumerate(
            _array_values(paragraph["qas"], label="contract qas")
        )
    ]
    if len(parsed_items) != len(DEAL_POINT_TYPES):
        raise MaudSourceError("contract must contain exactly 22 QAs")
    source_ids = [source_id for source_id, _item in parsed_items]
    items = [item for _source_id, item in parsed_items]
    if len(set(source_ids)) != len(source_ids):
        raise MaudSourceError("contract has duplicate QA IDs")
    observed_types = [item.deal_point_type for item in items]
    if len(set(observed_types)) != len(observed_types):
        raise MaudSourceError("contract has duplicate deal-point types")
    if set(observed_types) != set(DEAL_POINT_TYPES):
        raise MaudSourceError("contract deal-point registry is incomplete")
    items.sort(key=lambda item: _TYPE_ORDER[item.deal_point_type])
    return title_sha256, PreparedContract(
        split=split,
        block=block,
        work_id=contract_work_id,
        normalized_title_sha256=title_sha256,
        context=context,
        context_sha256=context_sha256,
        items=tuple(items),
    )


def parse_split(
    source: str | os.PathLike[str] | BinaryIO | TextIO,
    *,
    split: str,
    selection_secret: bytes,
    test_parse_capability: TestParseCapability | None = None,
    stream_chunk_size: int = 64 * 1024,
) -> PreparedSplit:
    """Parse one official split through the frozen private source boundary.

    TEST authority is checked before a path is opened or a supplied stream is
    read.  For TRAIN, the HMAC partition is contract-level and all 22 queries
    from a contract remain together.
    """

    if split not in SPLITS:
        raise MaudSourceError("split must be TRAIN, DEV, or TEST")
    secret = _selection_secret(selection_secret)
    if split == "TEST":
        if not isinstance(test_parse_capability, TestParseCapability):
            raise MaudSourceError(
                "TEST source requires an explicit parse capability"
            )
        test_parse_capability.validate()
    elif test_parse_capability is not None:
        raise MaudSourceError("TEST parse capability supplied to non-TEST")

    if isinstance(source, (str, os.PathLike)):
        context = open(Path(source), "rb")
    elif hasattr(source, "read"):
        context = nullcontext(source)
    else:
        raise MaudSourceError("source must be a path or readable stream")

    contracts: list[PreparedContract] = []
    seen_title_hashes: set[str] = set()
    excluded_count = 0
    source_count = 0
    with context as reader:
        for raw in _iter_raw_contracts(
            reader, chunk_size=stream_chunk_size
        ):
            source_count += 1
            title_hash, contract = _parse_contract(
                raw, split=split, secret=secret
            )
            if title_hash in seen_title_hashes:
                raise MaudSourceError(
                    "split has a duplicate normalized contract title"
                )
            seen_title_hashes.add(title_hash)
            if contract is None:
                excluded_count += 1
            else:
                contracts.append(contract)
    if source_count == 0:
        raise MaudSourceError("source data array is empty")
    return PreparedSplit(
        split=split,
        contracts=tuple(contracts),
        excluded_contract_count=excluded_count,
        source_contract_count=source_count,
    )


def write_gold_pack_exclusive(
    path: str | os.PathLike[str],
    gold_pack: Mapping[str, Any],
) -> str:
    """Write one non-F gold pack with O_EXCL/O_NOFOLLOW and exact mode 0600."""

    if not isinstance(gold_pack, Mapping):
        raise MaudSourceError("gold pack is not an object")
    if gold_pack.get("schema") != f"{VERSION}_private_gold_pack":
        raise MaudSourceError("gold pack schema is invalid")
    if gold_pack.get("block") == "F_search":
        raise MaudSourceError("F_search gold pack is forbidden")
    body = dict(gold_pack)
    declared = body.pop("gold_pack_sha256", None)
    if (
        not isinstance(declared, str)
        or _HEX64.fullmatch(declared) is None
        or stable_hash(body) != declared
    ):
        raise MaudSourceError("gold pack self-hash is invalid")
    destination = Path(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(destination, flags, 0o600)
    except OSError as exc:
        raise MaudSourceError("exclusive gold pack creation failed") from exc
    try:
        os.fchmod(descriptor, 0o600)
        raw = _canonical_bytes(gold_pack, newline=True)
        file_sha256 = hashlib.sha256(raw).hexdigest()
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise MaudSourceError("gold pack write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        try:
            os.close(descriptor)
        finally:
            try:
                destination.unlink()
            except OSError:
                pass
        raise
    else:
        os.close(descriptor)
    metadata = destination.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise MaudSourceError("gold pack mode or file type drifted")
    return file_sha256


__all__ = [
    "BLOCKS",
    "CONDITION_OBLIGATION_TYPES",
    "DEAL_POINT_TYPES",
    "DEFINITION_REFERENCE_TYPES",
    "EXCLUDED_NORMALIZED_TITLE_SHA256S",
    "GoldSpan",
    "MaudSourceError",
    "PROTECTION_EXCEPTION_REMEDY_TYPES",
    "PreparedContract",
    "PreparedItem",
    "PreparedSplit",
    "SPLITS",
    "STUDY_ID",
    "TYPE_ALIASES",
    "TestParseCapability",
    "VERSION",
    "assign_block",
    "contract_title_sha256",
    "deal_point_type_and_family",
    "normalize_contract_title",
    "parse_split",
    "question_for_type",
    "stable_hash",
    "write_gold_pack_exclusive",
]
