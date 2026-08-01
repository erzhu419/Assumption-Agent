"""Strict, offline adapter for the frozen ARN CSV release.

The adapter is deliberately narrow.  It performs CSV decoding and canonical
field conversion only; it does not load a model, compute a prediction, inspect
an aggregate score, or emit any row content.  Formal source identity is checked
again immediately before parsing.

The pure :func:`parse_arn_csv_bytes` entry point exists only so the same parser
can be qualified with synthetic, source-free fixtures before the official
archive is opened.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import re
import stat
from typing import Sequence

from .benchmarks.gscl_arn_intrinsic_protocol_v1 import (
    AdaptedArnRow,
    OFFICIAL_CELL_COUNTS,
    OFFICIAL_DATASET_SHA256,
    OFFICIAL_DATASET_SIZE,
    OFFICIAL_HEADER,
    OFFICIAL_ID_MAXIMUM,
    OFFICIAL_ID_MINIMUM,
    OFFICIAL_MISSING_IDS,
    OFFICIAL_ROW_COUNT,
)


ADAPTER_VERSION = "gscl_arn_raw_adapter_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAXIMUM_FIELD_CHARACTERS = 131_072


class ArnRawAdapterError(RuntimeError):
    """The source or its canonical ARN representation drifted."""


@dataclass(frozen=True)
class ArnTopology:
    row_count: int
    id_minimum: int
    id_maximum: int
    missing_ids: tuple[int, ...]
    cell_counts: dict[str, int]


OFFICIAL_TOPOLOGY = ArnTopology(
    row_count=OFFICIAL_ROW_COUNT,
    id_minimum=OFFICIAL_ID_MINIMUM,
    id_maximum=OFFICIAL_ID_MAXIMUM,
    missing_ids=OFFICIAL_MISSING_IDS,
    cell_counts=dict(OFFICIAL_CELL_COUNTS),
)


def _read_exact_regular_file(
    path: Path, *, expected_size: int, expected_sha256: str
) -> bytes:
    """Read one immutable regular file through a single no-follow descriptor."""

    if not isinstance(path, Path):
        raise ArnRawAdapterError("source path type drifted")
    try:
        before = path.lstat()
    except OSError as exc:
        raise ArnRawAdapterError("official ARN source is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
        or before.st_size != expected_size
    ):
        raise ArnRawAdapterError("official ARN source topology drifted")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArnRawAdapterError("official ARN source open failed") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_nlink != 1
            or opened.st_size != before.st_size
        ):
            raise ArnRawAdapterError("official ARN source changed at open")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
            or after.st_nlink != 1
        ):
            raise ArnRawAdapterError("official ARN source changed while read")
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if (
        len(raw) != expected_size
        or digest.hexdigest() != expected_sha256
    ):
        raise ArnRawAdapterError("official ARN source identity drifted")
    return raw


def _text(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or len(value) > _MAXIMUM_FIELD_CHARACTERS
    ):
        raise ArnRawAdapterError(f"{field} is invalid")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ArnRawAdapterError(f"{field} is not strict UTF-8") from exc
    return value


def _source_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.isdecimal()
        or (len(value) > 1 and value.startswith("0"))
        or int(value) <= 0
    ):
        raise ArnRawAdapterError("source id is not canonical")
    return value


def _normalized_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _binary_level(
    value: str, *, allowed: tuple[str, str], field: str
) -> str:
    normalized = _normalized_token(value)
    if normalized not in allowed:
        raise ArnRawAdapterError(f"{field} is not canonical")
    return normalized


_FIRST_ALIASES = frozenset(
    {
        "1",
        "a",
        "answer_1",
        "answer_a",
        "choice_1",
        "choice_a",
        "first",
        "first_choice",
    }
)
_SECOND_ALIASES = frozenset(
    {
        "2",
        "b",
        "answer_2",
        "answer_b",
        "choice_2",
        "choice_b",
        "second",
        "second_choice",
    }
)


def _gold_choice(
    value: str, *, first_choice: str, second_choice: str
) -> str:
    """Map only frozen, unambiguous answer encodings.

    Exact narrative equality is checked before symbolic aliases.  The adapter
    intentionally does not guess a zero-based numeric convention.
    """

    if value == first_choice and value != second_choice:
        return "first_choice"
    if value == second_choice and value != first_choice:
        return "second_choice"
    normalized = _normalized_token(value)
    if normalized in _FIRST_ALIASES:
        return "first_choice"
    if normalized in _SECOND_ALIASES:
        return "second_choice"
    raise ArnRawAdapterError("correct_answer encoding is unsupported")


def _validate_topology(
    rows: Sequence[AdaptedArnRow], expected: ArnTopology
) -> None:
    if len(rows) != expected.row_count:
        raise ArnRawAdapterError("ARN row count drifted")
    ids = [int(row.source_id) for row in rows]
    if (
        not ids
        or min(ids) != expected.id_minimum
        or max(ids) != expected.id_maximum
        or tuple(
            value
            for value in range(expected.id_minimum, expected.id_maximum + 1)
            if value not in set(ids)
        )
        != expected.missing_ids
    ):
        raise ArnRawAdapterError("ARN id topology drifted")
    cell_counts = {
        f"{analogy}_{distractor}": sum(
            row.analogy_level == analogy
            and row.distractor_similarity == distractor
            for row in rows
        )
        for analogy in ("far", "near")
        for distractor in ("high", "low")
    }
    if cell_counts != expected.cell_counts:
        raise ArnRawAdapterError("ARN four-cell topology drifted")


def parse_arn_csv_bytes(
    raw: bytes, *, expected_topology: ArnTopology | None
) -> tuple[AdaptedArnRow, ...]:
    """Parse strict CSV bytes without logging or returning source statistics."""

    if not isinstance(raw, bytes) or not raw:
        raise ArnRawAdapterError("ARN CSV bytes are invalid")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ArnRawAdapterError("ARN CSV is not strict UTF-8") from exc
    reader = csv.reader(
        io.StringIO(text, newline=""),
        dialect="excel",
        strict=True,
    )
    try:
        header = next(reader)
    except (StopIteration, csv.Error) as exc:
        raise ArnRawAdapterError("ARN CSV header is unavailable") from exc
    if tuple(header) != OFFICIAL_HEADER:
        raise ArnRawAdapterError("ARN CSV header drifted")

    rows: list[AdaptedArnRow] = []
    seen_ids: set[str] = set()
    try:
        for fields in reader:
            if len(fields) != len(OFFICIAL_HEADER):
                raise ArnRawAdapterError("ARN CSV row width drifted")
            record = dict(zip(OFFICIAL_HEADER, fields, strict=True))
            source_id = _source_id(record["id"])
            if source_id in seen_ids:
                raise ArnRawAdapterError("ARN source id is duplicated")
            seen_ids.add(source_id)
            proverb = _text(record["proverb"], field="proverb")
            query = _text(
                record["query_narrative"], field="query_narrative"
            )
            first = _text(record["first_choice"], field="first_choice")
            second = _text(record["second_choice"], field="second_choice")
            if first == second:
                raise ArnRawAdapterError("ARN choices are identical")
            answer = _text(
                record["correct_answer"], field="correct_answer"
            )
            rows.append(
                AdaptedArnRow(
                    source_id=source_id,
                    proverb=proverb,
                    query_narrative=query,
                    first_choice=first,
                    second_choice=second,
                    gold_choice=_gold_choice(
                        answer,
                        first_choice=first,
                        second_choice=second,
                    ),
                    analogy_level=_binary_level(
                        record["analogy_level"],
                        allowed=("far", "near"),
                        field="analogy_level",
                    ),
                    distractor_similarity=_binary_level(
                        record["distractor_similarity"],
                        allowed=("high", "low"),
                        field="distractor_similarity",
                    ),
                )
            )
    except csv.Error as exc:
        raise ArnRawAdapterError("ARN CSV quoting drifted") from exc
    if not rows:
        raise ArnRawAdapterError("ARN CSV contains no rows")
    result = tuple(rows)
    if expected_topology is not None:
        _validate_topology(result, expected_topology)
    return result


class OfficialArnRawNarrativeAdapter:
    """Content-addressed implementation of the protocol adapter interface."""

    def __init__(self, *, qualification_receipt_sha256: str) -> None:
        if _SHA256.fullmatch(qualification_receipt_sha256) is None:
            raise ArnRawAdapterError(
                "adapter qualification receipt hash is invalid"
            )
        self.qualification_receipt_sha256 = (
            qualification_receipt_sha256
        )
        self.implementation_sha256 = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()

    def adapt(self, source_path: Path) -> tuple[AdaptedArnRow, ...]:
        raw = _read_exact_regular_file(
            source_path,
            expected_size=OFFICIAL_DATASET_SIZE,
            expected_sha256=OFFICIAL_DATASET_SHA256,
        )
        return parse_arn_csv_bytes(raw, expected_topology=OFFICIAL_TOPOLOGY)


__all__ = [
    "ADAPTER_VERSION",
    "ArnRawAdapterError",
    "ArnTopology",
    "OFFICIAL_TOPOLOGY",
    "OfficialArnRawNarrativeAdapter",
    "parse_arn_csv_bytes",
]
