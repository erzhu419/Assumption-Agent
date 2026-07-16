from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


STUDY_ID = "noaa-gsod-auto-typed-operator-v1"
PACK_VERSION = "noaa_gsod_auto_typed_operator_pack_v1"
PUBLIC_RECEIPT_VERSION = "noaa_gsod_auto_typed_operator_acquisition_v1"
ORACLE_IDS = ("noaa_gsod_stdlib_decimal_v1", "noaa_gsod_sqlite_integer_v1")

YEAR = 2020
ITEM_COUNT = 24
PARTITION_COUNTS = {"train": 12, "development": 6, "sealed": 6}
SELECTION_SEED = "noaa-gsod-auto-typed-operator-v1:2020:station-out:24"
MIN_UNIQUE_DATES = 330
MIN_VALID_DAYS_PER_MONTH = 20

STATION_METADATA_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
GSOD_YEAR_INDEX_URL = (
    "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/2020/"
)
GSOD_FILE_URL_TEMPLATE = GSOD_YEAR_INDEX_URL + "{station_id}.csv"

REQUIRED_COLUMNS = frozenset({"STATION", "DATE", "PRCP"})
MISSING_PRCP_TOKENS = frozenset({"", "99.99"})
_STATION_ID = re.compile(r"^[0-9]{11}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# This is deliberately a relational program rather than a station-specific
# hand-written answer.  Its primitives are finite and can be represented by a
# later typed DSL: select -> normalize -> derive month -> group -> aggregate ->
# argmax(stable) -> convert -> round -> project -> canonical JSON.
TASK_CONTRACT: dict[str, Any] = {
    "contract_version": "noaa_gsod_wettest_month_contract_v1",
    "input": {
        "format": "RFC4180 CSV with one NOAA GSOD station for calendar year 2020",
        "required_columns": ["STATION", "DATE", "PRCP"],
        "row_scope": "DATE parses as YYYY-MM-DD and belongs to 2020",
    },
    "normalization": {
        "PRCP": {
            "source_unit": "inch",
            "decimal_parser": "base-10 Decimal after surrounding-whitespace trim",
            "missing_tokens": ["", "99.99"],
            "invalid_policy": "any other non-decimal or negative value invalidates the task",
        }
    },
    "relational_program": [
        "retain rows in calendar year 2020",
        "derive month as the two-digit DATE month",
        "drop rows whose PRCP is missing",
        "group by month",
        "aggregate sum(PRCP) and count(valid PRCP)",
        "derive mean_daily_precip_in = sum(PRCP) / count(valid PRCP)",
        "argmax mean_daily_precip_in; on an exact tie choose the earliest month",
        "convert the selected mean from inches to millimetres by multiplying by 25.4",
        "round millimetres to 0.01 with decimal ROUND_HALF_UP",
    ],
    "output": {
        "value": {
            "month": "two-digit string 01..12",
            "mean_daily_precip_mm": "fixed-point string with exactly two decimals",
            "valid_day_count": "base-10 integer",
        },
        "serialization": "UTF-8 canonical JSON: sorted keys, compact separators, no NaN",
    },
    "operator_capabilities": [
        "missing_normalization",
        "filter",
        "derive",
        "group",
        "aggregate_sum_count",
        "argmax",
        "stable_tie_break",
        "unit_conversion",
        "decimal_round_half_up",
        "json_serialize",
    ],
}


class NoaaGsodError(ValueError):
    """A source, item, oracle result, or pack violates the frozen contract."""


@dataclass(frozen=True)
class Completeness:
    eligible: bool
    unique_dates: int
    month_count: int
    minimum_valid_days_in_month: int
    row_count: int
    reason: str


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def with_self_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    if field in result:
        raise NoaaGsodError(f"{field} already exists")
    result[field] = payload_hash(result)
    return result


def verify_self_hash(payload: Mapping[str, Any], field: str) -> str:
    declared = payload.get(field)
    if not isinstance(declared, str) or _SHA256.fullmatch(declared) is None:
        raise NoaaGsodError(f"{field} is not a SHA-256 digest")
    body = dict(payload)
    del body[field]
    if payload_hash(body) != declared:
        raise NoaaGsodError(f"{field} mismatch")
    return declared


def validate_header(fieldnames: Iterable[str] | None) -> tuple[str, ...]:
    if fieldnames is None:
        raise NoaaGsodError("CSV has no header")
    header = tuple(str(value) for value in fieldnames)
    if len(header) != len(set(header)) or not REQUIRED_COLUMNS.issubset(header):
        raise NoaaGsodError("CSV header lacks unique required columns")
    return header


def parse_date_2020(text: object) -> date:
    try:
        parsed = date.fromisoformat(str(text or "").strip())
    except ValueError as exc:
        raise NoaaGsodError("DATE is not ISO YYYY-MM-DD") from exc
    if parsed.year != YEAR:
        raise NoaaGsodError("DATE lies outside calendar year 2020")
    return parsed


def parse_prcp(text: object) -> Decimal | None:
    token = str(text or "").strip()
    if token in MISSING_PRCP_TOKENS:
        return None
    try:
        value = Decimal(token)
    except InvalidOperation as exc:
        raise NoaaGsodError("PRCP is neither missing nor decimal") from exc
    if not value.is_finite() or value < 0:
        raise NoaaGsodError("PRCP is negative or non-finite")
    return value


def format_oracle_result(
    *, month: int, sum_inches: Decimal, valid_day_count: int
) -> dict[str, Any]:
    if month < 1 or month > 12 or valid_day_count <= 0:
        raise NoaaGsodError("oracle selected an invalid monthly aggregate")
    mean_mm = (sum_inches / Decimal(valid_day_count)) * Decimal("25.4")
    rounded = mean_mm.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return {
        "mean_daily_precip_mm": format(rounded, ".2f"),
        "month": f"{month:02d}",
        "valid_day_count": valid_day_count,
    }


def assess_completeness(path: str | Path, expected_station_id: str) -> Completeness:
    if _STATION_ID.fullmatch(expected_station_id) is None:
        raise NoaaGsodError("expected station id is invalid")
    unique_dates: set[date] = set()
    valid_by_month: dict[int, int] = defaultdict(int)
    row_count = 0
    try:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            validate_header(reader.fieldnames)
            for row in reader:
                row_count += 1
                if None in row:
                    raise NoaaGsodError("CSV row has excess fields")
                if str(row["STATION"] or "").strip() != expected_station_id:
                    raise NoaaGsodError("CSV contains another station")
                parsed_date = parse_date_2020(row["DATE"])
                if parsed_date in unique_dates:
                    raise NoaaGsodError("CSV contains duplicate DATE")
                unique_dates.add(parsed_date)
                if parse_prcp(row["PRCP"]) is not None:
                    valid_by_month[parsed_date.month] += 1
    except (OSError, UnicodeError, csv.Error, NoaaGsodError) as exc:
        return Completeness(False, 0, 0, 0, row_count, type(exc).__name__)
    month_count = len({value.month for value in unique_dates})
    minimum = min((valid_by_month.get(month, 0) for month in range(1, 13)), default=0)
    eligible = (
        len(unique_dates) >= MIN_UNIQUE_DATES
        and month_count == 12
        and minimum >= MIN_VALID_DAYS_PER_MONTH
    )
    reason = "eligible" if eligible else "insufficient_calendar_or_valid_day_coverage"
    return Completeness(
        eligible,
        len(unique_dates),
        month_count,
        minimum,
        row_count,
        reason,
    )
