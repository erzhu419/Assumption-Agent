from __future__ import annotations

from collections import defaultdict
import csv
from decimal import Decimal
from pathlib import Path
from typing import Any

from .contract import (
    NoaaGsodError,
    format_oracle_result,
    parse_date_2020,
    parse_prcp,
    validate_header,
)


ORACLE_ID = "noaa_gsod_stdlib_decimal_v1"


def evaluate(path: str | Path) -> dict[str, Any]:
    sums: dict[int, Decimal] = defaultdict(Decimal)
    counts: dict[int, int] = defaultdict(int)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_header(reader.fieldnames)
        for row in reader:
            if None in row:
                raise NoaaGsodError("CSV row has excess fields")
            parsed_date = parse_date_2020(row["DATE"])
            precipitation = parse_prcp(row["PRCP"])
            if precipitation is None:
                continue
            sums[parsed_date.month] += precipitation
            counts[parsed_date.month] += 1
    if not counts:
        raise NoaaGsodError("oracle found no valid PRCP")

    # Cross multiplication preserves exact Decimal comparison and the sorted
    # scan makes the earliest month the explicit stable tie-break.
    best_month: int | None = None
    for month in sorted(counts):
        if best_month is None:
            best_month = month
            continue
        left = sums[month] * counts[best_month]
        right = sums[best_month] * counts[month]
        if left > right:
            best_month = month
    assert best_month is not None
    return format_oracle_result(
        month=best_month,
        sum_inches=sums[best_month],
        valid_day_count=counts[best_month],
    )
