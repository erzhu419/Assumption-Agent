from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path
import sqlite3
from typing import Any

from .contract import (
    NoaaGsodError,
    format_oracle_result,
    parse_date_2020,
    parse_prcp,
    validate_header,
)


ORACLE_ID = "noaa_gsod_sqlite_integer_v1"


def _hundredths(value: Decimal) -> int:
    scaled = value * 100
    if scaled != scaled.to_integral_value():
        raise NoaaGsodError("PRCP has more than two decimal places")
    return int(scaled)


def evaluate(path: str | Path) -> dict[str, Any]:
    # This implementation deliberately uses a different aggregation engine and
    # exact integer representation from the streaming Decimal oracle.
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(
            "CREATE TABLE observations (month INTEGER NOT NULL, prcp_hundredths INTEGER NOT NULL)"
        )
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
                connection.execute(
                    "INSERT INTO observations(month, prcp_hundredths) VALUES (?, ?)",
                    (parsed_date.month, _hundredths(precipitation)),
                )
        aggregates = list(
            connection.execute(
                "SELECT month, SUM(prcp_hundredths), COUNT(*) "
                "FROM observations GROUP BY month ORDER BY month ASC"
            )
        )
    finally:
        connection.close()
    if not aggregates:
        raise NoaaGsodError("oracle found no valid PRCP")

    best_month, best_sum, best_count = (int(value) for value in aggregates[0])
    for raw_month, raw_sum, raw_count in aggregates[1:]:
        month, total, count = int(raw_month), int(raw_sum), int(raw_count)
        if total * best_count > best_sum * count:
            best_month, best_sum, best_count = month, total, count
    return format_oracle_result(
        month=best_month,
        sum_inches=Decimal(best_sum) / 100,
        valid_day_count=best_count,
    )
