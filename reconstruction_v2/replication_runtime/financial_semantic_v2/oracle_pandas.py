from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .pack import (
    COVERPAGE_COLUMNS,
    INFOTABLE_COLUMNS,
    STOCK_TITLE_CLASSES_V1,
    PeriodOutPackError,
    Sec13FSource,
    build_oracle_output,
    canonical_cusip,
    decimal_to_json_number,
    normalize_name,
    normalize_title_class,
    parse_sec_value,
    partition_items,
    read_json,
    validate_source_against_pack,
    verify_public_pack,
    write_json,
)


ORACLE_ID = "sec13f_pandas_chunked_v1"
CHUNK_ROWS = 250_000


@dataclass(frozen=True)
class _Snapshot:
    accession_to_manager: Mapping[str, str]
    display_by_manager: Mapping[str, str]
    unique_accession_by_manager: Mapping[str, str]


def _pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - deployment preflight.
        raise PeriodOutPackError("pandas oracle runtime lacks pandas") from exc
    return pd


def _validate_header(
    source: Sec13FSource, table: str, required: frozenset[str]
) -> None:
    with source.open_table(table) as handle:
        header = next(csv.reader(handle, delimiter="\t"), None)
    if (
        not isinstance(header, list)
        or len(header) != len(set(header))
        or not required.issubset(header)
    ):
        raise PeriodOutPackError(f"pandas oracle {table} header is incompatible")


def _load_snapshot(source: Sec13FSource) -> _Snapshot:
    pd = _pandas()
    _validate_header(source, "coverpage", COVERPAGE_COLUMNS)
    with source.open_table("coverpage") as handle:
        frame = pd.read_csv(
            handle,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=sorted(COVERPAGE_COLUMNS),
        )
    if frame.empty:
        raise PeriodOutPackError("pandas oracle found an empty COVERPAGE")
    try:
        frame["_date"] = pd.to_datetime(
            frame["REPORTCALENDARORQUARTER"].str.upper(),
            format="%d-%b-%Y",
            errors="raise",
        )
    except (TypeError, ValueError) as exc:
        raise PeriodOutPackError("pandas oracle found an invalid report date") from exc
    latest = frame["_date"].max()
    frame = frame[
        (frame["_date"] == latest)
        & ~frame["REPORTTYPE"].str.upper().str.contains("NOTICE", regex=False)
    ].copy()
    frame["ACCESSION_NUMBER"] = frame["ACCESSION_NUMBER"].str.strip()
    frame["FILINGMANAGER_NAME"] = frame["FILINGMANAGER_NAME"].str.strip()
    frame = frame[
        (frame["ACCESSION_NUMBER"] != "") & (frame["FILINGMANAGER_NAME"] != "")
    ]
    frame["_manager"] = frame["FILINGMANAGER_NAME"].map(normalize_name)
    frame = frame[frame["_manager"] != ""]
    if frame.empty:
        raise PeriodOutPackError("pandas oracle found no eligible manager")

    accession_to_manager: dict[str, str] = {}
    for accession, group in frame.groupby("ACCESSION_NUMBER", sort=True):
        managers = sorted(set(group["_manager"]))
        if len(managers) != 1:
            raise PeriodOutPackError("one accession maps to multiple managers")
        accession_to_manager[str(accession)] = managers[0]
    display_by_manager: dict[str, str] = {}
    unique_accession_by_manager: dict[str, str] = {}
    for manager, group in frame.groupby("_manager", sort=True):
        displays = sorted(
            set(group["FILINGMANAGER_NAME"]),
            key=lambda value: (normalize_name(value), value),
        )
        accessions = sorted(set(group["ACCESSION_NUMBER"]))
        display_by_manager[str(manager)] = str(displays[0])
        if len(accessions) == 1:
            unique_accession_by_manager[str(manager)] = str(accessions[0])
    return _Snapshot(
        accession_to_manager=accession_to_manager,
        display_by_manager=display_by_manager,
        unique_accession_by_manager=unique_accession_by_manager,
    )


def _iter_info_chunks(source: Sec13FSource) -> Iterator[Any]:
    pd = _pandas()
    _validate_header(source, "infotable", INFOTABLE_COLUMNS)
    with source.open_table("infotable") as handle:
        reader = pd.read_csv(
            handle,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=sorted(INFOTABLE_COLUMNS),
            chunksize=CHUNK_ROWS,
        )
        yield from reader


def _add_grouped(
    destination: dict[Any, Decimal], grouped: Any
) -> None:
    for key, value in grouped.items():
        destination[key] = destination.get(key, Decimal()) + value


def _target_accession(snapshot: _Snapshot, display: str) -> str:
    manager = normalize_name(display)
    accession = snapshot.unique_accession_by_manager.get(manager)
    if accession is None:
        raise PeriodOutPackError("pandas oracle manager is not uniquely resolvable")
    return accession


def _scan_current_first_pass(
    source: Sec13FSource,
    *,
    snapshot: _Snapshot,
    aum_accessions: frozenset[str],
    increase_accessions: frozenset[str],
    issuer_targets: frozenset[str],
) -> tuple[
    dict[str, Decimal],
    dict[str, int],
    dict[tuple[str, str], Decimal],
    dict[tuple[str, str], Decimal],
]:
    aum: dict[str, Decimal] = defaultdict(Decimal)
    stock_count: dict[str, int] = defaultdict(int)
    stock_values: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    issuer_values: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    valid_accessions = set(snapshot.accession_to_manager)
    tracked = set(aum_accessions) | set(increase_accessions)
    for chunk in _iter_info_chunks(source):
        chunk["ACCESSION_NUMBER"] = chunk["ACCESSION_NUMBER"].str.strip()
        chunk = chunk[chunk["ACCESSION_NUMBER"].isin(valid_accessions)].copy()
        if chunk.empty:
            continue
        chunk["_value"] = chunk["VALUE"].map(parse_sec_value)
        chunk["_cusip"] = chunk["CUSIP"].map(canonical_cusip)
        tracked_rows = chunk[chunk["ACCESSION_NUMBER"].isin(tracked)].copy()
        if not tracked_rows.empty:
            _add_grouped(
                aum,
                tracked_rows[tracked_rows["ACCESSION_NUMBER"].isin(aum_accessions)]
                .groupby("ACCESSION_NUMBER", sort=True)["_value"]
                .sum(),
            )
            tracked_rows["_stock"] = tracked_rows["TITLEOFCLASS"].map(
                normalize_title_class
            ).isin(STOCK_TITLE_CLASSES_V1)
            count_rows = tracked_rows[
                tracked_rows["_stock"]
                & tracked_rows["ACCESSION_NUMBER"].isin(aum_accessions)
            ]
            for accession, count in count_rows.groupby(
                "ACCESSION_NUMBER", sort=True
            ).size().items():
                stock_count[str(accession)] += int(count)
            holding_rows = tracked_rows[
                tracked_rows["_stock"]
                & tracked_rows["ACCESSION_NUMBER"].isin(increase_accessions)
                & (tracked_rows["_cusip"] != "")
            ]
            _add_grouped(
                stock_values,
                holding_rows.groupby(
                    ["ACCESSION_NUMBER", "_cusip"], sort=True
                )["_value"].sum(),
            )
        chunk["_issuer"] = chunk["NAMEOFISSUER"].map(normalize_name)
        issuer_rows = chunk[
            chunk["_issuer"].isin(issuer_targets) & (chunk["_cusip"] != "")
        ]
        _add_grouped(
            issuer_values,
            issuer_rows.groupby(["_issuer", "_cusip"], sort=True)[
                "_value"
            ].sum(),
        )
    return aum, stock_count, stock_values, issuer_values


def _scan_previous_holdings(
    source: Sec13FSource,
    *,
    snapshot: _Snapshot,
    accessions: frozenset[str],
) -> dict[tuple[str, str], Decimal]:
    result: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    valid_accessions = set(snapshot.accession_to_manager)
    for chunk in _iter_info_chunks(source):
        chunk["ACCESSION_NUMBER"] = chunk["ACCESSION_NUMBER"].str.strip()
        chunk = chunk[
            chunk["ACCESSION_NUMBER"].isin(valid_accessions & set(accessions))
        ].copy()
        if chunk.empty:
            continue
        chunk["_stock"] = chunk["TITLEOFCLASS"].map(
            normalize_title_class
        ).isin(STOCK_TITLE_CLASSES_V1)
        chunk["_cusip"] = chunk["CUSIP"].map(canonical_cusip)
        chunk = chunk[chunk["_stock"] & (chunk["_cusip"] != "")].copy()
        if chunk.empty:
            continue
        chunk["_value"] = chunk["VALUE"].map(parse_sec_value)
        _add_grouped(
            result,
            chunk.groupby(["ACCESSION_NUMBER", "_cusip"], sort=True)[
                "_value"
            ].sum(),
        )
    return result


def _resolve_issuer_cusips(
    issuer_values: Mapping[tuple[str, str], Decimal],
    issuer_targets: frozenset[str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for issuer in issuer_targets:
        candidates = [
            (cusip, value)
            for (observed, cusip), value in issuer_values.items()
            if observed == issuer
        ]
        if not candidates:
            raise PeriodOutPackError("pandas oracle cannot resolve issuer")
        result[issuer] = sorted(candidates, key=lambda row: (-row[1], row[0]))[
            0
        ][0]
    return result


def _scan_current_holders(
    source: Sec13FSource,
    *,
    snapshot: _Snapshot,
    target_cusips: frozenset[str],
) -> dict[tuple[str, str], Decimal]:
    values: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    valid_accessions = set(snapshot.accession_to_manager)
    for chunk in _iter_info_chunks(source):
        chunk["ACCESSION_NUMBER"] = chunk["ACCESSION_NUMBER"].str.strip()
        chunk["_cusip"] = chunk["CUSIP"].map(canonical_cusip)
        chunk = chunk[
            chunk["ACCESSION_NUMBER"].isin(valid_accessions)
            & chunk["_cusip"].isin(target_cusips)
        ].copy()
        if chunk.empty:
            continue
        chunk["_manager"] = chunk["ACCESSION_NUMBER"].map(
            snapshot.accession_to_manager
        )
        chunk["_value"] = chunk["VALUE"].map(parse_sec_value)
        _add_grouped(
            values,
            chunk.groupby(["_cusip", "_manager"], sort=True)["_value"].sum(),
        )
    return values


def evaluate_partition(
    *,
    pack: Mapping[str, Any],
    previous_source: str | Path | Sec13FSource,
    current_source: str | Path | Sec13FSource,
    partition: str,
) -> dict[str, Any]:
    verified = verify_public_pack(pack)
    previous = validate_source_against_pack(
        previous_source, verified, role="previous"
    )
    current = validate_source_against_pack(current_source, verified, role="current")
    items = partition_items(verified, partition)
    previous_snapshot = _load_snapshot(previous)
    current_snapshot = _load_snapshot(current)

    aum_accession_by_item: dict[str, str] = {}
    previous_increase_by_item: dict[str, str] = {}
    current_increase_by_item: dict[str, str] = {}
    issuer_by_item: dict[str, str] = {}
    for item in items:
        query = item["query"]
        item_id = item["item_id"]
        aum_accession_by_item[item_id] = _target_accession(
            current_snapshot, query["aum_manager"]
        )
        previous_increase_by_item[item_id] = _target_accession(
            previous_snapshot, query["increase_manager"]
        )
        current_increase_by_item[item_id] = _target_accession(
            current_snapshot, query["increase_manager"]
        )
        issuer_by_item[item_id] = normalize_name(query["issuer"])

    current_aum, current_counts, current_stock, issuer_values = (
        _scan_current_first_pass(
            current,
            snapshot=current_snapshot,
            aum_accessions=frozenset(aum_accession_by_item.values()),
            increase_accessions=frozenset(current_increase_by_item.values()),
            issuer_targets=frozenset(issuer_by_item.values()),
        )
    )
    previous_stock = _scan_previous_holdings(
        previous,
        snapshot=previous_snapshot,
        accessions=frozenset(previous_increase_by_item.values()),
    )
    issuer_cusips = _resolve_issuer_cusips(
        issuer_values, frozenset(issuer_by_item.values())
    )
    holder_values = _scan_current_holders(
        current,
        snapshot=current_snapshot,
        target_cusips=frozenset(issuer_cusips.values()),
    )

    answers_by_item: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = item["item_id"]
        query = item["query"]
        aum_accession = aum_accession_by_item[item_id]
        previous_accession = previous_increase_by_item[item_id]
        current_accession = current_increase_by_item[item_id]
        previous_values = {
            cusip: value
            for (accession, cusip), value in previous_stock.items()
            if accession == previous_accession
        }
        current_values = {
            cusip: value
            for (accession, cusip), value in current_stock.items()
            if accession == current_accession
        }
        deltas = {
            cusip: current_values.get(cusip, Decimal())
            - previous_values.get(cusip, Decimal())
            for cusip in set(previous_values) | set(current_values)
        }
        increases = [
            cusip
            for cusip, value in sorted(
                deltas.items(), key=lambda row: (-row[1], row[0])
            )
            if value > 0
        ][: query["increase_top_k"]]
        target_cusip = issuer_cusips[issuer_by_item[item_id]]
        manager_rows = [
            (manager, value)
            for (cusip, manager), value in holder_values.items()
            if cusip == target_cusip and value > 0
        ]
        ranked_managers = [
            current_snapshot.display_by_manager[manager]
            for manager, _ in sorted(
                manager_rows, key=lambda row: (-row[1], row[0])
            )[: query["manager_top_k"]]
        ]
        if item["template"] == "four_question_v1":
            answers = {
                "q1_answer": decimal_to_json_number(
                    current_aum.get(aum_accession, Decimal())
                ),
                "q2_answer": int(current_counts.get(aum_accession, 0)),
                "q3_answer": increases,
                "q4_answer": ranked_managers,
            }
        else:
            answers = {
                "q1_answer": decimal_to_json_number(
                    current_aum.get(aum_accession, Decimal())
                ),
                "q2_answer": increases,
                "q3_answer": ranked_managers,
            }
        answers_by_item[item_id] = answers
    return build_oracle_output(
        pack=verified,
        partition=partition,
        oracle_id=ORACLE_ID,
        answers_by_item=answers_by_item,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the chunked-pandas SEC 13F period-out oracle."
    )
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument(
        "--partition", choices=("measurement", "sealed"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = evaluate_partition(
        pack=read_json(args.pack),
        previous_source=args.previous,
        current_source=args.current,
        partition=args.partition,
    )
    write_json(args.output, output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI APIs.
    raise SystemExit(main())
