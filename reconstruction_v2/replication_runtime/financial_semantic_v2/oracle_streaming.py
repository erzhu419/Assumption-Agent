from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime
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


ORACLE_ID = "sec13f_stdlib_streaming_v1"


@dataclass(frozen=True)
class _Snapshot:
    accession_to_manager: Mapping[str, str]
    display_by_manager: Mapping[str, str]
    unique_accession_by_manager: Mapping[str, str]


def _rows(
    source: Sec13FSource,
    table: str,
    required: frozenset[str],
) -> Iterator[dict[str, str]]:
    with source.open_table(table) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        header = reader.fieldnames
        if (
            not isinstance(header, list)
            or len(header) != len(set(header))
            or not required.issubset(header)
        ):
            raise PeriodOutPackError(
                f"streaming oracle {table} header is incompatible"
            )
        for row in reader:
            if None in row:
                raise PeriodOutPackError(
                    f"streaming oracle {table} row has excess fields"
                )
            yield {str(key): str(value or "") for key, value in row.items()}


def _report_date(text: str) -> datetime:
    try:
        return datetime.strptime(text.strip().upper(), "%d-%b-%Y")
    except ValueError as exc:
        raise PeriodOutPackError(
            "streaming oracle found an invalid report date"
        ) from exc


def _load_snapshot(source: Sec13FSource) -> _Snapshot:
    cover: list[tuple[datetime, str, str, str]] = []
    for row in _rows(source, "coverpage", COVERPAGE_COLUMNS):
        accession = row["ACCESSION_NUMBER"].strip()
        display = row["FILINGMANAGER_NAME"].strip()
        if accession and display:
            cover.append(
                (
                    _report_date(row["REPORTCALENDARORQUARTER"]),
                    accession,
                    row["REPORTTYPE"].strip(),
                    display,
                )
            )
    if not cover:
        raise PeriodOutPackError("streaming oracle found an empty COVERPAGE")
    latest = max(row[0] for row in cover)
    accession_to_manager: dict[str, str] = {}
    displays: dict[str, set[str]] = defaultdict(set)
    accessions: dict[str, set[str]] = defaultdict(set)
    for date, accession, report_type, display in cover:
        if date != latest or "NOTICE" in report_type.upper():
            continue
        manager = normalize_name(display)
        if not manager:
            continue
        prior = accession_to_manager.get(accession)
        if prior is not None and prior != manager:
            raise PeriodOutPackError("one accession maps to multiple managers")
        accession_to_manager[accession] = manager
        displays[manager].add(display)
        accessions[manager].add(accession)
    if not accession_to_manager:
        raise PeriodOutPackError("streaming oracle found no eligible manager")
    return _Snapshot(
        accession_to_manager=accession_to_manager,
        display_by_manager={
            manager: sorted(names, key=lambda value: (normalize_name(value), value))[0]
            for manager, names in displays.items()
        },
        unique_accession_by_manager={
            manager: next(iter(values))
            for manager, values in accessions.items()
            if len(values) == 1
        },
    )


def _target_accession(snapshot: _Snapshot, display: str) -> str:
    accession = snapshot.unique_accession_by_manager.get(normalize_name(display))
    if accession is None:
        raise PeriodOutPackError(
            "streaming oracle manager is not uniquely resolvable"
        )
    return accession


def _current_first_pass(
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
    valid_accessions = snapshot.accession_to_manager
    for row in _rows(source, "infotable", INFOTABLE_COLUMNS):
        accession = row["ACCESSION_NUMBER"].strip()
        if accession not in valid_accessions:
            continue
        value = parse_sec_value(row["VALUE"])
        cusip = canonical_cusip(row["CUSIP"])
        if accession in aum_accessions:
            aum[accession] += value
        stock = normalize_title_class(row["TITLEOFCLASS"])
        if stock in STOCK_TITLE_CLASSES_V1:
            if accession in aum_accessions:
                stock_count[accession] += 1
            if accession in increase_accessions and cusip:
                stock_values[(accession, cusip)] += value
        issuer = normalize_name(row["NAMEOFISSUER"])
        if issuer in issuer_targets and cusip:
            issuer_values[(issuer, cusip)] += value
    return aum, stock_count, stock_values, issuer_values


def _previous_holdings(
    source: Sec13FSource,
    *,
    snapshot: _Snapshot,
    accessions: frozenset[str],
) -> dict[tuple[str, str], Decimal]:
    values: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    valid = snapshot.accession_to_manager
    for row in _rows(source, "infotable", INFOTABLE_COLUMNS):
        accession = row["ACCESSION_NUMBER"].strip()
        if accession not in valid or accession not in accessions:
            continue
        if normalize_title_class(row["TITLEOFCLASS"]) not in STOCK_TITLE_CLASSES_V1:
            continue
        cusip = canonical_cusip(row["CUSIP"])
        if cusip:
            values[(accession, cusip)] += parse_sec_value(row["VALUE"])
    return values


def _resolve_issuers(
    values: Mapping[tuple[str, str], Decimal],
    targets: frozenset[str],
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for issuer in targets:
        candidates = [
            (cusip, value)
            for (observed, cusip), value in values.items()
            if observed == issuer
        ]
        if not candidates:
            raise PeriodOutPackError("streaming oracle cannot resolve issuer")
        resolved[issuer] = min(
            candidates,
            key=lambda row: (-row[1], row[0]),
        )[0]
    return resolved


def _current_holders(
    source: Sec13FSource,
    *,
    snapshot: _Snapshot,
    target_cusips: frozenset[str],
) -> dict[tuple[str, str], Decimal]:
    values: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    for row in _rows(source, "infotable", INFOTABLE_COLUMNS):
        accession = row["ACCESSION_NUMBER"].strip()
        manager = snapshot.accession_to_manager.get(accession)
        if manager is None:
            continue
        cusip = canonical_cusip(row["CUSIP"])
        if cusip in target_cusips:
            values[(cusip, manager)] += parse_sec_value(row["VALUE"])
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

    aum_by_item: dict[str, str] = {}
    previous_increase_by_item: dict[str, str] = {}
    current_increase_by_item: dict[str, str] = {}
    issuer_by_item: dict[str, str] = {}
    for item in items:
        item_id = item["item_id"]
        query = item["query"]
        aum_by_item[item_id] = _target_accession(
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
        _current_first_pass(
            current,
            snapshot=current_snapshot,
            aum_accessions=frozenset(aum_by_item.values()),
            increase_accessions=frozenset(current_increase_by_item.values()),
            issuer_targets=frozenset(issuer_by_item.values()),
        )
    )
    previous_stock = _previous_holdings(
        previous,
        snapshot=previous_snapshot,
        accessions=frozenset(previous_increase_by_item.values()),
    )
    issuer_cusips = _resolve_issuers(
        issuer_values, frozenset(issuer_by_item.values())
    )
    holder_values = _current_holders(
        current,
        snapshot=current_snapshot,
        target_cusips=frozenset(issuer_cusips.values()),
    )

    answers_by_item: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = item["item_id"]
        query = item["query"]
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
        manager_values = [
            (manager, value)
            for (cusip, manager), value in holder_values.items()
            if cusip == target_cusip and value > 0
        ]
        managers = [
            current_snapshot.display_by_manager[manager]
            for manager, _ in sorted(
                manager_values, key=lambda row: (-row[1], row[0])
            )[: query["manager_top_k"]]
        ]
        aum = decimal_to_json_number(
            current_aum.get(aum_by_item[item_id], Decimal())
        )
        if item["template"] == "four_question_v1":
            answers = {
                "q1_answer": aum,
                "q2_answer": int(current_counts.get(aum_by_item[item_id], 0)),
                "q3_answer": increases,
                "q4_answer": managers,
            }
        else:
            answers = {
                "q1_answer": aum,
                "q2_answer": increases,
                "q3_answer": managers,
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
        description="Run the stdlib-streaming SEC 13F period-out oracle."
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
