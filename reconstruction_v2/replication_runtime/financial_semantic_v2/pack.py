from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence, TextIO
import unicodedata
import zipfile


PACK_VERSION = "financial_semantic_period_out_pack_v1"
MEASUREMENT_VIEW_VERSION = "financial_semantic_period_out_measurement_view_v1"
SOURCE_POLICY = "official_sec_form_13f_quarterly_flattened_v1"
SELECTION_POLICY = "sha256_ranked_period_out_4fold_x2_plus_4sealed_v1"
ORACLE_OUTPUT_VERSION = "financial_semantic_period_out_oracle_output_v1"
CONSENSUS_GOLD_VERSION = "financial_semantic_period_out_consensus_gold_v1"

MEASUREMENT_FOLD_COUNT = 4
MEASUREMENT_ITEMS_PER_FOLD = 2
MEASUREMENT_ITEM_COUNT = MEASUREMENT_FOLD_COUNT * MEASUREMENT_ITEMS_PER_FOLD
SEALED_ITEM_COUNT = 4
TOTAL_ITEM_COUNT = MEASUREMENT_ITEM_COUNT + SEALED_ITEM_COUNT

REQUIRED_ORACLE_IDS = frozenset(
    {"sec13f_pandas_chunked_v1", "sec13f_stdlib_streaming_v1"}
)

_SOURCE_ROLES = frozenset({"previous", "current"})
_SOURCE_RECEIPT_FIELDS = frozenset(
    {
        "period_label",
        "source_policy",
        "source_kind_at_formation",
        "coverpage_sha256",
        "infotable_sha256",
        "source_fingerprint",
        "source_path_persisted",
    }
)
_ITEM_FIELDS = frozenset(
    {
        "item_id",
        "partition",
        "fold",
        "replicate",
        "template",
        "query",
        "instruction",
        "instruction_sha256",
    }
)
_QUERY_FIELDS = frozenset(
    {
        "aum_manager",
        "include_stock_count",
        "increase_manager",
        "issuer",
        "increase_top_k",
        "manager_top_k",
    }
)

# Limits are part of the deterministic generation policy.  They keep pack
# formation bounded on the multi-million-row SEC tables without selecting on a
# model outcome.  Candidates are SHA-256 ranked before any eligibility test.
MANAGER_ELIGIBILITY_SCAN_LIMIT = 256
ISSUER_ELIGIBILITY_SCAN_LIMIT = 4096

COVERPAGE_COLUMNS = frozenset(
    {
        "ACCESSION_NUMBER",
        "REPORTCALENDARORQUARTER",
        "REPORTTYPE",
        "FILINGMANAGER_NAME",
    }
)
INFOTABLE_COLUMNS = frozenset(
    {
        "ACCESSION_NUMBER",
        "NAMEOFISSUER",
        "TITLEOFCLASS",
        "CUSIP",
        "VALUE",
    }
)

# This is the corrected, explicit stock-title ontology for the extension.  It
# is a public task semantic, shared by the two independent oracle
# implementations; neither oracle imports a candidate or a consumed solution.
STOCK_TITLE_CLASSES_V1 = frozenset(
    {
        "ADR",
        "CAP STK CL A",
        "CAP STK CL C",
        "CL A",
        "CL A COM",
        "CL A NEW",
        "CL B",
        "CL B NEW",
        "CMN",
        "COM",
        "COM CL A",
        "COM NEW",
        "COM SHS",
        "COMM STK",
        "COMMON",
        "COMMON STOCK",
        "EQUITY",
        "FOREIGN STOCK",
        "ORD SHS",
        "SHS CL A",
        "SPONSORED ADR",
        "SPONSORED ADS",
        "STOCK",
        "CLASS A",
        "CLASS A COM",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PARTITIONS = frozenset({"measurement", "sealed"})


class PeriodOutPackError(ValueError):
    """A period-out source or frozen pack violates its public contract."""


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
    source = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_sha256(handle: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def normalize_title_class(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).upper()
    return " ".join(text.split())


def canonical_cusip(value: object) -> str:
    return "".join(str(value or "").upper().split())


def parse_sec_value(value: object) -> Decimal:
    text = str(value or "").strip().replace(",", "")
    if not text:
        raise PeriodOutPackError("SEC VALUE is empty")
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise PeriodOutPackError("SEC VALUE is not decimal") from exc
    if not parsed.is_finite() or parsed < 0:
        raise PeriodOutPackError("SEC VALUE is negative or non-finite")
    return parsed


def decimal_to_json_number(value: Decimal) -> int | float:
    if not value.is_finite():
        raise PeriodOutPackError("oracle result is non-finite")
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def _parse_sec_date(value: str) -> datetime:
    try:
        return datetime.strptime(value.strip().upper(), "%d-%b-%Y")
    except ValueError as exc:
        raise PeriodOutPackError(
            "REPORTCALENDARORQUARTER is not DD-MON-YYYY"
        ) from exc


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PeriodOutPackError(f"{label} is not a SHA-256 digest")
    return value


def _with_self_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    if field in result:
        raise PeriodOutPackError(f"{field} already exists")
    result[field] = payload_hash(result)
    return result


def _verify_self_hash(
    payload: Mapping[str, Any], *, field: str, label: str
) -> str:
    declared = _require_sha256(payload.get(field), f"{label} {field}")
    body = dict(payload)
    del body[field]
    if payload_hash(body) != declared:
        raise PeriodOutPackError(f"{label} self hash mismatch")
    return declared


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str] | set[str],
    label: str,
) -> None:
    if set(value) != set(expected):
        raise PeriodOutPackError(f"{label} fields drifted")


def _validate_source_metadata(
    *,
    sources: object,
    snapshot_dates: object,
    roots: object,
    label: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    if (
        not isinstance(sources, Mapping)
        or not isinstance(snapshot_dates, Mapping)
        or not isinstance(roots, Mapping)
    ):
        raise PeriodOutPackError(f"{label} sources are malformed")
    _require_exact_fields(sources, _SOURCE_ROLES, f"{label} sources")
    _require_exact_fields(
        snapshot_dates,
        _SOURCE_ROLES,
        f"{label} snapshot dates",
    )
    _require_exact_fields(roots, _SOURCE_ROLES, f"{label} container roots")
    labels: list[str] = []
    for role in ("previous", "current"):
        receipt = sources.get(role)
        if not isinstance(receipt, Mapping):
            raise PeriodOutPackError(f"{label} source receipt is malformed")
        _require_exact_fields(
            receipt,
            _SOURCE_RECEIPT_FIELDS,
            f"{label} {role} source receipt",
        )
        period_label = receipt.get("period_label")
        if not isinstance(period_label, str) or not period_label.strip():
            raise PeriodOutPackError(f"{label} period label is empty")
        labels.append(period_label.strip())
        if (
            receipt.get("source_policy") != SOURCE_POLICY
            or receipt.get("source_kind_at_formation")
            not in {"zip", "directory"}
            or receipt.get("source_path_persisted") is not False
        ):
            raise PeriodOutPackError(f"{label} source policy drifted")
        for field in (
            "coverpage_sha256",
            "infotable_sha256",
            "source_fingerprint",
        ):
            _require_sha256(receipt.get(field), f"{role} {field}")
        expected_fingerprint = payload_hash(
            {
                "source_policy": SOURCE_POLICY,
                "coverpage_sha256": receipt["coverpage_sha256"],
                "infotable_sha256": receipt["infotable_sha256"],
            }
        )
        if receipt.get("source_fingerprint") != expected_fingerprint:
            raise PeriodOutPackError(f"{label} source fingerprint drifted")
        root = roots.get(role)
        if not isinstance(root, str):
            raise PeriodOutPackError(f"{label} container root is invalid")
        parsed_root = PurePosixPath(root)
        if not parsed_root.is_absolute() or ".." in parsed_root.parts:
            raise PeriodOutPackError(f"{label} container root is unsafe")
    if labels[0] == labels[1] or roots["previous"] == roots["current"]:
        raise PeriodOutPackError(f"{label} period roles are not distinct")
    try:
        previous_date = datetime.strptime(
            str(snapshot_dates["previous"]), "%Y-%m-%d"
        )
        current_date = datetime.strptime(
            str(snapshot_dates["current"]), "%Y-%m-%d"
        )
    except (TypeError, ValueError) as exc:
        raise PeriodOutPackError(f"{label} snapshot date is invalid") from exc
    if previous_date >= current_date:
        raise PeriodOutPackError(f"{label} period order drifted")
    return sources, snapshot_dates, roots


def _validate_zip_members(archive: zipfile.ZipFile) -> None:
    seen: set[str] = set()
    for info in archive.infolist():
        name = info.filename
        path = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or path.is_absolute()
            or ".." in path.parts
            or name in seen
        ):
            raise PeriodOutPackError("SEC ZIP contains an unsafe member")
        seen.add(name)
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        if unix_mode and stat.S_ISLNK(unix_mode):
            raise PeriodOutPackError("SEC ZIP contains a symbolic link")


@dataclass(frozen=True)
class Sec13FSource:
    """A read-only official SEC 13F ZIP or extracted directory.

    The semantic fingerprint binds the two tables used by this extension and
    is identical for a ZIP and its byte-identical extracted form.
    """

    path: Path
    source_kind: str
    coverpage_ref: str
    infotable_ref: str
    coverpage_sha256: str
    infotable_sha256: str
    source_fingerprint: str

    @classmethod
    def open(cls, value: str | Path) -> "Sec13FSource":
        path = Path(value).expanduser().resolve(strict=True)
        if path.is_dir():
            cover = _find_directory_table(path, "COVERPAGE.tsv")
            info = _find_directory_table(path, "INFOTABLE.tsv")
            cover_hash = sha256_file(cover)
            info_hash = sha256_file(info)
            kind = "directory"
            cover_ref = cover.relative_to(path).as_posix()
            info_ref = info.relative_to(path).as_posix()
        elif path.is_file() and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                _validate_zip_members(archive)
                cover_ref = _find_zip_table(archive, "COVERPAGE.tsv")
                info_ref = _find_zip_table(archive, "INFOTABLE.tsv")
                with archive.open(cover_ref) as handle:
                    cover_hash = _stream_sha256(handle)
                with archive.open(info_ref) as handle:
                    info_hash = _stream_sha256(handle)
            kind = "zip"
        else:
            raise PeriodOutPackError("SEC 13F source must be a ZIP or directory")
        fingerprint = payload_hash(
            {
                "source_policy": SOURCE_POLICY,
                "coverpage_sha256": cover_hash,
                "infotable_sha256": info_hash,
            }
        )
        return cls(
            path=path,
            source_kind=kind,
            coverpage_ref=cover_ref,
            infotable_ref=info_ref,
            coverpage_sha256=cover_hash,
            infotable_sha256=info_hash,
            source_fingerprint=fingerprint,
        )

    def receipt(self, *, period_label: str) -> dict[str, Any]:
        label = str(period_label).strip()
        if not label:
            raise PeriodOutPackError("period label is empty")
        return {
            "period_label": label,
            "source_policy": SOURCE_POLICY,
            "source_kind_at_formation": self.source_kind,
            "coverpage_sha256": self.coverpage_sha256,
            "infotable_sha256": self.infotable_sha256,
            "source_fingerprint": self.source_fingerprint,
            "source_path_persisted": False,
        }

    @contextmanager
    def open_table(self, table: str) -> Iterator[TextIO]:
        if table == "coverpage":
            reference = self.coverpage_ref
        elif table == "infotable":
            reference = self.infotable_ref
        else:
            raise PeriodOutPackError("unknown SEC 13F table")
        if self.source_kind == "directory":
            path = (self.path / reference).resolve(strict=True)
            try:
                path.relative_to(self.path)
            except ValueError as exc:
                raise PeriodOutPackError("table escaped extracted source") from exc
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                yield handle
            return
        archive = zipfile.ZipFile(self.path)
        raw = archive.open(reference)
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        try:
            yield text
        finally:
            text.close()
            archive.close()


def _find_directory_table(root: Path, basename: str) -> Path:
    matches = [
        path
        for path in root.rglob("*")
        if path.name.casefold() == basename.casefold() and path.is_file()
    ]
    if len(matches) != 1:
        raise PeriodOutPackError(
            f"expected exactly one {basename} in extracted SEC source"
        )
    if matches[0].is_symlink():
        raise PeriodOutPackError("SEC table may not be a symlink")
    return matches[0].resolve(strict=True)


def _find_zip_table(archive: zipfile.ZipFile, basename: str) -> str:
    matches = [
        info.filename
        for info in archive.infolist()
        if not info.is_dir()
        and Path(info.filename).name.casefold() == basename.casefold()
    ]
    if len(matches) != 1:
        raise PeriodOutPackError(
            f"expected exactly one {basename} in SEC ZIP"
        )
    return matches[0]


def _iter_tsv(
    source: Sec13FSource,
    table: str,
    required_columns: frozenset[str],
) -> Iterator[dict[str, str]]:
    with source.open_table(table) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        header = reader.fieldnames
        if (
            not isinstance(header, list)
            or len(header) != len(set(header))
            or not required_columns.issubset(header)
        ):
            raise PeriodOutPackError(f"{table} header is incompatible")
        for row in reader:
            if None in row:
                raise PeriodOutPackError(f"{table} row has excess fields")
            yield {str(key): str(value or "") for key, value in row.items()}


@dataclass(frozen=True)
class _Snapshot:
    report_date: str
    accession_to_manager: Mapping[str, str]
    display_by_manager: Mapping[str, str]
    accessions_by_manager: Mapping[str, tuple[str, ...]]

    @property
    def unique_accession_by_manager(self) -> dict[str, str]:
        return {
            manager: accessions[0]
            for manager, accessions in self.accessions_by_manager.items()
            if len(accessions) == 1
        }


def _load_snapshot(source: Sec13FSource) -> _Snapshot:
    rows: list[tuple[datetime, str, str, str]] = []
    for row in _iter_tsv(source, "coverpage", COVERPAGE_COLUMNS):
        accession = row["ACCESSION_NUMBER"].strip()
        manager = row["FILINGMANAGER_NAME"].strip()
        if not accession or not manager:
            continue
        rows.append(
            (
                _parse_sec_date(row["REPORTCALENDARORQUARTER"]),
                accession,
                row["REPORTTYPE"].strip(),
                manager,
            )
        )
    if not rows:
        raise PeriodOutPackError("COVERPAGE contains no usable filings")
    latest = max(row[0] for row in rows)
    eligible = [
        row for row in rows if row[0] == latest and "NOTICE" not in row[2].upper()
    ]
    if not eligible:
        raise PeriodOutPackError("latest SEC snapshot contains only notices")
    accession_to_manager: dict[str, str] = {}
    display_candidates: dict[str, set[str]] = defaultdict(set)
    accession_candidates: dict[str, set[str]] = defaultdict(set)
    for _, accession, _, display in eligible:
        manager = normalize_name(display)
        if not manager:
            continue
        prior = accession_to_manager.get(accession)
        if prior is not None and prior != manager:
            raise PeriodOutPackError("one accession maps to multiple managers")
        accession_to_manager[accession] = manager
        display_candidates[manager].add(display)
        accession_candidates[manager].add(accession)
    if not accession_to_manager:
        raise PeriodOutPackError("latest SEC snapshot has no manager inventory")
    return _Snapshot(
        report_date=latest.strftime("%Y-%m-%d"),
        accession_to_manager=accession_to_manager,
        display_by_manager={
            manager: sorted(names, key=lambda value: (normalize_name(value), value))[0]
            for manager, names in display_candidates.items()
        },
        accessions_by_manager={
            manager: tuple(sorted(accessions))
            for manager, accessions in accession_candidates.items()
        },
    )


def _rank_key(seed: str, role: str, key: str) -> tuple[str, str]:
    return (
        hashlib.sha256(
            canonical_json_bytes({"seed": seed, "role": role, "key": key})
        ).hexdigest(),
        key,
    )


def _ranked(seed: str, role: str, values: Iterable[str]) -> list[str]:
    return sorted(set(values), key=lambda value: _rank_key(seed, role, value))


def derive_selection_seed(
    *,
    preregistration_seed: str,
    previous_source_fingerprint: str,
    current_source_fingerprint: str,
) -> str:
    seed = str(preregistration_seed).strip()
    if not seed:
        raise PeriodOutPackError("preregistration seed is empty")
    return payload_hash(
        {
            "selection_policy": SELECTION_POLICY,
            "preregistration_seed": seed,
            "previous_source_fingerprint": _require_sha256(
                previous_source_fingerprint,
                "previous source fingerprint",
            ),
            "current_source_fingerprint": _require_sha256(
                current_source_fingerprint,
                "current source fingerprint",
            ),
        }
    )


@dataclass
class _ManagerStats:
    aum: dict[str, Decimal]
    stock_count: dict[str, int]
    stock_values: dict[str, dict[str, Decimal]]


def _new_manager_stats() -> _ManagerStats:
    return _ManagerStats(
        aum=defaultdict(Decimal),
        stock_count=defaultdict(int),
        stock_values=defaultdict(lambda: defaultdict(Decimal)),
    )


def _scan_manager_stats_and_issuers(
    source: Sec13FSource,
    *,
    snapshot: _Snapshot,
    tracked_accessions: frozenset[str],
    collect_issuers: bool,
) -> tuple[
    _ManagerStats,
    dict[tuple[str, str], Decimal],
    dict[tuple[str, str], set[str]],
]:
    stats = _new_manager_stats()
    issuer_totals: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    issuer_displays: dict[tuple[str, str], set[str]] = defaultdict(set)
    snapshot_accessions = snapshot.accession_to_manager
    for row in _iter_tsv(source, "infotable", INFOTABLE_COLUMNS):
        accession = row["ACCESSION_NUMBER"].strip()
        if accession not in snapshot_accessions:
            continue
        value = parse_sec_value(row["VALUE"])
        cusip = canonical_cusip(row["CUSIP"])
        if accession in tracked_accessions:
            stats.aum[accession] += value
            if normalize_title_class(row["TITLEOFCLASS"]) in STOCK_TITLE_CLASSES_V1:
                stats.stock_count[accession] += 1
                if cusip:
                    stats.stock_values[accession][cusip] += value
        if collect_issuers and cusip:
            display = row["NAMEOFISSUER"].strip()
            issuer = normalize_name(display)
            if issuer:
                key = (issuer, cusip)
                issuer_totals[key] += value
                issuer_displays[key].add(display)
    return stats, issuer_totals, issuer_displays


def _canonical_issuer_candidates(
    totals: Mapping[tuple[str, str], Decimal],
    displays: Mapping[tuple[str, str], set[str]],
) -> dict[str, tuple[str, str]]:
    by_issuer: dict[str, list[tuple[str, Decimal]]] = defaultdict(list)
    for (issuer, cusip), value in totals.items():
        by_issuer[issuer].append((cusip, value))
    result: dict[str, tuple[str, str]] = {}
    for issuer, rows in by_issuer.items():
        cusip, _ = sorted(rows, key=lambda row: (-row[1], row[0]))[0]
        names = displays.get((issuer, cusip), set())
        if not names:
            continue
        result[issuer] = (
            sorted(names, key=lambda value: (normalize_name(value), value))[0],
            cusip,
        )
    return result


def _scan_issuer_holders(
    source: Sec13FSource,
    *,
    snapshot: _Snapshot,
    target_cusips: frozenset[str],
) -> dict[str, set[str]]:
    holders: dict[str, set[str]] = defaultdict(set)
    if not target_cusips:
        return holders
    for row in _iter_tsv(source, "infotable", INFOTABLE_COLUMNS):
        accession = row["ACCESSION_NUMBER"].strip()
        manager = snapshot.accession_to_manager.get(accession)
        if manager is None:
            continue
        cusip = canonical_cusip(row["CUSIP"])
        if cusip in target_cusips and parse_sec_value(row["VALUE"]) > 0:
            holders[cusip].add(manager)
    return holders


def _positive_delta_count(
    previous: Mapping[str, Decimal], current: Mapping[str, Decimal]
) -> int:
    return sum(
        1
        for cusip in set(previous) | set(current)
        if current.get(cusip, Decimal()) - previous.get(cusip, Decimal()) > 0
    )


def _deterministic_distinct_assignment(
    anchors: Sequence[str], candidates: Sequence[str]
) -> list[str]:
    """Choose a globally disjoint second role in its frozen hash order."""

    blocked = set(anchors)
    available = [candidate for candidate in candidates if candidate not in blocked]
    if len(available) < len(anchors):
        raise PeriodOutPackError("cannot form globally disjoint manager roles")
    return available[: len(anchors)]


def _selection_inventory(
    previous: Sec13FSource,
    current: Sec13FSource,
    *,
    seed: str,
) -> tuple[
    _Snapshot,
    _Snapshot,
    list[str],
    list[str],
    list[str],
    Mapping[str, tuple[str, str]],
    dict[str, int],
]:
    previous_snapshot = _load_snapshot(previous)
    current_snapshot = _load_snapshot(current)
    previous_unique = previous_snapshot.unique_accession_by_manager
    current_unique = current_snapshot.unique_accession_by_manager

    aum_preliminary = _ranked(
        seed, "aum-manager", current_unique
    )[:MANAGER_ELIGIBILITY_SCAN_LIMIT]
    increase_preliminary = _ranked(
        seed,
        "increase-manager",
        set(previous_unique) & set(current_unique),
    )[:MANAGER_ELIGIBILITY_SCAN_LIMIT]
    current_tracked = frozenset(
        current_unique[manager]
        for manager in set(aum_preliminary) | set(increase_preliminary)
    )
    previous_tracked = frozenset(
        previous_unique[manager] for manager in increase_preliminary
    )

    current_stats, issuer_totals, issuer_displays = (
        _scan_manager_stats_and_issuers(
            current,
            snapshot=current_snapshot,
            tracked_accessions=current_tracked,
            collect_issuers=True,
        )
    )
    previous_stats, _, _ = _scan_manager_stats_and_issuers(
        previous,
        snapshot=previous_snapshot,
        tracked_accessions=previous_tracked,
        collect_issuers=False,
    )

    aum_eligible = [
        manager
        for manager in aum_preliminary
        if current_stats.aum.get(current_unique[manager], Decimal()) > 0
        and current_stats.stock_count.get(current_unique[manager], 0) > 0
    ]
    increase_eligible = [
        manager
        for manager in increase_preliminary
        if _positive_delta_count(
            previous_stats.stock_values.get(previous_unique[manager], {}),
            current_stats.stock_values.get(current_unique[manager], {}),
        )
        >= 5
    ]
    if len(aum_eligible) < TOTAL_ITEM_COUNT:
        raise PeriodOutPackError("fewer than 12 eligible current managers")
    if len(increase_eligible) < TOTAL_ITEM_COUNT:
        raise PeriodOutPackError("fewer than 12 eligible comparison managers")
    selected_aum = aum_eligible[:TOTAL_ITEM_COUNT]
    selected_increase = _deterministic_distinct_assignment(
        selected_aum, increase_eligible
    )

    issuer_candidates = _canonical_issuer_candidates(
        issuer_totals, issuer_displays
    )
    issuer_preliminary = _ranked(
        seed, "issuer", issuer_candidates
    )[:ISSUER_ELIGIBILITY_SCAN_LIMIT]
    target_cusips = frozenset(
        issuer_candidates[issuer][1] for issuer in issuer_preliminary
    )
    holder_sets = _scan_issuer_holders(
        current,
        snapshot=current_snapshot,
        target_cusips=target_cusips,
    )
    issuer_eligible = [
        issuer
        for issuer in issuer_preliminary
        if len(holder_sets.get(issuer_candidates[issuer][1], set())) >= 5
        and issuer not in set(selected_aum)
        and issuer not in set(selected_increase)
    ]
    if len(issuer_eligible) < TOTAL_ITEM_COUNT:
        raise PeriodOutPackError("fewer than 12 eligible current issuers")
    selected_issuers = issuer_eligible[:TOTAL_ITEM_COUNT]
    counts = {
        "aum_manager_preliminary": len(aum_preliminary),
        "aum_manager_eligible": len(aum_eligible),
        "comparison_manager_preliminary": len(increase_preliminary),
        "comparison_manager_eligible": len(increase_eligible),
        "canonical_issuer_count": len(issuer_candidates),
        "issuer_preliminary": len(issuer_preliminary),
        "issuer_eligible": len(issuer_eligible),
    }
    return (
        previous_snapshot,
        current_snapshot,
        selected_aum,
        selected_increase,
        selected_issuers,
        issuer_candidates,
        counts,
    )


def _semantic_contract_text() -> str:
    stock_classes = ", ".join(sorted(STOCK_TITLE_CLASSES_V1))
    return (
        "Use only non-NOTICE filings at the latest REPORTCALENDARORQUARTER "
        "within each period. Manager and issuer matching is Unicode NFKC, "
        "case-insensitive, and punctuation-insensitive. Every selected fund "
        "has exactly one eligible accession per required period. AUM is the "
        "sum of VALUE over all rows for the selected current accession. "
        "Stock-only operations accept exactly these normalized TITLEOFCLASS "
        f"values: {stock_classes}. For an issuer name, choose the matching "
        "CUSIP with greatest aggregate current VALUE (CUSIP ascending on a "
        "tie). Aggregate rows by CUSIP or normalized manager before ranking; "
        "rank VALUE descending and use canonical CUSIP or normalized manager "
        "ascending to break ties."
    )


def render_instruction(
    *,
    template: str,
    previous_period_label: str,
    current_period_label: str,
    previous_container_root: str,
    current_container_root: str,
    aum_manager: str,
    increase_manager: str,
    issuer: str,
    increase_top_k: int,
    manager_top_k: int,
) -> str:
    if template not in {"four_question_v1", "three_question_v1"}:
        raise PeriodOutPackError("unknown period-out template")
    if increase_top_k not in {3, 5} or manager_top_k not in {3, 5}:
        raise PeriodOutPackError("period-out top-k is outside the frozen set")
    contract = _semantic_contract_text()
    header = (
        "You are a financial analyst comparing official SEC Form 13F data "
        f"for {current_period_label} against {previous_period_label}. The "
        f"previous data is in `{previous_container_root}` and current data is "
        f"in `{current_container_root}`.\n\nFrozen data semantics: {contract}\n\n"
        "Questions:\n\n"
    )
    if template == "four_question_v1":
        questions = (
            f"1. What is the current-period AUM of {aum_manager}?\n\n"
            f"2. How many stock rows are held by {aum_manager} in the current period?\n\n"
            f"3. What are the top {increase_top_k} CUSIPs with increased "
            f"investment by {increase_manager} from the previous period to "
            "the current period, ranked by dollar-value increase?\n\n"
            f"4. Which top {manager_top_k} fund managers hold {issuer} in the "
            "current period, ranked by aggregate position value?\n\n"
            "Write `/root/answers.json` with keys `q1_answer`, `q2_answer`, "
            "`q3_answer`, and `q4_answer` in that order. q1 and q2 are numbers; "
            "q3 and q4 are ordered JSON arrays.\n"
        )
    else:
        questions = (
            f"1. What is the current-period AUM of {aum_manager}?\n\n"
            f"2. What are the top {increase_top_k} CUSIPs with increased "
            f"investment by {increase_manager} from the previous period to "
            "the current period, ranked by dollar-value increase?\n\n"
            f"3. Which top {manager_top_k} fund managers hold {issuer} in the "
            "current period, ranked by aggregate position value?\n\n"
            "Write `/root/answers.json` with keys `q1_answer`, `q2_answer`, "
            "and `q3_answer` in that order. q1 is a number; q2 and q3 are "
            "ordered JSON arrays.\n"
        )
    return header + questions


def _item_layout(index: int) -> tuple[str, str, int | None, int, str, int, int]:
    if not 0 <= index < TOTAL_ITEM_COUNT:
        raise PeriodOutPackError("item index is outside frozen layout")
    if index < MEASUREMENT_ITEM_COUNT:
        fold = index // MEASUREMENT_ITEMS_PER_FOLD
        replicate = index % MEASUREMENT_ITEMS_PER_FOLD
        partition = "measurement"
        item_id = f"financial-period-out-measurement-f{fold}-r{replicate}"
    else:
        fold = None
        replicate = index - MEASUREMENT_ITEM_COUNT
        partition = "sealed"
        item_id = f"financial-period-out-sealed-{replicate}"
    if replicate % 2 == 0:
        return item_id, partition, fold, replicate, "four_question_v1", 5, 3
    return item_id, partition, fold, replicate, "three_question_v1", 3, 5


def build_public_pack(
    *,
    previous_source: str | Path | Sec13FSource,
    current_source: str | Path | Sec13FSource,
    previous_period_label: str,
    current_period_label: str,
    preregistration_seed: str,
    previous_container_root: str = "/root/period-previous",
    current_container_root: str = "/root/period-current",
) -> dict[str, Any]:
    previous = (
        previous_source
        if isinstance(previous_source, Sec13FSource)
        else Sec13FSource.open(previous_source)
    )
    current = (
        current_source
        if isinstance(current_source, Sec13FSource)
        else Sec13FSource.open(current_source)
    )
    if previous.source_fingerprint == current.source_fingerprint:
        raise PeriodOutPackError("previous and current SEC sources are identical")
    selection_seed = derive_selection_seed(
        preregistration_seed=preregistration_seed,
        previous_source_fingerprint=previous.source_fingerprint,
        current_source_fingerprint=current.source_fingerprint,
    )
    (
        previous_snapshot,
        current_snapshot,
        aum_managers,
        increase_managers,
        issuers,
        issuer_candidates,
        pool_counts,
    ) = _selection_inventory(previous, current, seed=selection_seed)

    items: list[dict[str, Any]] = []
    for index in range(TOTAL_ITEM_COUNT):
        (
            item_id,
            partition,
            fold,
            replicate,
            template,
            increase_top_k,
            manager_top_k,
        ) = _item_layout(index)
        aum_manager = current_snapshot.display_by_manager[aum_managers[index]]
        increase_manager = current_snapshot.display_by_manager[
            increase_managers[index]
        ]
        issuer = issuer_candidates[issuers[index]][0]
        query = {
            "aum_manager": aum_manager,
            "include_stock_count": template == "four_question_v1",
            "increase_manager": increase_manager,
            "issuer": issuer,
            "increase_top_k": increase_top_k,
            "manager_top_k": manager_top_k,
        }
        instruction = render_instruction(
            template=template,
            previous_period_label=previous_period_label,
            current_period_label=current_period_label,
            previous_container_root=previous_container_root,
            current_container_root=current_container_root,
            aum_manager=aum_manager,
            increase_manager=increase_manager,
            issuer=issuer,
            increase_top_k=increase_top_k,
            manager_top_k=manager_top_k,
        )
        items.append(
            {
                "item_id": item_id,
                "partition": partition,
                "fold": fold,
                "replicate": replicate,
                "template": template,
                "query": query,
                "instruction": instruction,
                "instruction_sha256": hashlib.sha256(
                    instruction.encode("utf-8")
                ).hexdigest(),
            }
        )

    body: dict[str, Any] = {
        "pack_version": PACK_VERSION,
        "source_policy": SOURCE_POLICY,
        "selection_policy": SELECTION_POLICY,
        "selection_seed": selection_seed,
        "sources": {
            "previous": previous.receipt(period_label=previous_period_label),
            "current": current.receipt(period_label=current_period_label),
        },
        "snapshot_report_dates": {
            "previous": previous_snapshot.report_date,
            "current": current_snapshot.report_date,
        },
        "container_roots": {
            "previous": str(previous_container_root),
            "current": str(current_container_root),
        },
        "selection_pool_counts": pool_counts,
        "stock_title_class_semantics": sorted(STOCK_TITLE_CLASSES_V1),
        "items": items,
        "partition_contract": {
            "measurement_item_count": MEASUREMENT_ITEM_COUNT,
            "measurement_fold_count": MEASUREMENT_FOLD_COUNT,
            "measurement_items_per_fold": MEASUREMENT_ITEMS_PER_FOLD,
            "sealed_item_count": SEALED_ITEM_COUNT,
            "sealed_gold_must_be_stored_separately": True,
        },
        "ground_truth_persisted": False,
        "candidate_imports": 0,
        "model_calls": 0,
        "network_calls": 0,
    }
    return verify_public_pack(_with_self_hash(body, "pack_hash"))


def verify_public_pack(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PeriodOutPackError("public pack must be an object")
    payload = dict(value)
    _verify_self_hash(payload, field="pack_hash", label="public pack")
    _require_exact_fields(
        payload,
        {
            "pack_version",
            "source_policy",
            "selection_policy",
            "selection_seed",
            "sources",
            "snapshot_report_dates",
            "container_roots",
            "selection_pool_counts",
            "stock_title_class_semantics",
            "items",
            "partition_contract",
            "ground_truth_persisted",
            "candidate_imports",
            "model_calls",
            "network_calls",
            "pack_hash",
        },
        "public pack",
    )
    if (
        payload.get("pack_version") != PACK_VERSION
        or payload.get("source_policy") != SOURCE_POLICY
        or payload.get("selection_policy") != SELECTION_POLICY
        or payload.get("ground_truth_persisted") is not False
        or payload.get("candidate_imports") != 0
        or payload.get("model_calls") != 0
        or payload.get("network_calls") != 0
    ):
        raise PeriodOutPackError("public pack policy drifted")
    _require_sha256(payload.get("selection_seed"), "selection seed")
    sources, _, roots = _validate_source_metadata(
        sources=payload.get("sources"),
        snapshot_dates=payload.get("snapshot_report_dates"),
        roots=payload.get("container_roots"),
        label="public pack",
    )
    if sources["previous"]["source_fingerprint"] == sources["current"][
        "source_fingerprint"
    ]:
        raise PeriodOutPackError("public pack source periods are identical")
    if payload.get("stock_title_class_semantics") != sorted(
        STOCK_TITLE_CLASSES_V1
    ):
        raise PeriodOutPackError("stock-title semantics drifted")
    pool_counts = payload.get("selection_pool_counts")
    pool_fields = {
        "aum_manager_preliminary",
        "aum_manager_eligible",
        "comparison_manager_preliminary",
        "comparison_manager_eligible",
        "canonical_issuer_count",
        "issuer_preliminary",
        "issuer_eligible",
    }
    if (
        not isinstance(pool_counts, Mapping)
        or set(pool_counts) != pool_fields
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in pool_counts.values()
        )
    ):
        raise PeriodOutPackError("selection pool counts drifted")
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != TOTAL_ITEM_COUNT:
        raise PeriodOutPackError("public pack item count drifted")
    seen: set[str] = set()
    folds: dict[int, int] = defaultdict(int)
    partition_counts: dict[str, int] = defaultdict(int)
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise PeriodOutPackError("public item is malformed")
        _require_exact_fields(item, _ITEM_FIELDS, "public item")
        expected = _item_layout(index)
        item_id, partition, fold, replicate, template, inc_k, manager_k = expected
        if (
            item.get("item_id") != item_id
            or item.get("partition") != partition
            or item.get("fold") != fold
            or item.get("replicate") != replicate
            or item.get("template") != template
            or item_id in seen
        ):
            raise PeriodOutPackError("public item layout drifted")
        seen.add(item_id)
        partition_counts[partition] += 1
        if fold is not None:
            folds[fold] += 1
        query = item.get("query")
        if not isinstance(query, Mapping):
            raise PeriodOutPackError("public query is malformed")
        _require_exact_fields(query, _QUERY_FIELDS, "public query")
        if (
            query.get("increase_top_k") != inc_k
            or query.get("manager_top_k") != manager_k
            or query.get("include_stock_count")
            is not (template == "four_question_v1")
        ):
            raise PeriodOutPackError("public query scalar drifted")
        for field in ("aum_manager", "increase_manager", "issuer"):
            if not isinstance(query.get(field), str) or not query[field].strip():
                raise PeriodOutPackError("public query entity is invalid")
        expected_instruction = render_instruction(
            template=template,
            previous_period_label=str(sources["previous"]["period_label"]),
            current_period_label=str(sources["current"]["period_label"]),
            previous_container_root=str(roots["previous"]),
            current_container_root=str(roots["current"]),
            aum_manager=str(query["aum_manager"]),
            increase_manager=str(query["increase_manager"]),
            issuer=str(query["issuer"]),
            increase_top_k=inc_k,
            manager_top_k=manager_k,
        )
        if item.get("instruction") != expected_instruction:
            raise PeriodOutPackError("public instruction drifted")
        if hashlib.sha256(expected_instruction.encode("utf-8")).hexdigest() != (
            item.get("instruction_sha256")
        ):
            raise PeriodOutPackError("public instruction hash drifted")
    if partition_counts != {
        "measurement": MEASUREMENT_ITEM_COUNT,
        "sealed": SEALED_ITEM_COUNT,
    } or folds != {
        fold: MEASUREMENT_ITEMS_PER_FOLD
        for fold in range(MEASUREMENT_FOLD_COUNT)
    }:
        raise PeriodOutPackError("public partition contract drifted")
    contract = payload.get("partition_contract")
    expected_contract = {
        "measurement_item_count": MEASUREMENT_ITEM_COUNT,
        "measurement_fold_count": MEASUREMENT_FOLD_COUNT,
        "measurement_items_per_fold": MEASUREMENT_ITEMS_PER_FOLD,
        "sealed_item_count": SEALED_ITEM_COUNT,
        "sealed_gold_must_be_stored_separately": True,
    }
    if not isinstance(contract, Mapping) or dict(contract) != expected_contract:
        raise PeriodOutPackError("gold separation contract is absent")
    return payload


def build_measurement_view(pack: Mapping[str, Any]) -> dict[str, Any]:
    """Redact every sealed query/entity/instruction behind commitments.

    The returned view is the only pack artifact intended for development
    workers.  The full pack remains private beside the separately stored sealed
    consensus gold.
    """

    private = verify_public_pack(pack)
    measurement_items = [
        dict(item)
        for item in private["items"]
        if item["partition"] == "measurement"
    ]
    sealed_commitments: list[dict[str, Any]] = []
    for item in private["items"]:
        if item["partition"] != "sealed":
            continue
        commitment = {
            "item_id": item["item_id"],
            "template": item["template"],
            "fold": item["fold"],
            "instruction_sha256": item["instruction_sha256"],
            "query_commitment_hash": payload_hash(item["query"]),
            "full_item_commitment_hash": payload_hash(item),
        }
        sealed_commitments.append(commitment)
    body = {
        "measurement_view_version": MEASUREMENT_VIEW_VERSION,
        "source_policy": SOURCE_POLICY,
        "selection_policy": SELECTION_POLICY,
        "private_pack_hash": private["pack_hash"],
        "selection_seed_commitment_hash": payload_hash(
            {"selection_seed": private["selection_seed"]}
        ),
        "sources": dict(private["sources"]),
        "snapshot_report_dates": dict(private["snapshot_report_dates"]),
        "container_roots": dict(private["container_roots"]),
        "stock_title_class_semantics": list(
            private["stock_title_class_semantics"]
        ),
        "measurement_items": measurement_items,
        "sealed_item_commitments": sealed_commitments,
        "measurement_item_count": MEASUREMENT_ITEM_COUNT,
        "sealed_item_count": SEALED_ITEM_COUNT,
        "sealed_content_persisted": False,
        "ground_truth_persisted": False,
        "private_pack_required_for_sealed_evaluation": True,
        "candidate_imports": 0,
        "model_calls": 0,
        "network_calls": 0,
    }
    return verify_measurement_view(
        _with_self_hash(body, "measurement_view_hash")
    )


def verify_measurement_view(
    value: Mapping[str, Any],
    *,
    private_pack: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PeriodOutPackError("measurement view must be an object")
    payload = dict(value)
    _verify_self_hash(
        payload,
        field="measurement_view_hash",
        label="measurement view",
    )
    _require_exact_fields(
        payload,
        {
            "measurement_view_version",
            "source_policy",
            "selection_policy",
            "private_pack_hash",
            "selection_seed_commitment_hash",
            "sources",
            "snapshot_report_dates",
            "container_roots",
            "stock_title_class_semantics",
            "measurement_items",
            "sealed_item_commitments",
            "measurement_item_count",
            "sealed_item_count",
            "sealed_content_persisted",
            "ground_truth_persisted",
            "private_pack_required_for_sealed_evaluation",
            "candidate_imports",
            "model_calls",
            "network_calls",
            "measurement_view_hash",
        },
        "measurement view",
    )
    if (
        payload.get("measurement_view_version") != MEASUREMENT_VIEW_VERSION
        or payload.get("source_policy") != SOURCE_POLICY
        or payload.get("selection_policy") != SELECTION_POLICY
        or payload.get("sealed_content_persisted") is not False
        or payload.get("ground_truth_persisted") is not False
        or payload.get("private_pack_required_for_sealed_evaluation") is not True
        or payload.get("candidate_imports") != 0
        or payload.get("model_calls") != 0
        or payload.get("network_calls") != 0
    ):
        raise PeriodOutPackError("measurement-view policy drifted")
    for field in (
        "private_pack_hash",
        "selection_seed_commitment_hash",
    ):
        _require_sha256(payload.get(field), f"measurement view {field}")
    sources, _, roots = _validate_source_metadata(
        sources=payload.get("sources"),
        snapshot_dates=payload.get("snapshot_report_dates"),
        roots=payload.get("container_roots"),
        label="measurement view",
    )
    if payload.get("stock_title_class_semantics") != sorted(
        STOCK_TITLE_CLASSES_V1
    ):
        raise PeriodOutPackError("measurement-view stock semantics drifted")
    if payload.get("measurement_item_count") != MEASUREMENT_ITEM_COUNT or (
        payload.get("sealed_item_count") != SEALED_ITEM_COUNT
    ):
        raise PeriodOutPackError("measurement-view counts drifted")
    measurement = payload.get("measurement_items")
    if not isinstance(measurement, list) or len(measurement) != MEASUREMENT_ITEM_COUNT:
        raise PeriodOutPackError("measurement items are malformed")
    for index, item in enumerate(measurement):
        if not isinstance(item, Mapping):
            raise PeriodOutPackError("measurement item is malformed")
        _require_exact_fields(item, _ITEM_FIELDS, "measurement item")
        expected = _item_layout(index)
        query = item.get("query")
        if not isinstance(query, Mapping):
            raise PeriodOutPackError("measurement query is malformed")
        _require_exact_fields(query, _QUERY_FIELDS, "measurement query")
        if (
            item.get("item_id") != expected[0]
            or item.get("partition") != "measurement"
            or item.get("fold") != expected[2]
            or item.get("replicate") != expected[3]
            or item.get("template") != expected[4]
            or not isinstance(item.get("instruction"), str)
            or query.get("increase_top_k") != expected[5]
            or query.get("manager_top_k") != expected[6]
            or query.get("include_stock_count")
            is not (expected[4] == "four_question_v1")
        ):
            raise PeriodOutPackError("measurement item layout drifted")
        for field in ("aum_manager", "increase_manager", "issuer"):
            if not isinstance(query.get(field), str) or not query[field].strip():
                raise PeriodOutPackError("measurement query entity is invalid")
        instruction = render_instruction(
            template=expected[4],
            previous_period_label=str(sources["previous"]["period_label"]),
            current_period_label=str(sources["current"]["period_label"]),
            previous_container_root=str(roots["previous"]),
            current_container_root=str(roots["current"]),
            aum_manager=str(query["aum_manager"]),
            increase_manager=str(query["increase_manager"]),
            issuer=str(query["issuer"]),
            increase_top_k=expected[5],
            manager_top_k=expected[6],
        )
        if (
            item.get("instruction") != instruction
            or item.get("instruction_sha256")
            != hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        ):
            raise PeriodOutPackError("measurement instruction drifted")
    commitments = payload.get("sealed_item_commitments")
    if not isinstance(commitments, list) or len(commitments) != SEALED_ITEM_COUNT:
        raise PeriodOutPackError("sealed commitments are malformed")
    allowed_fields = {
        "item_id",
        "template",
        "fold",
        "instruction_sha256",
        "query_commitment_hash",
        "full_item_commitment_hash",
    }
    for offset, commitment in enumerate(commitments):
        if not isinstance(commitment, Mapping) or set(commitment) != allowed_fields:
            raise PeriodOutPackError("sealed commitment exposed extra fields")
        expected = _item_layout(MEASUREMENT_ITEM_COUNT + offset)
        if (
            commitment.get("item_id") != expected[0]
            or commitment.get("template") != expected[4]
            or commitment.get("fold") is not None
        ):
            raise PeriodOutPackError("sealed commitment layout drifted")
        for field in (
            "instruction_sha256",
            "query_commitment_hash",
            "full_item_commitment_hash",
        ):
            _require_sha256(commitment.get(field), f"sealed commitment {field}")
    if private_pack is not None:
        private = verify_public_pack(private_pack)
        if payload.get("private_pack_hash") != private["pack_hash"]:
            raise PeriodOutPackError("measurement view binds another private pack")
        expected = build_measurement_view(private)
        if payload != expected:
            raise PeriodOutPackError("measurement view differs from private pack")
    return payload


def validate_source_against_pack(
    source: str | Path | Sec13FSource,
    pack: Mapping[str, Any],
    *,
    role: str,
) -> Sec13FSource:
    verified = verify_public_pack(pack)
    if role not in {"previous", "current"}:
        raise PeriodOutPackError("unknown period role")
    opened = source if isinstance(source, Sec13FSource) else Sec13FSource.open(source)
    expected = verified["sources"][role]
    if (
        opened.source_fingerprint != expected["source_fingerprint"]
        or opened.coverpage_sha256 != expected["coverpage_sha256"]
        or opened.infotable_sha256 != expected["infotable_sha256"]
    ):
        raise PeriodOutPackError(f"{role} SEC source differs from public pack")
    return opened


def partition_items(
    pack: Mapping[str, Any], partition: str
) -> tuple[dict[str, Any], ...]:
    verified = verify_public_pack(pack)
    if partition not in _PARTITIONS:
        raise PeriodOutPackError("partition must be measurement or sealed")
    return tuple(
        dict(item)
        for item in verified["items"]
        if item["partition"] == partition
    )


def build_oracle_output(
    *,
    pack: Mapping[str, Any],
    partition: str,
    oracle_id: str,
    answers_by_item: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    verified = verify_public_pack(pack)
    items = partition_items(verified, partition)
    expected_ids = [item["item_id"] for item in items]
    if oracle_id not in REQUIRED_ORACLE_IDS:
        raise PeriodOutPackError("oracle ID is invalid")
    if set(answers_by_item) != set(expected_ids):
        raise PeriodOutPackError("oracle output item set drifted")
    rows: list[dict[str, Any]] = []
    for item in items:
        item_id = item["item_id"]
        answers = dict(answers_by_item[item_id])
        _validate_answers(item, answers)
        rows.append(
            {
                "item_id": item_id,
                "answers": answers,
                "answers_hash": payload_hash(answers),
            }
        )
    body = {
        "oracle_output_version": ORACLE_OUTPUT_VERSION,
        "oracle_id": oracle_id,
        "partition": partition,
        "public_pack_hash": verified["pack_hash"],
        "source_fingerprints": {
            role: verified["sources"][role]["source_fingerprint"]
            for role in ("previous", "current")
        },
        "item_count": len(rows),
        "items": rows,
        "candidate_imports": 0,
        "model_calls": 0,
        "network_calls": 0,
    }
    return _with_self_hash(body, "oracle_output_hash")


def _validate_answers(item: Mapping[str, Any], answers: Mapping[str, Any]) -> None:
    if item["template"] == "four_question_v1":
        keys = ["q1_answer", "q2_answer", "q3_answer", "q4_answer"]
        scalar_keys = ("q1_answer", "q2_answer")
        list_keys = ("q3_answer", "q4_answer")
    else:
        keys = ["q1_answer", "q2_answer", "q3_answer"]
        scalar_keys = ("q1_answer",)
        list_keys = ("q2_answer", "q3_answer")
    if list(answers) != keys:
        raise PeriodOutPackError("oracle answer key order drifted")
    for key in scalar_keys:
        value = answers[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PeriodOutPackError("oracle scalar answer is invalid")
    for key in list_keys:
        value = answers[key]
        if not isinstance(value, list) or not all(
            isinstance(element, str) and element for element in value
        ):
            raise PeriodOutPackError("oracle list answer is invalid")
    query = item["query"]
    if item["template"] == "four_question_v1":
        increase_key, manager_key = "q3_answer", "q4_answer"
    else:
        increase_key, manager_key = "q2_answer", "q3_answer"
    if len(answers[increase_key]) != query["increase_top_k"] or len(
        answers[manager_key]
    ) != query["manager_top_k"]:
        raise PeriodOutPackError("oracle ranked-answer length drifted")


def verify_oracle_output(
    value: Mapping[str, Any],
    *,
    pack: Mapping[str, Any],
    expected_partition: str | None = None,
) -> dict[str, Any]:
    payload = dict(value)
    _verify_self_hash(payload, field="oracle_output_hash", label="oracle output")
    _require_exact_fields(
        payload,
        {
            "oracle_output_version",
            "oracle_id",
            "partition",
            "public_pack_hash",
            "source_fingerprints",
            "item_count",
            "items",
            "candidate_imports",
            "model_calls",
            "network_calls",
            "oracle_output_hash",
        },
        "oracle output",
    )
    verified_pack = verify_public_pack(pack)
    partition = payload.get("partition")
    if expected_partition is not None and partition != expected_partition:
        raise PeriodOutPackError("oracle output partition drifted")
    if partition not in _PARTITIONS:
        raise PeriodOutPackError("oracle output partition is invalid")
    if (
        payload.get("oracle_output_version") != ORACLE_OUTPUT_VERSION
        or payload.get("oracle_id") not in REQUIRED_ORACLE_IDS
        or payload.get("public_pack_hash") != verified_pack["pack_hash"]
        or payload.get("candidate_imports") != 0
        or payload.get("model_calls") != 0
        or payload.get("network_calls") != 0
    ):
        raise PeriodOutPackError("oracle output policy drifted")
    expected_sources = {
        role: verified_pack["sources"][role]["source_fingerprint"]
        for role in ("previous", "current")
    }
    if payload.get("source_fingerprints") != expected_sources:
        raise PeriodOutPackError("oracle output source binding drifted")
    expected_items = partition_items(verified_pack, str(partition))
    rows = payload.get("items")
    if (
        payload.get("item_count") != len(expected_items)
        or not isinstance(rows, list)
        or len(rows) != len(expected_items)
    ):
        raise PeriodOutPackError("oracle output rows are malformed")
    for expected, row in zip(expected_items, rows):
        if not isinstance(row, Mapping) or row.get("item_id") != expected["item_id"]:
            raise PeriodOutPackError("oracle output item order drifted")
        _require_exact_fields(
            row,
            {"item_id", "answers", "answers_hash"},
            "oracle output item",
        )
        answers = row.get("answers")
        if not isinstance(answers, Mapping):
            raise PeriodOutPackError("oracle answers are malformed")
        _validate_answers(expected, answers)
        if row.get("answers_hash") != payload_hash(answers):
            raise PeriodOutPackError("oracle answer hash drifted")
    return payload


def build_consensus_gold(
    *,
    pack: Mapping[str, Any],
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    partition: str,
) -> dict[str, Any]:
    verified_pack = verify_public_pack(pack)
    first = verify_oracle_output(
        left, pack=verified_pack, expected_partition=partition
    )
    second = verify_oracle_output(
        right, pack=verified_pack, expected_partition=partition
    )
    if {first["oracle_id"], second["oracle_id"]} != REQUIRED_ORACLE_IDS:
        raise PeriodOutPackError("consensus requires both frozen oracle IDs")
    first_rows = first["items"]
    second_rows = second["items"]
    if [row["answers_hash"] for row in first_rows] != [
        row["answers_hash"] for row in second_rows
    ] or [row["answers"] for row in first_rows] != [
        row["answers"] for row in second_rows
    ]:
        raise PeriodOutPackError("independent financial oracles disagree")
    rows = [
        {
            "item_id": row["item_id"],
            "answers": row["answers"],
            "answers_hash": row["answers_hash"],
        }
        for row in first_rows
    ]
    body = {
        "gold_version": CONSENSUS_GOLD_VERSION,
        "partition": partition,
        "public_pack_hash": verified_pack["pack_hash"],
        "source_fingerprints": dict(first["source_fingerprints"]),
        "oracle_ids": sorted([first["oracle_id"], second["oracle_id"]]),
        "oracle_output_hashes": sorted(
            [first["oracle_output_hash"], second["oracle_output_hash"]]
        ),
        "item_count": len(rows),
        "items": rows,
        "cross_oracle_agreement": True,
        "candidate_imports": 0,
        "model_calls": 0,
        "network_calls": 0,
    }
    return _with_self_hash(body, "gold_hash")


def verify_consensus_gold(
    value: Mapping[str, Any],
    *,
    pack: Mapping[str, Any],
    expected_partition: str | None = None,
) -> dict[str, Any]:
    payload = dict(value)
    _verify_self_hash(payload, field="gold_hash", label="consensus gold")
    _require_exact_fields(
        payload,
        {
            "gold_version",
            "partition",
            "public_pack_hash",
            "source_fingerprints",
            "oracle_ids",
            "oracle_output_hashes",
            "item_count",
            "items",
            "cross_oracle_agreement",
            "candidate_imports",
            "model_calls",
            "network_calls",
            "gold_hash",
        },
        "consensus gold",
    )
    verified = verify_public_pack(pack)
    partition = payload.get("partition")
    if expected_partition is not None and partition != expected_partition:
        raise PeriodOutPackError("consensus partition drifted")
    if (
        payload.get("gold_version") != CONSENSUS_GOLD_VERSION
        or partition not in _PARTITIONS
        or payload.get("public_pack_hash") != verified["pack_hash"]
        or payload.get("cross_oracle_agreement") is not True
        or payload.get("candidate_imports") != 0
        or payload.get("model_calls") != 0
        or payload.get("network_calls") != 0
    ):
        raise PeriodOutPackError("consensus gold policy drifted")
    items = partition_items(verified, str(partition))
    expected_sources = {
        role: verified["sources"][role]["source_fingerprint"]
        for role in ("previous", "current")
    }
    oracle_hashes = payload.get("oracle_output_hashes")
    if (
        payload.get("source_fingerprints") != expected_sources
        or payload.get("oracle_ids") != sorted(REQUIRED_ORACLE_IDS)
        or not isinstance(oracle_hashes, list)
        or len(oracle_hashes) != 2
        or len(set(oracle_hashes)) != 2
    ):
        raise PeriodOutPackError("consensus oracle provenance drifted")
    for oracle_hash in oracle_hashes:
        _require_sha256(oracle_hash, "consensus oracle output hash")
    rows = payload.get("items")
    if (
        payload.get("item_count") != len(items)
        or not isinstance(rows, list)
        or len(rows) != len(items)
    ):
        raise PeriodOutPackError("consensus gold rows are malformed")
    for item, row in zip(items, rows):
        if not isinstance(row, Mapping) or row.get("item_id") != item["item_id"]:
            raise PeriodOutPackError("consensus gold item order drifted")
        _require_exact_fields(
            row,
            {"item_id", "answers", "answers_hash"},
            "consensus gold item",
        )
        answers = row.get("answers")
        if not isinstance(answers, Mapping):
            raise PeriodOutPackError("consensus answers are malformed")
        _validate_answers(item, answers)
        if row.get("answers_hash") != payload_hash(answers):
            raise PeriodOutPackError("consensus answer hash drifted")
    return payload


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PeriodOutPackError("JSON artifact is unreadable") from exc
    if not isinstance(value, dict):
        raise PeriodOutPackError("JSON artifact must contain one object")
    return value


def write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=destination.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    temporary.replace(destination)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or bind a deterministic SEC 13F period-out pack."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--previous", type=Path, required=True)
    build.add_argument("--current", type=Path, required=True)
    build.add_argument("--previous-label", required=True)
    build.add_argument("--current-label", required=True)
    build.add_argument("--seed", required=True)
    build.add_argument(
        "--previous-container-root", default="/root/period-previous"
    )
    build.add_argument("--current-container-root", default="/root/period-current")
    build.add_argument("--output", type=Path, required=True)

    agree = commands.add_parser("agree")
    agree.add_argument("--pack", type=Path, required=True)
    agree.add_argument("--left", type=Path, required=True)
    agree.add_argument("--right", type=Path, required=True)
    agree.add_argument("--partition", choices=sorted(_PARTITIONS), required=True)
    agree.add_argument("--output", type=Path, required=True)

    view = commands.add_parser("measurement-view")
    view.add_argument("--pack", type=Path, required=True)
    view.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--pack", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        pack = build_public_pack(
            previous_source=args.previous,
            current_source=args.current,
            previous_period_label=args.previous_label,
            current_period_label=args.current_label,
            preregistration_seed=args.seed,
            previous_container_root=args.previous_container_root,
            current_container_root=args.current_container_root,
        )
        write_json(args.output, pack)
        return 0
    if args.command == "agree":
        pack = verify_public_pack(read_json(args.pack))
        gold = build_consensus_gold(
            pack=pack,
            left=read_json(args.left),
            right=read_json(args.right),
            partition=args.partition,
        )
        write_json(args.output, gold)
        return 0
    if args.command == "measurement-view":
        view = build_measurement_view(read_json(args.pack))
        write_json(args.output, view)
        return 0
    verify_public_pack(read_json(args.pack))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI APIs.
    raise SystemExit(main())
