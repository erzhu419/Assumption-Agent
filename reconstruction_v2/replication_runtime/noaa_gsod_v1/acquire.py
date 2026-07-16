from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen

from .contract import (
    GSOD_FILE_URL_TEMPLATE,
    GSOD_YEAR_INDEX_URL,
    ITEM_COUNT,
    MIN_UNIQUE_DATES,
    MIN_VALID_DAYS_PER_MONTH,
    PARTITION_COUNTS,
    SELECTION_SEED,
    STATION_METADATA_URL,
    NoaaGsodError,
    assess_completeness,
    canonical_json_bytes,
    payload_hash,
    sha256_file,
)
from .pack import build_private_pack, build_public_receipt, write_json


_INDEX_FILE = re.compile(rb"(?<![0-9])([0-9]{11})\.csv(?![A-Za-z0-9])")


def _download(url: str, destination: Path, *, allow_network: bool) -> bool:
    if destination.is_file() and destination.stat().st_size > 0:
        return False
    if not allow_network:
        raise NoaaGsodError(f"required cached source is absent: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    request = Request(url, headers={"User-Agent": "assumption-agent-research/1.0"})
    try:
        with urlopen(request, timeout=90) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise NoaaGsodError("official source download is empty")
    temporary.replace(destination)
    return True


def parse_year_index(path: str | Path) -> frozenset[str]:
    station_ids = {match.decode("ascii") for match in _INDEX_FILE.findall(Path(path).read_bytes())}
    if not station_ids:
        raise NoaaGsodError("year index contains no station CSV names")
    return frozenset(station_ids)


def parse_us_full_year_metadata(path: str | Path) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"USAF", "WBAN", "CTRY", "BEGIN", "END"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise NoaaGsodError("station metadata header is incompatible")
        for raw in reader:
            row = {str(key): str(value or "").strip() for key, value in raw.items()}
            station_id = row["USAF"] + row["WBAN"]
            if (
                row["CTRY"] != "US"
                or len(station_id) != 11
                or not station_id.isdigit()
                or row["BEGIN"] > "20200101"
                or row["END"] < "20201231"
            ):
                continue
            prior = selected.get(station_id)
            if prior is None or canonical_json_bytes(row) < canonical_json_bytes(prior):
                selected[station_id] = row
    if not selected:
        raise NoaaGsodError("station metadata has no full-year US station")
    return selected


def ranked_candidate_ids(
    metadata: Mapping[str, Mapping[str, str]], indexed: frozenset[str]
) -> list[str]:
    candidates = set(metadata).intersection(indexed)
    return sorted(
        candidates,
        key=lambda station_id: (
            hashlib.sha256(f"{SELECTION_SEED}:{station_id}".encode("ascii")).hexdigest(),
            station_id,
        ),
    )


def acquire(
    *,
    artifact_root: str | Path,
    receipt_path: str | Path,
    allow_network: bool = True,
) -> dict[str, Any]:
    root = Path(artifact_root).resolve()
    source_root = root / "official_sources"
    candidate_root = root / "candidate_station_csv"
    private_root = root / "private_pack"
    metadata_path = source_root / "isd-history.csv"
    index_path = source_root / "2020-index.html"
    actual_network_calls = int(
        _download(STATION_METADATA_URL, metadata_path, allow_network=allow_network)
    )
    actual_network_calls += int(
        _download(GSOD_YEAR_INDEX_URL, index_path, allow_network=allow_network)
    )

    metadata = parse_us_full_year_metadata(metadata_path)
    indexed = parse_year_index(index_path)
    candidates = ranked_candidate_ids(metadata, indexed)
    if len(candidates) < ITEM_COUNT:
        raise NoaaGsodError("official sources provide fewer candidates than required")

    accepted: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    checked = 0
    for station_id in candidates:
        if len(accepted) == ITEM_COUNT:
            break
        checked += 1
        station_path = candidate_root / f"{station_id}.csv"
        actual_network_calls += int(
            _download(
                GSOD_FILE_URL_TEMPLATE.format(station_id=station_id),
                station_path,
                allow_network=allow_network,
            )
        )
        completeness = assess_completeness(station_path, station_id)
        if not completeness.eligible:
            rejections[completeness.reason] += 1
            continue
        accepted.append(
            {
                "source_path": str(station_path),
                "station_id": station_id,
                "station_metadata_commitment": payload_hash(metadata[station_id]),
            }
        )
    if len(accepted) != ITEM_COUNT:
        raise NoaaGsodError("deterministic candidate scan did not yield 24 complete stations")

    statistics: dict[str, Any] = {
        "accepted_station_count": len(accepted),
        "candidate_files_checked": checked,
        "candidate_intersection_count": len(candidates),
        "completeness_rule": {
            "all_12_months_required": True,
            "minimum_unique_dates": MIN_UNIQUE_DATES,
            "minimum_valid_prcp_days_per_month": MIN_VALID_DAYS_PER_MONTH,
            "required_columns": ["STATION", "DATE", "PRCP"],
            "single_station_and_unique_dates_required": True,
        },
        "full_year_us_metadata_station_count": len(metadata),
        "indexed_station_file_count": len(indexed),
        "official_source_object_count_bound": checked + 2,
        "rejected_candidate_count": sum(rejections.values()),
        "rejection_reason_counts": dict(sorted(rejections.items())),
    }
    private_pack = build_private_pack(
        selected=accepted,
        private_root=private_root,
        metadata_sha256=sha256_file(metadata_path),
        index_sha256=sha256_file(index_path),
        acquisition_statistics=statistics,
    )
    receipt = build_public_receipt(
        private_pack,
        metadata_url=STATION_METADATA_URL,
        index_url=GSOD_YEAR_INDEX_URL,
        network_calls=checked + 2,
    )
    write_json(receipt_path, receipt)
    safe_summary = {
        "accepted_station_count": len(accepted),
        "cache_network_calls_this_invocation": actual_network_calls,
        "candidate_files_checked": checked,
        "pack_hash": private_pack["pack_hash"],
        "partition_counts": PARTITION_COUNTS,
        "receipt_hash": receipt["receipt_hash"],
        "source_commitment_set_hash": payload_hash(private_pack["source_commitments"]),
    }
    return safe_summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Acquire and privately pack the frozen NOAA GSOD 2020 station-out corpus."
    )
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--offline", action="store_true")
    arguments = parser.parse_args(argv)
    summary = acquire(
        artifact_root=arguments.artifact_root,
        receipt_path=arguments.receipt,
        allow_network=not arguments.offline,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
