from __future__ import annotations

import copy
import csv
from datetime import date, timedelta
import json
from pathlib import Path

import pytest

from replication_runtime.noaa_gsod_v1 import oracle_sqlite, oracle_stdlib
from replication_runtime.noaa_gsod_v1.acquire import (
    parse_us_full_year_metadata,
    parse_year_index,
    ranked_candidate_ids,
)
from replication_runtime.noaa_gsod_v1.contract import (
    ORACLE_IDS,
    PARTITION_COUNTS,
    TASK_CONTRACT,
    NoaaGsodError,
    assess_completeness,
)
from replication_runtime.noaa_gsod_v1.pack import (
    build_private_pack,
    build_public_receipt,
    verify_private_pack,
    verify_public_receipt,
)
from replication_runtime.noaa_gsod_v1.schemas import SCHEMA_SET, SCHEMA_SET_HASH
from replication_runtime.noaa_gsod_v1.train_export import (
    export_train_view,
    verify_train_preparation_receipt,
    verify_train_view,
)


def _write_station(path: Path, station_id: str, *, tie: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = date(2020, 1, 1)
    end = date(2021, 1, 1)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["STATION", "DATE", "PRCP", "NAME"])
        writer.writeheader()
        while current < end:
            if current.month == 1:
                precipitation = "1.00"
            elif current.month == 2:
                precipitation = "1.00" if tie else "1.01"
            else:
                precipitation = "0.10"
            if current == date(2020, 1, 1):
                precipitation = "99.99"
            writer.writerow(
                {
                    "STATION": station_id,
                    "DATE": current.isoformat(),
                    "PRCP": precipitation,
                    "NAME": "synthetic fixture",
                }
            )
            current += timedelta(days=1)


def test_independent_oracles_cover_missing_group_argmax_tie_convert_and_round(
    tmp_path: Path,
) -> None:
    source = tmp_path / "station.csv"
    _write_station(source, "70000100001")
    completeness = assess_completeness(source, "70000100001")
    assert completeness.eligible
    assert completeness.unique_dates == 366
    assert completeness.month_count == 12
    first = oracle_stdlib.evaluate(source)
    second = oracle_sqlite.evaluate(source)
    assert first == second == {
        "mean_daily_precip_mm": "25.40",
        "month": "01",
        "valid_day_count": 30,
    }

    untied = tmp_path / "untied.csv"
    _write_station(untied, "70000100002", tie=False)
    assert oracle_stdlib.evaluate(untied)["month"] == "02"
    assert oracle_sqlite.evaluate(untied) == oracle_stdlib.evaluate(untied)


def test_official_index_metadata_filter_and_seeded_rank_are_deterministic(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        '"USAF","WBAN","STATION NAME","CTRY","STATE","ICAO","LAT","LON","ELEV(M)","BEGIN","END"\n'
        '"700001","00001","A","US","AA","","0","0","0","20190101","20211231"\n'
        '"700001","00002","B","CA","AA","","0","0","0","20190101","20211231"\n'
        '"700001","00003","C","US","AA","","0","0","0","20200601","20211231"\n'
        '"700001","00004","D","US","AA","","0","0","0","20190101","20201231"\n',
        encoding="utf-8",
    )
    index = tmp_path / "index.html"
    index.write_bytes(
        b'<a href="70000100001.csv">one</a><a href="70000100004.csv">four</a>'
    )
    rows = parse_us_full_year_metadata(metadata)
    indexed = parse_year_index(index)
    assert set(rows) == {"70000100001", "70000100004"}
    assert ranked_candidate_ids(rows, indexed) == ranked_candidate_ids(rows, indexed)
    assert set(ranked_candidate_ids(rows, indexed)) == set(rows)


def test_private_pack_closes_inputs_and_public_receipt_redacts_station_and_gold(
    tmp_path: Path,
) -> None:
    selected = []
    station_ids = []
    for index in range(24):
        station_id = f"71{index:09d}"
        station_ids.append(station_id)
        source = tmp_path / "sources" / f"source-{index}.csv"
        _write_station(source, station_id)
        selected.append(
            {
                "source_path": str(source),
                "station_id": station_id,
                "station_metadata_commitment": f"{index:064x}",
            }
        )
    private_root = tmp_path / "private"
    private = build_private_pack(
        selected=selected,
        private_root=private_root,
        metadata_sha256="a" * 64,
        index_sha256="b" * 64,
        acquisition_statistics={"accepted_station_count": 24},
    )
    assert verify_private_pack(private, private_root=private_root) == private
    assert tuple(private["oracle_ids"]) == ORACLE_IDS
    assert {key: sum(item["partition"] == key for item in private["items"]) for key in PARTITION_COUNTS} == PARTITION_COUNTS

    public = build_public_receipt(
        private,
        metadata_url="https://official.example/metadata.csv",
        index_url="https://official.example/2020/",
        network_calls=26,
    )
    assert verify_public_receipt(public) == public
    serialized = json.dumps(public, sort_keys=True)
    for station_id in station_ids:
        assert station_id not in serialized
    assert "mean_daily_precip_mm" not in serialized
    assert "raw_csv_relative_path" not in serialized
    assert public["content_boundary"]["sealed_raw_task_git_persisted"] is False
    assert public["content_boundary"]["sealed_gold_git_persisted"] is False

    tampered = copy.deepcopy(public)
    tampered["partition_counts"]["sealed"] = 5
    with pytest.raises(NoaaGsodError, match="receipt_hash mismatch"):
        verify_public_receipt(tampered)


def test_task_contract_is_finite_relational_and_not_station_specific() -> None:
    capabilities = set(TASK_CONTRACT["operator_capabilities"])
    assert {
        "missing_normalization",
        "group",
        "aggregate_sum_count",
        "argmax",
        "stable_tie_break",
        "unit_conversion",
        "decimal_round_half_up",
        "json_serialize",
    }.issubset(capabilities)
    assert "station_id" not in json.dumps(TASK_CONTRACT, sort_keys=True)
    assert SCHEMA_SET["private_pack"]["additionalProperties"] is False
    assert len(SCHEMA_SET_HASH) == 64


def test_train_export_is_anonymous_and_excludes_every_non_train_mapping(
    tmp_path: Path,
) -> None:
    selected = []
    station_ids = []
    for index in range(24):
        station_id = f"72{index:09d}"
        station_ids.append(station_id)
        source = tmp_path / "sources" / f"source-{index}.csv"
        _write_station(source, station_id)
        selected.append(
            {
                "source_path": str(source),
                "station_id": station_id,
                "station_metadata_commitment": f"{index + 100:064x}",
            }
        )
    private_root = tmp_path / "private"
    private = build_private_pack(
        selected=selected,
        private_root=private_root,
        metadata_sha256="c" * 64,
        index_sha256="d" * 64,
        acquisition_statistics={"accepted_station_count": 24},
    )
    train_root = tmp_path / "train-view"
    receipt_path = tmp_path / "train-preparation-receipt.json"
    safe_summary = export_train_view(
        private_pack_path=private_root / "private_pack.json",
        private_root=private_root,
        train_view_root=train_root,
        receipt_path=receipt_path,
    )
    assert safe_summary["train_item_count"] == 12

    train_view = json.loads((train_root / "train_view.json").read_text())
    assert verify_train_view(train_view, train_view_root=train_root) == train_view
    assert len(list((train_root / "inputs").glob("*.csv"))) == 12
    exported_bytes = (train_root / "train_view.json").read_bytes() + b"".join(
        path.read_bytes() for path in sorted((train_root / "inputs").glob("*.csv"))
    )
    for station_id in station_ids:
        assert station_id.encode() not in exported_bytes
    serialized = json.dumps(train_view, sort_keys=True)
    assert '"development"' not in serialized
    assert '"sealed"' not in serialized
    train_gold_commitments = {
        item["gold_commitment"]
        for item in private["items"]
        if item["partition"] == "train"
    }
    for source_item in private["items"]:
        if source_item["partition"] != "train":
            assert source_item["raw_csv_sha256"] not in serialized
            assert source_item["item_commitment"] not in serialized
            if source_item["gold_commitment"] not in train_gold_commitments:
                assert source_item["gold_commitment"] not in serialized
    for ordinal, item in enumerate(train_view["items"]):
        with (train_root / item["input_relative_path"]).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        assert set(rows[0]) == {"STATION", "DATE", "PRCP"}
        assert {row["STATION"] for row in rows} == {f"TRAIN_STATION_{ordinal:02d}"}

    receipt = json.loads(receipt_path.read_text())
    assert verify_train_preparation_receipt(receipt) == receipt
    receipt_serialized = json.dumps(receipt, sort_keys=True)
    assert "oracle_consensus" not in receipt_serialized
    assert "mean_daily_precip_mm" not in receipt_serialized
    assert receipt["content_boundary"]["sealed_raw_exported"] is False
    assert receipt["content_boundary"]["development_gold_exported"] is False

    first_input = train_root / train_view["items"][0]["input_relative_path"]
    first_input.write_text(first_input.read_text() + "\n", encoding="utf-8")
    with pytest.raises(NoaaGsodError, match="input hash mismatch"):
        verify_train_view(train_view, train_view_root=train_root)


def test_train_export_refuses_destructive_or_mixed_output_roots(
    tmp_path: Path,
) -> None:
    selected = []
    for index in range(24):
        station_id = f"73{index:09d}"
        source = tmp_path / "sources" / f"source-{index}.csv"
        _write_station(source, station_id)
        selected.append(
            {
                "source_path": str(source),
                "station_id": station_id,
                "station_metadata_commitment": f"{index + 200:064x}",
            }
        )
    private_root = tmp_path / "private"
    build_private_pack(
        selected=selected,
        private_root=private_root,
        metadata_sha256="e" * 64,
        index_sha256="f" * 64,
        acquisition_statistics={"accepted_station_count": 24},
    )
    arguments = {
        "private_pack_path": private_root / "private_pack.json",
        "private_root": private_root,
        "receipt_path": tmp_path / "receipt.json",
    }
    with pytest.raises(NoaaGsodError, match="disjoint"):
        export_train_view(train_view_root=private_root, **arguments)

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("preserve me", encoding="utf-8")
    with pytest.raises(NoaaGsodError, match="refusing to replace"):
        export_train_view(train_view_root=unrelated, **arguments)
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "preserve me"

    train_root = tmp_path / "train-view"
    export_train_view(train_view_root=train_root, **arguments)
    first_hash = json.loads((train_root / "train_view.json").read_text())[
        "train_view_hash"
    ]
    export_train_view(train_view_root=train_root, **arguments)
    assert json.loads((train_root / "train_view.json").read_text())[
        "train_view_hash"
    ] == first_hash

    with pytest.raises(NoaaGsodError, match="receipt must be outside"):
        export_train_view(
            private_pack_path=private_root / "private_pack.json",
            private_root=private_root,
            train_view_root=tmp_path / "other-train-view",
            receipt_path=tmp_path / "other-train-view" / "receipt.json",
        )
