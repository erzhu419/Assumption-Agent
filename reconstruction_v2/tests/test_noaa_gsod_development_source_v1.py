from __future__ import annotations

import copy
import csv
from datetime import date, timedelta
import json
from pathlib import Path

import pytest

import replication_runtime.noaa_gsod_v1.development_source as development_source
from replication_runtime.noaa_gsod_v1.contract import (
    TASK_CONTRACT,
    NoaaGsodError,
    payload_hash,
    with_self_hash,
)
from replication_runtime.noaa_gsod_v1.development_source import (
    ANONYMOUS_COLUMNS,
    DEVELOPMENT_ITEM_COUNT,
    DEVELOPMENT_SOURCE_INDEX_FIELDS,
    DEVELOPMENT_SOURCE_RECEIPT_FIELDS,
    PRIVATE_INDEX_NAME,
    export_development_source_view,
    verify_development_source_bundle,
    verify_development_source_index,
    verify_public_development_source_receipt,
)
from replication_runtime.noaa_gsod_v1.pack import (
    build_private_pack,
    build_public_receipt,
    read_json,
    write_json,
)


def _write_station(path: Path, station_id: str, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = date(2020, 1, 1)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["STATION", "DATE", "PRCP", "NAME", "LATITUDE"],
        )
        writer.writeheader()
        while current.year == 2020:
            writer.writerow(
                {
                    "STATION": station_id,
                    "DATE": current.isoformat(),
                    "PRCP": f"{((current.month + offset) % 7) / 10:.2f}",
                    "NAME": f"private synthetic station {offset}",
                    "LATITUDE": str(offset),
                }
            )
            current += timedelta(days=1)


def _source_bundle(tmp_path: Path) -> dict[str, object]:
    selected = []
    station_ids = []
    for index in range(24):
        station_id = f"73{index:09d}"
        station_ids.append(station_id)
        source = tmp_path / "source" / f"station-{index}.csv"
        _write_station(source, station_id, index)
        selected.append(
            {
                "source_path": str(source),
                "station_id": station_id,
                "station_metadata_commitment": f"{index + 200:064x}",
            }
        )
    private_root = tmp_path / "private-pack"
    private = build_private_pack(
        selected=selected,
        private_root=private_root,
        metadata_sha256="a" * 64,
        index_sha256="b" * 64,
        acquisition_statistics={"accepted_station_count": 24},
    )
    acquisition = build_public_receipt(
        private,
        metadata_url="https://official.example/metadata.csv",
        index_url="https://official.example/2020/",
        network_calls=26,
    )
    acquisition_path = tmp_path / "acquisition.json"
    write_json(acquisition_path, acquisition)
    return {
        "acquisition": acquisition,
        "acquisition_path": acquisition_path,
        "private": private,
        "private_root": private_root,
        "station_ids": station_ids,
    }


def _export(tmp_path: Path) -> tuple[dict[str, object], Path, Path, dict, dict]:
    bundle = _source_bundle(tmp_path)
    source_view_root = tmp_path / "development-source-view"
    public_receipt_path = tmp_path / "development-source-receipt.json"
    export_development_source_view(
        private_pack_path=Path(bundle["private_root"]) / "private_pack.json",
        private_pack_root=Path(bundle["private_root"]),
        acquisition_receipt_path=Path(bundle["acquisition_path"]),
        source_view_root=source_view_root,
        public_receipt_path=public_receipt_path,
    )
    index = read_json(source_view_root / PRIVATE_INDEX_NAME)
    receipt = read_json(public_receipt_path)
    return bundle, source_view_root, public_receipt_path, index, receipt


def test_custodian_exports_exact_gold_free_anonymous_development_view(
    tmp_path: Path,
) -> None:
    bundle, root, public_path, index, receipt = _export(tmp_path)
    acquisition = bundle["acquisition"]
    assert set(index) == DEVELOPMENT_SOURCE_INDEX_FIELDS
    assert set(receipt) == DEVELOPMENT_SOURCE_RECEIPT_FIELDS
    assert index["development_item_count"] == DEVELOPMENT_ITEM_COUNT
    assert len(index["items"]) == DEVELOPMENT_ITEM_COUNT
    assert verify_development_source_index(
        index,
        source_view_root=root,
        expected_development_commitments=acquisition[
            "item_commitments_by_partition"
        ]["development"],
    ) == index
    assert verify_public_development_source_receipt(receipt) == receipt
    assert verify_development_source_bundle(
        receipt,
        source_view_root=root,
        acquisition_receipt_path=bundle["acquisition_path"],
    ) == receipt
    assert receipt["development_input_materialized"] is True
    assert all(value == 0 for value in receipt["call_ledger"].values())
    assert receipt["binding_hashes"]["task_contract_hash"] == payload_hash(
        TASK_CONTRACT
    )
    assert receipt["custody_boundary"] == {
        "acquisition_custodian_development_identity_accessed": True,
        "acquisition_custodian_monolithic_private_pack_accessed": True,
        "development_controller_gold_or_oracle_exposed": False,
        "development_controller_monolithic_private_pack_exposed": False,
        "development_controller_sealed_mapping_exposed": False,
        "sealed_runtime_any_item_material_exposed": False,
        "source_view_contains_development_inputs_only": True,
    }

    serialized_private = json.dumps(index, sort_keys=True)
    serialized_public = public_path.read_text(encoding="utf-8")
    for forbidden_key in (
        "gold_commitment",
        "oracle_outputs",
        "raw_csv_relative_path",
        "station_id",
    ):
        assert f'"{forbidden_key}"' not in serialized_private
        assert f'"{forbidden_key}"' not in serialized_public
    private = bundle["private"]
    sealed_commitments = {
        item["item_commitment"]
        for item in private["items"]
        if item["partition"] == "sealed"
    }
    assert not any(value in serialized_private for value in sealed_commitments)
    assert not any(value in serialized_public for value in sealed_commitments)

    staged_bytes = b"".join(
        (root / item["input_relative_path"]).read_bytes() for item in index["items"]
    )
    for station_id in bundle["station_ids"]:
        assert station_id.encode("ascii") not in staged_bytes
    for ordinal, item in enumerate(index["items"]):
        with (root / item["input_relative_path"]).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            assert tuple(reader.fieldnames or ()) == ANONYMOUS_COLUMNS
            assert all(
                row["STATION"] == f"DEVELOPMENT_STATION_{ordinal:02d}"
                for row in reader
            )


def test_index_and_receipt_fail_closed_on_nested_schema_and_file_tampering(
    tmp_path: Path,
) -> None:
    bundle, root, _, index, receipt = _export(tmp_path)
    index_body = copy.deepcopy(index)
    index_body.pop("private_index_hash")
    index_body["items"][0]["unexpected"] = False
    tampered_index = with_self_hash(index_body, "private_index_hash")
    with pytest.raises(NoaaGsodError, match="private item schema"):
        verify_development_source_index(
            tampered_index,
            source_view_root=root,
        )

    receipt_body = copy.deepcopy(receipt)
    receipt_body.pop("development_source_receipt_hash")
    receipt_body["binding_hashes"].pop("private_index_hash")
    tampered_receipt = with_self_hash(
        receipt_body,
        "development_source_receipt_hash",
    )
    with pytest.raises(NoaaGsodError, match="binding schema"):
        verify_public_development_source_receipt(tampered_receipt)

    unexpected = root / "unexpected-private-material.json"
    unexpected.write_text("{}\n", encoding="utf-8")
    with pytest.raises(NoaaGsodError, match="unexpected material"):
        verify_development_source_index(index, source_view_root=root)
    unexpected.unlink()

    first_path = root / index["items"][0]["input_relative_path"]
    first_path.write_bytes(first_path.read_bytes() + b"\n")
    with pytest.raises(NoaaGsodError, match="input hash mismatch"):
        verify_development_source_bundle(
            receipt,
            source_view_root=root,
            acquisition_receipt_path=bundle["acquisition_path"],
        )


def test_custodian_rejects_acquisition_development_commitment_drift(
    tmp_path: Path,
) -> None:
    bundle = _source_bundle(tmp_path)
    acquisition = copy.deepcopy(bundle["acquisition"])
    acquisition.pop("receipt_hash")
    acquisition["item_commitments_by_partition"]["development"][0] = "f" * 64
    drifted = with_self_hash(acquisition, "receipt_hash")
    drifted_path = tmp_path / "drifted-acquisition.json"
    write_json(drifted_path, drifted)
    with pytest.raises(NoaaGsodError, match="public acquisition commitments"):
        export_development_source_view(
            private_pack_path=Path(bundle["private_root"]) / "private_pack.json",
            private_pack_root=Path(bundle["private_root"]),
            acquisition_receipt_path=drifted_path,
            source_view_root=tmp_path / "view",
            public_receipt_path=tmp_path / "public.json",
        )


@pytest.mark.parametrize("relation", ["inside", "ancestor"])
def test_source_view_must_not_overlap_private_pack_root(
    tmp_path: Path,
    relation: str,
) -> None:
    bundle = _source_bundle(tmp_path)
    private_root = Path(bundle["private_root"])
    source_view_root = (
        private_root / "development-source"
        if relation == "inside"
        else private_root.parent
    )
    with pytest.raises(NoaaGsodError, match="overlaps private pack"):
        export_development_source_view(
            private_pack_path=private_root / "private_pack.json",
            private_pack_root=private_root,
            acquisition_receipt_path=Path(bundle["acquisition_path"]),
            source_view_root=source_view_root,
            public_receipt_path=tmp_path / f"public-{relation}.json",
        )


def test_public_receipt_is_no_clobber_and_outside_private_roots(
    tmp_path: Path,
) -> None:
    bundle = _source_bundle(tmp_path)
    private_root = Path(bundle["private_root"])
    public_path = tmp_path / "public.json"
    public_path.write_bytes(b"external-owner\n")
    with pytest.raises(NoaaGsodError, match="no-clobber"):
        export_development_source_view(
            private_pack_path=private_root / "private_pack.json",
            private_pack_root=private_root,
            acquisition_receipt_path=Path(bundle["acquisition_path"]),
            source_view_root=tmp_path / "view",
            public_receipt_path=public_path,
        )
    assert public_path.read_bytes() == b"external-owner\n"
    assert not (tmp_path / "view").exists()

    with pytest.raises(NoaaGsodError, match="public source receipt overlaps"):
        export_development_source_view(
            private_pack_path=private_root / "private_pack.json",
            private_pack_root=private_root,
            acquisition_receipt_path=Path(bundle["acquisition_path"]),
            source_view_root=tmp_path / "view-two",
            public_receipt_path=private_root / "public.json",
        )


def test_public_receipt_atomic_publish_preserves_racing_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _source_bundle(tmp_path)
    private_root = Path(bundle["private_root"])
    public_path = tmp_path / "public.json"
    view_root = tmp_path / "view"
    original_link = development_source.os.link

    def racing_link(source, destination):
        Path(destination).write_bytes(b"racing-owner\n")
        return original_link(source, destination)

    monkeypatch.setattr(development_source.os, "link", racing_link)
    with pytest.raises(NoaaGsodError, match="no-clobber"):
        export_development_source_view(
            private_pack_path=private_root / "private_pack.json",
            private_pack_root=private_root,
            acquisition_receipt_path=Path(bundle["acquisition_path"]),
            source_view_root=view_root,
            public_receipt_path=public_path,
        )
    assert public_path.read_bytes() == b"racing-owner\n"
    assert not view_root.exists()
