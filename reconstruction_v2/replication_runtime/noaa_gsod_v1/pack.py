from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from . import oracle_sqlite, oracle_stdlib
from .contract import (
    ITEM_COUNT,
    ORACLE_IDS,
    PACK_VERSION,
    PARTITION_COUNTS,
    PUBLIC_RECEIPT_VERSION,
    SELECTION_SEED,
    STUDY_ID,
    TASK_CONTRACT,
    NoaaGsodError,
    canonical_json_bytes,
    payload_hash,
    sha256_file,
    verify_self_hash,
    with_self_hash,
)
from .schemas import (
    ORACLE_RESULT_FIELDS,
    PRIVATE_ITEM_FIELDS,
    PRIVATE_PACK_FIELDS,
    PUBLIC_RECEIPT_FIELDS,
    SCHEMA_SET_HASH,
)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(payload) + b"\n")
    temporary.replace(destination)


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise NoaaGsodError("JSON root is not an object")
    return value


def _partition_for_position(position: int) -> tuple[str, int]:
    if 0 <= position < 12:
        return "train", position
    if 12 <= position < 18:
        return "development", position - 12
    if 18 <= position < 24:
        return "sealed", position - 18
    raise NoaaGsodError("selected position exceeds the frozen layout")


def build_private_pack(
    *,
    selected: Sequence[Mapping[str, Any]],
    private_root: str | Path,
    metadata_sha256: str,
    index_sha256: str,
    acquisition_statistics: Mapping[str, Any],
) -> dict[str, Any]:
    if len(selected) != ITEM_COUNT:
        raise NoaaGsodError(f"formation requires exactly {ITEM_COUNT} stations")
    root = Path(private_root).resolve()
    item_root = root / "items"
    item_root.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    station_ids: set[str] = set()
    for position, selected_item in enumerate(selected):
        station_id = str(selected_item["station_id"])
        if station_id in station_ids:
            raise NoaaGsodError("selected station ids are not unique")
        station_ids.add(station_id)
        source_path = Path(str(selected_item["source_path"])).resolve(strict=True)
        partition, ordinal = _partition_for_position(position)
        item_id = f"noaa_gsod_{partition}_{ordinal:02d}"
        relative_path = f"items/{item_id}.csv"
        destination = root / relative_path
        shutil.copyfile(source_path, destination)
        raw_sha256 = sha256_file(destination)
        task_payload_commitment = payload_hash(
            {
                "raw_csv_sha256": raw_sha256,
                "task_contract_hash": payload_hash(TASK_CONTRACT),
            }
        )
        first = oracle_stdlib.evaluate(destination)
        second = oracle_sqlite.evaluate(destination)
        if first != second:
            raise NoaaGsodError("independent oracle disagreement")
        gold_commitment = payload_hash(first)
        item_body = {
            "gold_commitment": gold_commitment,
            "item_id": item_id,
            "oracle_outputs": {
                oracle_stdlib.ORACLE_ID: first,
                oracle_sqlite.ORACLE_ID: second,
            },
            "ordinal": ordinal,
            "partition": partition,
            "raw_csv_relative_path": relative_path,
            "raw_csv_sha256": raw_sha256,
            "station_id": station_id,
            "station_metadata_commitment": str(
                selected_item["station_metadata_commitment"]
            ),
            "task_payload_commitment": task_payload_commitment,
        }
        item_body["item_commitment"] = payload_hash(item_body)
        items.append(item_body)
    pack_body: dict[str, Any] = {
        "acquisition_statistics": dict(acquisition_statistics),
        "candidate_imports": 0,
        "ground_truth_persisted_only_in_private_root": True,
        "items": items,
        "model_calls": 0,
        "online_judge_calls": 0,
        "oracle_ids": list(ORACLE_IDS),
        "pack_version": PACK_VERSION,
        "partition_counts": dict(PARTITION_COUNTS),
        "scoring_calls": 0,
        "selection_policy": {
            "candidate_order": "sha256(seed + ':' + station_id), then station_id",
            "eligibility_precedes_partition_assignment": True,
            "partition_assignment": "accepted positions 0:12 TRAIN, 12:18 development, 18:24 residual sealed",
            "seed": SELECTION_SEED,
            "station_out_disjoint": True,
        },
        "source_commitments": {
            "station_metadata_sha256": metadata_sha256,
            "year_index_sha256": index_sha256,
        },
        "study_id": STUDY_ID,
        "task_contract": TASK_CONTRACT,
        "task_contract_hash": payload_hash(TASK_CONTRACT),
        "year": 2020,
    }
    private_pack = with_self_hash(pack_body, "pack_hash")
    verify_private_pack(private_pack, private_root=root)
    write_json(root / "private_pack.json", private_pack)
    return private_pack


def verify_private_pack(
    payload: Mapping[str, Any], *, private_root: str | Path
) -> dict[str, Any]:
    pack = dict(payload)
    verify_self_hash(pack, "pack_hash")
    if set(pack) != PRIVATE_PACK_FIELDS:
        raise NoaaGsodError("private pack schema mismatch")
    if pack.get("pack_version") != PACK_VERSION or pack.get("study_id") != STUDY_ID:
        raise NoaaGsodError("private pack identity mismatch")
    if pack.get("task_contract") != TASK_CONTRACT:
        raise NoaaGsodError("private pack task contract mismatch")
    if pack.get("task_contract_hash") != payload_hash(TASK_CONTRACT):
        raise NoaaGsodError("private pack task contract hash mismatch")
    if tuple(pack.get("oracle_ids", ())) != ORACLE_IDS:
        raise NoaaGsodError("private pack oracle ids mismatch")
    items = pack.get("items")
    if not isinstance(items, list) or len(items) != ITEM_COUNT:
        raise NoaaGsodError("private pack item count mismatch")
    root = Path(private_root).resolve()
    seen_stations: set[str] = set()
    counts = {key: 0 for key in PARTITION_COUNTS}
    for item in items:
        if not isinstance(item, dict):
            raise NoaaGsodError("private item is not an object")
        if set(item) != PRIVATE_ITEM_FIELDS:
            raise NoaaGsodError("private item schema mismatch")
        body = dict(item)
        declared_item_hash = body.pop("item_commitment", None)
        if declared_item_hash != payload_hash(body):
            raise NoaaGsodError("private item commitment mismatch")
        partition = item.get("partition")
        if partition not in counts:
            raise NoaaGsodError("private item partition mismatch")
        counts[str(partition)] += 1
        station_id = str(item.get("station_id", ""))
        if station_id in seen_stations:
            raise NoaaGsodError("station appears in multiple items")
        seen_stations.add(station_id)
        relative = str(item.get("raw_csv_relative_path", ""))
        path = (root / relative).resolve(strict=True)
        if root not in path.parents or sha256_file(path) != item.get("raw_csv_sha256"):
            raise NoaaGsodError("private item raw source binding mismatch")
        outputs = item.get("oracle_outputs")
        if not isinstance(outputs, dict) or set(outputs) != set(ORACLE_IDS):
            raise NoaaGsodError("private item oracle output schema mismatch")
        if outputs[ORACLE_IDS[0]] != outputs[ORACLE_IDS[1]]:
            raise NoaaGsodError("private item oracle outputs disagree")
        if set(outputs[ORACLE_IDS[0]]) != ORACLE_RESULT_FIELDS:
            raise NoaaGsodError("private item oracle result schema mismatch")
        if payload_hash(outputs[ORACLE_IDS[0]]) != item.get("gold_commitment"):
            raise NoaaGsodError("private item gold commitment mismatch")
    if counts != PARTITION_COUNTS:
        raise NoaaGsodError("private pack partition layout mismatch")
    return pack


def build_public_receipt(
    private_pack: Mapping[str, Any],
    *,
    metadata_url: str,
    index_url: str,
    network_calls: int,
) -> dict[str, Any]:
    commitments: dict[str, list[str]] = {key: [] for key in PARTITION_COUNTS}
    for item in private_pack["items"]:
        commitments[str(item["partition"])].append(str(item["item_commitment"]))
    body: dict[str, Any] = {
        "acquisition_statistics": dict(private_pack["acquisition_statistics"]),
        "candidate_imports": 0,
        "content_boundary": {
            "git_persisted": [
                "acquisition implementation",
                "closed task contract and schema verifier",
                "official source URLs and source-byte commitments",
                "partition counts and opaque item commitments",
                "focused synthetic tests",
            ],
            "local_artifacts_only": [
                "station metadata and year index bytes",
                "downloaded candidate station CSV bytes",
                "private station-to-item assignments",
                "private pack",
                "all raw item CSVs",
                "all oracle outputs and consensus gold",
            ],
            "sealed_gold_git_persisted": False,
            "sealed_raw_task_git_persisted": False,
            "sealed_station_identity_git_persisted": False,
        },
        "item_commitments_by_partition": commitments,
        "model_calls": 0,
        "network_calls": network_calls,
        "online_judge_calls": 0,
        "oracle_ids": list(ORACLE_IDS),
        "pack_hash": str(private_pack["pack_hash"]),
        "partition_counts": dict(PARTITION_COUNTS),
        "private_pack_git_persisted": False,
        "public_receipt_version": PUBLIC_RECEIPT_VERSION,
        "resampling_used": False,
        "schema_set_hash": SCHEMA_SET_HASH,
        "scoring_calls": 0,
        "selection_policy_hash": payload_hash(private_pack["selection_policy"]),
        "source_commitments": dict(private_pack["source_commitments"]),
        "source_urls": {
            "station_metadata": metadata_url,
            "year_index": index_url,
        },
        "study_id": STUDY_ID,
        "task_contract_hash": payload_hash(TASK_CONTRACT),
        "typed_operator_formed": False,
        "year": 2020,
    }
    receipt = with_self_hash(body, "receipt_hash")
    verify_public_receipt(receipt)
    return receipt


def verify_public_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(payload)
    verify_self_hash(receipt, "receipt_hash")
    if set(receipt) != PUBLIC_RECEIPT_FIELDS:
        raise NoaaGsodError("public receipt schema mismatch")
    if (
        receipt.get("public_receipt_version") != PUBLIC_RECEIPT_VERSION
        or receipt.get("study_id") != STUDY_ID
    ):
        raise NoaaGsodError("public receipt identity mismatch")
    if receipt.get("schema_set_hash") != SCHEMA_SET_HASH:
        raise NoaaGsodError("public receipt schema set hash mismatch")
    if receipt.get("partition_counts") != PARTITION_COUNTS:
        raise NoaaGsodError("public receipt partition counts mismatch")
    commitments = receipt.get("item_commitments_by_partition")
    if not isinstance(commitments, dict) or set(commitments) != set(PARTITION_COUNTS):
        raise NoaaGsodError("public receipt item commitment schema mismatch")
    for partition, expected in PARTITION_COUNTS.items():
        values = commitments.get(partition)
        if not isinstance(values, list) or len(values) != expected:
            raise NoaaGsodError("public receipt item commitment count mismatch")
        if any(not isinstance(value, str) or len(value) != 64 for value in values):
            raise NoaaGsodError("public receipt contains a malformed item commitment")
    boundary = receipt.get("content_boundary")
    if not isinstance(boundary, dict) or any(
        boundary.get(field) is not False
        for field in (
            "sealed_gold_git_persisted",
            "sealed_raw_task_git_persisted",
            "sealed_station_identity_git_persisted",
        )
    ):
        raise NoaaGsodError("public receipt content boundary mismatch")
    return receipt
