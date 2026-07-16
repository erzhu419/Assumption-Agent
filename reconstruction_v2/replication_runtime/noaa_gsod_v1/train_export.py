from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

from . import oracle_sqlite, oracle_stdlib
from .contract import (
    ORACLE_IDS,
    STUDY_ID,
    TASK_CONTRACT,
    NoaaGsodError,
    canonical_json_bytes,
    payload_hash,
    sha256_file,
    verify_self_hash,
    with_self_hash,
)
from .pack import read_json, verify_private_pack, write_json
from .schemas import ORACLE_RESULT_FIELDS
from .train_schemas import (
    TRAIN_PREPARATION_RECEIPT_FIELDS,
    TRAIN_SCHEMA_SET_HASH,
    TRAIN_VIEW_FIELDS,
    TRAIN_VIEW_ITEM_FIELDS,
)


TRAIN_VIEW_VERSION = "noaa_gsod_anonymous_train_view_v1"
TRAIN_PREPARATION_RECEIPT_VERSION = "noaa_gsod_train_preparation_receipt_v1"
TRAIN_ITEM_COUNT = 12
ANONYMOUS_COLUMNS = ("STATION", "DATE", "PRCP")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^TRAIN_STATION_[0-9]{2}$")


def _anonymous_csv(
    source: Path,
    destination: Path,
    *,
    source_station_id: str,
    anonymized_station_token: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with source.open("r", encoding="utf-8-sig", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None or not set(ANONYMOUS_COLUMNS).issubset(
            reader.fieldnames
        ):
            raise NoaaGsodError("TRAIN source lacks required columns")
        with temporary.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(
                output_handle,
                fieldnames=list(ANONYMOUS_COLUMNS),
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in reader:
                if None in row:
                    raise NoaaGsodError("TRAIN source row has excess fields")
                if str(row["STATION"] or "").strip() != source_station_id:
                    raise NoaaGsodError("TRAIN source contains another station identity")
                writer.writerow(
                    {
                        "STATION": anonymized_station_token,
                        "DATE": str(row["DATE"] or "").strip(),
                        "PRCP": str(row["PRCP"] or "").strip(),
                    }
                )
    temporary.replace(destination)


def _resolve_beneath(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if root != path and root not in path.parents:
        raise NoaaGsodError("TRAIN view path escapes its root")
    return path


def _prepare_output_root(*, output_root: Path, source_root: Path) -> None:
    """Create a clean TRAIN root without deleting an arbitrary directory."""

    if (
        output_root == source_root
        or output_root in source_root.parents
        or source_root in output_root.parents
        or output_root == Path(output_root.anchor)
    ):
        raise NoaaGsodError("TRAIN view and private source roots must be disjoint")
    if output_root.exists():
        manifest_path = output_root / "train_view.json"
        if not manifest_path.is_file():
            raise NoaaGsodError("refusing to replace an undeclared TRAIN view root")
        verify_train_view(read_json(manifest_path), train_view_root=output_root)
        shutil.rmtree(output_root)


def export_train_view(
    *,
    private_pack_path: str | Path,
    private_root: str | Path,
    train_view_root: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    source_root = Path(private_root).resolve(strict=True)
    private_pack = verify_private_pack(
        read_json(private_pack_path), private_root=source_root
    )
    train_items = sorted(
        (item for item in private_pack["items"] if item["partition"] == "train"),
        key=lambda item: int(item["ordinal"]),
    )
    if len(train_items) != TRAIN_ITEM_COUNT or [
        int(item["ordinal"]) for item in train_items
    ] != list(range(TRAIN_ITEM_COUNT)):
        raise NoaaGsodError("private pack has an invalid TRAIN layout")

    output_root = Path(train_view_root).resolve()
    receipt = Path(receipt_path).resolve()
    if receipt == output_root or output_root in receipt.parents:
        raise NoaaGsodError("TRAIN preparation receipt must be outside the TRAIN view")
    _prepare_output_root(output_root=output_root, source_root=source_root)
    exported: list[dict[str, Any]] = []
    for ordinal, source_item in enumerate(train_items):
        token = f"TRAIN_STATION_{ordinal:02d}"
        train_item_id = f"noaa_gsod_train_export_{ordinal:02d}"
        relative_path = f"inputs/{train_item_id}.csv"
        destination = output_root / relative_path
        source_path = _resolve_beneath(
            source_root, str(source_item["raw_csv_relative_path"])
        )
        _anonymous_csv(
            source_path,
            destination,
            source_station_id=str(source_item["station_id"]),
            anonymized_station_token=token,
        )
        consensus = source_item["oracle_outputs"][ORACLE_IDS[0]]
        item_body: dict[str, Any] = {
            "anonymized_station_token": token,
            "input_columns": list(ANONYMOUS_COLUMNS),
            "input_relative_path": relative_path,
            "input_sha256": sha256_file(destination),
            "oracle_consensus": consensus,
            "oracle_consensus_hash": payload_hash(consensus),
            "ordinal": ordinal,
            "train_item_id": train_item_id,
        }
        item_body["train_item_hash"] = payload_hash(item_body)
        exported.append(item_body)

    view_body: dict[str, Any] = {
        "candidate_imports": 0,
        "items": exported,
        "model_calls": 0,
        "network_calls": 0,
        "online_judge_calls": 0,
        "oracle_consensus_ids": list(ORACLE_IDS),
        "partition": "train",
        "role": "candidate_formation_input_only",
        "scoring_calls": 0,
        "source_private_pack_hash": private_pack["pack_hash"],
        "study_id": STUDY_ID,
        "task_contract": TASK_CONTRACT,
        "task_contract_hash": payload_hash(TASK_CONTRACT),
        "train_item_count": TRAIN_ITEM_COUNT,
        "train_view_version": TRAIN_VIEW_VERSION,
        "typed_operator_formed": False,
    }
    train_view = with_self_hash(view_body, "train_view_hash")
    write_json(output_root / "train_view.json", train_view)
    verify_train_view(train_view, train_view_root=output_root)

    # Strictly prove that no private non-TRAIN mapping token entered the view.
    exported_bytes = canonical_json_bytes(train_view) + b"".join(
        (output_root / item["input_relative_path"]).read_bytes()
        for item in train_view["items"]
    )
    for source_item in private_pack["items"]:
        if str(source_item["station_id"]).encode("utf-8") in exported_bytes:
            raise NoaaGsodError("private station identity entered anonymous TRAIN view")
    train_gold_commitments = {
        str(item["gold_commitment"])
        for item in private_pack["items"]
        if item["partition"] == "train"
    }
    for source_item in private_pack["items"]:
        if source_item["partition"] == "train":
            continue
        forbidden = [
            str(source_item["station_id"]),
            str(source_item["raw_csv_sha256"]),
            str(source_item["item_commitment"]),
            str(source_item["task_payload_commitment"]),
        ]
        # A non-TRAIN result may equal a legitimate TRAIN result.  Equality of
        # the value is not an exported mapping, so only unique non-TRAIN gold
        # commitments are forbidden here.
        if str(source_item["gold_commitment"]) not in train_gold_commitments:
            forbidden.append(str(source_item["gold_commitment"]))
        if any(value.encode("utf-8") in exported_bytes for value in forbidden):
            raise NoaaGsodError("non-TRAIN private mapping entered TRAIN view")

    receipt_body: dict[str, Any] = {
        "candidate_imports": 0,
        "content_boundary": {
            "development_commitment_mapping_exported": False,
            "development_gold_exported": False,
            "development_raw_exported": False,
            "development_station_identity_exported": False,
            "sealed_commitment_mapping_exported": False,
            "sealed_gold_exported": False,
            "sealed_raw_exported": False,
            "sealed_station_identity_exported": False,
            "train_inputs_anonymized": True,
            "train_view_local_git_ignored_only": True,
        },
        "model_calls": 0,
        "network_calls": 0,
        "online_judge_calls": 0,
        "oracle_verification_calls": TRAIN_ITEM_COUNT * len(ORACLE_IDS),
        "preparation_receipt_version": TRAIN_PREPARATION_RECEIPT_VERSION,
        "schema_set_hash": TRAIN_SCHEMA_SET_HASH,
        "scoring_calls": 0,
        "source_private_pack_hash": private_pack["pack_hash"],
        "study_id": STUDY_ID,
        "task_contract_hash": payload_hash(TASK_CONTRACT),
        "train_consensus_set_hash": payload_hash(
            [item["oracle_consensus_hash"] for item in exported]
        ),
        "train_input_set_hash": payload_hash(
            [item["input_sha256"] for item in exported]
        ),
        "train_item_count": TRAIN_ITEM_COUNT,
        "train_view_git_persisted": False,
        "train_view_hash": train_view["train_view_hash"],
        "typed_operator_formed": False,
    }
    receipt = with_self_hash(receipt_body, "preparation_receipt_hash")
    verify_train_preparation_receipt(receipt)
    write_json(receipt_path, receipt)
    return {
        "preparation_receipt_hash": receipt["preparation_receipt_hash"],
        "train_consensus_set_hash": receipt["train_consensus_set_hash"],
        "train_input_set_hash": receipt["train_input_set_hash"],
        "train_item_count": TRAIN_ITEM_COUNT,
        "train_view_hash": train_view["train_view_hash"],
    }


def verify_train_view(
    payload: Mapping[str, Any], *, train_view_root: str | Path
) -> dict[str, Any]:
    view = dict(payload)
    verify_self_hash(view, "train_view_hash")
    if set(view) != TRAIN_VIEW_FIELDS:
        raise NoaaGsodError("TRAIN view schema mismatch")
    if (
        view.get("train_view_version") != TRAIN_VIEW_VERSION
        or view.get("study_id") != STUDY_ID
        or view.get("partition") != "train"
        or view.get("role") != "candidate_formation_input_only"
        or view.get("task_contract") != TASK_CONTRACT
        or view.get("task_contract_hash") != payload_hash(TASK_CONTRACT)
        or view.get("train_item_count") != TRAIN_ITEM_COUNT
        or tuple(view.get("oracle_consensus_ids", ())) != ORACLE_IDS
    ):
        raise NoaaGsodError("TRAIN view frozen identity mismatch")
    if any(view.get(field) != 0 for field in (
        "candidate_imports", "model_calls", "network_calls", "online_judge_calls", "scoring_calls"
    )) or view.get("typed_operator_formed") is not False:
        raise NoaaGsodError("TRAIN view preparation crossed a role boundary")
    source_hash = view.get("source_private_pack_hash")
    if not isinstance(source_hash, str) or _SHA256.fullmatch(source_hash) is None:
        raise NoaaGsodError("TRAIN view source pack hash malformed")
    items = view.get("items")
    if not isinstance(items, list) or len(items) != TRAIN_ITEM_COUNT:
        raise NoaaGsodError("TRAIN view item count mismatch")

    root = Path(train_view_root).resolve(strict=True)
    seen_paths: set[str] = set()
    for expected_ordinal, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != TRAIN_VIEW_ITEM_FIELDS:
            raise NoaaGsodError("TRAIN view item schema mismatch")
        body = dict(item)
        declared = body.pop("train_item_hash", None)
        if declared != payload_hash(body):
            raise NoaaGsodError("TRAIN view item hash mismatch")
        token = item.get("anonymized_station_token")
        if (
            item.get("ordinal") != expected_ordinal
            or item.get("train_item_id") != f"noaa_gsod_train_export_{expected_ordinal:02d}"
            or token != f"TRAIN_STATION_{expected_ordinal:02d}"
            or not isinstance(token, str)
            or _TOKEN.fullmatch(token) is None
            or tuple(item.get("input_columns", ())) != ANONYMOUS_COLUMNS
        ):
            raise NoaaGsodError("TRAIN view anonymous item identity mismatch")
        relative = str(item.get("input_relative_path", ""))
        if relative in seen_paths:
            raise NoaaGsodError("TRAIN view repeats an input path")
        seen_paths.add(relative)
        path = _resolve_beneath(root, relative)
        if sha256_file(path) != item.get("input_sha256"):
            raise NoaaGsodError("TRAIN view input hash mismatch")
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != ANONYMOUS_COLUMNS:
                raise NoaaGsodError("TRAIN view contains a non-minimal input schema")
            if any(
                None in row or str(row["STATION"] or "") != token
                for row in reader
            ):
                raise NoaaGsodError("TRAIN view input is not fully anonymized")
        consensus = item.get("oracle_consensus")
        if not isinstance(consensus, dict) or set(consensus) != ORACLE_RESULT_FIELDS:
            raise NoaaGsodError("TRAIN oracle consensus schema mismatch")
        if payload_hash(consensus) != item.get("oracle_consensus_hash"):
            raise NoaaGsodError("TRAIN oracle consensus hash mismatch")
        first = oracle_stdlib.evaluate(path)
        second = oracle_sqlite.evaluate(path)
        if first != second or first != consensus:
            raise NoaaGsodError("TRAIN anonymous input and consensus disagree")
    expected_files = {root / "train_view.json"} | {
        (root / relative).resolve() for relative in seen_paths
    }
    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise NoaaGsodError("TRAIN view contains an undeclared file")
    return view


def verify_train_preparation_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(payload)
    verify_self_hash(receipt, "preparation_receipt_hash")
    if set(receipt) != TRAIN_PREPARATION_RECEIPT_FIELDS:
        raise NoaaGsodError("TRAIN preparation receipt schema mismatch")
    if (
        receipt.get("preparation_receipt_version")
        != TRAIN_PREPARATION_RECEIPT_VERSION
        or receipt.get("study_id") != STUDY_ID
        or receipt.get("schema_set_hash") != TRAIN_SCHEMA_SET_HASH
        or receipt.get("task_contract_hash") != payload_hash(TASK_CONTRACT)
        or receipt.get("train_item_count") != TRAIN_ITEM_COUNT
        or receipt.get("train_view_git_persisted") is not False
    ):
        raise NoaaGsodError("TRAIN preparation receipt identity mismatch")
    if any(receipt.get(field) != 0 for field in (
        "candidate_imports", "model_calls", "network_calls", "online_judge_calls", "scoring_calls"
    )) or receipt.get("typed_operator_formed") is not False:
        raise NoaaGsodError("TRAIN preparation receipt crossed a role boundary")
    if receipt.get("oracle_verification_calls") != TRAIN_ITEM_COUNT * len(ORACLE_IDS):
        raise NoaaGsodError("TRAIN preparation oracle call count mismatch")
    boundary = receipt.get("content_boundary")
    if not isinstance(boundary, dict):
        raise NoaaGsodError("TRAIN preparation content boundary missing")
    if any(
        boundary.get(field) is not False
        for field in (
            "development_commitment_mapping_exported",
            "development_gold_exported",
            "development_raw_exported",
            "development_station_identity_exported",
            "sealed_commitment_mapping_exported",
            "sealed_gold_exported",
            "sealed_raw_exported",
            "sealed_station_identity_exported",
        )
    ) or any(
        boundary.get(field) is not True
        for field in ("train_inputs_anonymized", "train_view_local_git_ignored_only")
    ):
        raise NoaaGsodError("TRAIN preparation content boundary mismatch")
    for field in (
        "source_private_pack_hash",
        "train_consensus_set_hash",
        "train_input_set_hash",
        "train_view_hash",
    ):
        value = receipt.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise NoaaGsodError(f"TRAIN preparation {field} malformed")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the anonymous TRAIN-only NOAA GSOD formation view."
    )
    parser.add_argument("--private-pack", required=True)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--train-view-root", required=True)
    parser.add_argument("--receipt", required=True)
    arguments = parser.parse_args(argv)
    summary = export_train_view(
        private_pack_path=arguments.private_pack,
        private_root=arguments.private_root,
        train_view_root=arguments.train_view_root,
        receipt_path=arguments.receipt,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
