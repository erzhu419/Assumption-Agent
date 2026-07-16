from __future__ import annotations

"""One-way acquisition-custodian export for NOAA development inputs.

This module is the only development preparation layer that may open the
monolithic acquisition pack.  Its output contains six anonymous, minimal CSV
inputs and a gold-free private index.  A later development controller can
therefore operate without access to the acquisition pack or sealed material.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from .contract import (
    PARTITION_COUNTS,
    STUDY_ID,
    TASK_CONTRACT,
    NoaaGsodError,
    canonical_json_bytes,
    payload_hash,
    sha256_file,
    verify_self_hash,
    with_self_hash,
)
from .pack import (
    read_json,
    verify_private_pack,
    verify_public_receipt,
    write_json,
)


DEVELOPMENT_SOURCE_INDEX_VERSION = "noaa_gsod_development_source_index_v1"
DEVELOPMENT_SOURCE_RECEIPT_VERSION = "noaa_gsod_development_source_receipt_v1"
DEVELOPMENT_ITEM_COUNT = PARTITION_COUNTS["development"]
PRIVATE_INDEX_NAME = "development_source.private.json"
ANONYMOUS_COLUMNS = ("STATION", "DATE", "PRCP")

DEVELOPMENT_SOURCE_ITEM_FIELDS = frozenset(
    {
        "anonymous_item_id",
        "anonymized_station_token",
        "input_columns",
        "input_relative_path",
        "input_sha256",
        "ordinal",
        "source_item_commitment",
    }
)
DEVELOPMENT_SOURCE_INDEX_FIELDS = frozenset(
    {
        "content_boundary",
        "development_commitment_set_hash",
        "development_item_count",
        "input_set_hash",
        "items",
        "partition",
        "private_index_hash",
        "private_index_version",
        "role",
        "source_acquisition_receipt_hash",
        "source_private_pack_hash",
        "study_id",
    }
)
DEVELOPMENT_SOURCE_RECEIPT_FIELDS = frozenset(
    {
        "binding_hashes",
        "call_ledger",
        "content_boundary",
        "custody_boundary",
        "development_input_materialized",
        "development_source_receipt_hash",
        "development_source_receipt_version",
        "partition_counts",
        "study_id",
    }
)
BINDING_HASH_FIELDS = frozenset(
    {
        "acquisition_receipt_file_sha256",
        "acquisition_receipt_hash",
        "development_commitment_set_hash",
        "development_source_schema_set_hash",
        "private_index_file_sha256",
        "private_index_hash",
        "source_private_pack_hash",
        "source_view_input_set_hash",
        "task_contract_hash",
    }
)
CALL_LEDGER_FIELDS = frozenset(
    {
        "model_calls",
        "network_calls",
        "offline_oracle_calls",
        "online_judge_calls",
        "operator_calls",
        "scoring_calls",
    }
)
INDEX_CONTENT_BOUNDARY = {
    "gold_commitments_included": False,
    "oracle_outputs_included": False,
    "sealed_item_mapping_included": False,
    "source_raw_paths_included": False,
    "source_station_ids_included": False,
}
PUBLIC_CONTENT_BOUNDARY = {
    "development_raw_input_persisted_publicly": False,
    "development_station_identity_persisted_publicly": False,
    "gold_commitments_exported": False,
    "oracle_outputs_exported": False,
    "private_index_persisted_publicly": False,
    "sealed_item_mapping_exported": False,
    "sealed_task_content_exported": False,
    "source_raw_paths_exported": False,
}
CUSTODY_BOUNDARY = {
    "acquisition_custodian_development_identity_accessed": True,
    "acquisition_custodian_monolithic_private_pack_accessed": True,
    "development_controller_gold_or_oracle_exposed": False,
    "development_controller_monolithic_private_pack_exposed": False,
    "development_controller_sealed_mapping_exposed": False,
    "sealed_runtime_any_item_material_exposed": False,
    "source_view_contains_development_inputs_only": True,
}
ZERO_CALL_LEDGER = {field: 0 for field in sorted(CALL_LEDGER_FIELDS)}

DEVELOPMENT_SOURCE_SCHEMA_SET: dict[str, Any] = {
    "schema_version": "noaa_gsod_development_source_schema_set_v1",
    "private_index_item": {
        "additionalProperties": False,
        "required": sorted(DEVELOPMENT_SOURCE_ITEM_FIELDS),
        "type": "object",
    },
    "private_index": {
        "additionalProperties": False,
        "required": sorted(DEVELOPMENT_SOURCE_INDEX_FIELDS),
        "type": "object",
    },
    "public_receipt": {
        "additionalProperties": False,
        "required": sorted(DEVELOPMENT_SOURCE_RECEIPT_FIELDS),
        "type": "object",
    },
    "binding_hashes": {
        "additionalProperties": False,
        "required": sorted(BINDING_HASH_FIELDS),
        "type": "object",
    },
    "call_ledger": {
        "additionalProperties": False,
        "required": sorted(CALL_LEDGER_FIELDS),
        "type": "object",
    },
}
DEVELOPMENT_SOURCE_SCHEMA_SET_HASH = payload_hash(DEVELOPMENT_SOURCE_SCHEMA_SET)


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NoaaGsodError(f"{label} is not a SHA-256 hash")
    return value


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _resolve_beneath(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if root not in path.parents:
        raise NoaaGsodError("custodian source path escapes private pack root")
    return path


def _require_ignored_when_inside_repository(path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    try:
        path.relative_to(repository)
    except ValueError:
        return
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", str(path)],
        cwd=repository,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise NoaaGsodError("development source-view root is not git-ignored")


def _write_json_no_clobber(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise NoaaGsodError("public source receipt already exists; no-clobber required")
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.tmp-",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise NoaaGsodError(
                "public source receipt already exists; no-clobber required"
            ) from exc
        temporary_path.unlink()
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_anonymous_input(
    source: Path,
    destination: Path,
    *,
    source_station_id: str,
    anonymous_token: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    row_count = 0
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as input_handle:
            reader = csv.DictReader(input_handle)
            if reader.fieldnames is None or not set(ANONYMOUS_COLUMNS).issubset(
                reader.fieldnames
            ):
                raise NoaaGsodError("development source lacks required columns")
            with temporary.open("w", encoding="utf-8", newline="") as output_handle:
                writer = csv.DictWriter(
                    output_handle,
                    fieldnames=list(ANONYMOUS_COLUMNS),
                    lineterminator="\n",
                )
                writer.writeheader()
                for row in reader:
                    if None in row:
                        raise NoaaGsodError(
                            "development source row has excess fields"
                        )
                    if str(row["STATION"] or "").strip() != source_station_id:
                        raise NoaaGsodError(
                            "development source contains another station"
                        )
                    writer.writerow(
                        {
                            "STATION": anonymous_token,
                            "DATE": str(row["DATE"] or "").strip(),
                            "PRCP": str(row["PRCP"] or "").strip(),
                        }
                    )
                    row_count += 1
        if row_count == 0:
            raise NoaaGsodError("development source is empty")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _input_set_hash(items: Sequence[Mapping[str, Any]]) -> str:
    return payload_hash(
        [
            {
                "anonymous_item_id": item["anonymous_item_id"],
                "input_sha256": item["input_sha256"],
            }
            for item in items
        ]
    )


def _commitment_set_hash(items: Sequence[Mapping[str, Any]]) -> str:
    return payload_hash([item["source_item_commitment"] for item in items])


def verify_development_source_index(
    payload: Mapping[str, Any],
    *,
    source_view_root: str | Path,
    expected_development_commitments: Sequence[str] | None = None,
) -> dict[str, Any]:
    index = dict(payload)
    verify_self_hash(index, "private_index_hash")
    if set(index) != DEVELOPMENT_SOURCE_INDEX_FIELDS:
        raise NoaaGsodError("development source private index schema mismatch")
    if (
        index.get("private_index_version") != DEVELOPMENT_SOURCE_INDEX_VERSION
        or index.get("study_id") != STUDY_ID
        or index.get("partition") != "development"
        or index.get("role") != "gold_free_anonymous_generation_source"
        or index.get("development_item_count") != DEVELOPMENT_ITEM_COUNT
        or index.get("content_boundary") != INDEX_CONTENT_BOUNDARY
    ):
        raise NoaaGsodError("development source private index identity mismatch")
    _require_sha256(index.get("source_acquisition_receipt_hash"), "acquisition receipt")
    _require_sha256(index.get("source_private_pack_hash"), "private pack")
    items = index.get("items")
    if not isinstance(items, list) or len(items) != DEVELOPMENT_ITEM_COUNT:
        raise NoaaGsodError("development source private index item count mismatch")
    root_request = Path(source_view_root)
    if root_request.is_symlink():
        raise NoaaGsodError("development source-view root is a symbolic link")
    root = root_request.resolve(strict=True)
    if not root.is_dir():
        raise NoaaGsodError("development source-view root is not a directory")
    observed_commitments: list[str] = []
    expected_entries = {PRIVATE_INDEX_NAME, "items"}
    for ordinal, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != DEVELOPMENT_SOURCE_ITEM_FIELDS:
            raise NoaaGsodError("development source private item schema mismatch")
        anonymous_item_id = f"development_item_{ordinal:02d}"
        relative_path = f"items/{anonymous_item_id}.csv"
        token = f"DEVELOPMENT_STATION_{ordinal:02d}"
        if (
            item.get("ordinal") != ordinal
            or item.get("anonymous_item_id") != anonymous_item_id
            or item.get("anonymized_station_token") != token
            or tuple(item.get("input_columns", ())) != ANONYMOUS_COLUMNS
            or item.get("input_relative_path") != relative_path
        ):
            raise NoaaGsodError("development source anonymous item identity mismatch")
        commitment = _require_sha256(
            item.get("source_item_commitment"), "development item commitment"
        )
        input_sha256 = _require_sha256(item.get("input_sha256"), "development input")
        path = (root / relative_path).resolve(strict=True)
        if root not in path.parents or path.is_symlink() or not path.is_file():
            raise NoaaGsodError("development source input escapes source-view root")
        if sha256_file(path) != input_sha256:
            raise NoaaGsodError("development source input hash mismatch")
        row_count = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != ANONYMOUS_COLUMNS:
                raise NoaaGsodError("development source input is not minimal")
            for row in reader:
                if None in row or row.get("STATION") != token:
                    raise NoaaGsodError("development source input is not anonymous")
                row_count += 1
        if row_count == 0:
            raise NoaaGsodError("development source input is empty")
        observed_commitments.append(commitment)
        expected_entries.add(relative_path)
    if len(set(observed_commitments)) != DEVELOPMENT_ITEM_COUNT:
        raise NoaaGsodError("development source commitments are not unique")
    if (
        expected_development_commitments is not None
        and observed_commitments != list(expected_development_commitments)
    ):
        raise NoaaGsodError("development source commitments differ from acquisition")
    if index.get("input_set_hash") != _input_set_hash(items):
        raise NoaaGsodError("development source input set hash mismatch")
    if index.get("development_commitment_set_hash") != _commitment_set_hash(items):
        raise NoaaGsodError("development source commitment set hash mismatch")
    actual_entries: set[str] = set()
    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise NoaaGsodError("development source-view contains a symbolic link")
        relative = entry.relative_to(root).as_posix()
        if entry.is_dir() or entry.is_file():
            actual_entries.add(relative)
    if actual_entries != expected_entries:
        raise NoaaGsodError("development source-view contains unexpected material")
    return index


def verify_public_development_source_receipt(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = dict(payload)
    verify_self_hash(receipt, "development_source_receipt_hash")
    if set(receipt) != DEVELOPMENT_SOURCE_RECEIPT_FIELDS:
        raise NoaaGsodError("public development source receipt schema mismatch")
    if (
        receipt.get("development_source_receipt_version")
        != DEVELOPMENT_SOURCE_RECEIPT_VERSION
        or receipt.get("study_id") != STUDY_ID
        or receipt.get("partition_counts") != {"development": DEVELOPMENT_ITEM_COUNT}
        or receipt.get("development_input_materialized") is not True
        or receipt.get("call_ledger") != ZERO_CALL_LEDGER
        or receipt.get("content_boundary") != PUBLIC_CONTENT_BOUNDARY
        or receipt.get("custody_boundary") != CUSTODY_BOUNDARY
    ):
        raise NoaaGsodError("public development source receipt identity mismatch")
    bindings = receipt.get("binding_hashes")
    if not isinstance(bindings, dict) or set(bindings) != BINDING_HASH_FIELDS:
        raise NoaaGsodError("public development source binding schema mismatch")
    for field, value in bindings.items():
        _require_sha256(value, field)
    if (
        bindings.get("development_source_schema_set_hash")
        != DEVELOPMENT_SOURCE_SCHEMA_SET_HASH
        or bindings.get("task_contract_hash") != payload_hash(TASK_CONTRACT)
    ):
        raise NoaaGsodError("public development source fixed binding mismatch")
    return receipt


def verify_development_source_bundle(
    receipt_payload: Mapping[str, Any],
    *,
    source_view_root: str | Path,
    acquisition_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    receipt = verify_public_development_source_receipt(receipt_payload)
    root_request = Path(source_view_root)
    if root_request.is_symlink():
        raise NoaaGsodError("development source-view root is a symbolic link")
    root = root_request.resolve(strict=True)
    index_path = root / PRIVATE_INDEX_NAME
    acquisition: dict[str, Any] | None = None
    expected_commitments: Sequence[str] | None = None
    acquisition_file: Path | None = None
    if acquisition_receipt_path is not None:
        acquisition_file = Path(acquisition_receipt_path).resolve(strict=True)
        acquisition = verify_public_receipt(read_json(acquisition_file))
        expected_commitments = acquisition["item_commitments_by_partition"][
            "development"
        ]
    index = verify_development_source_index(
        read_json(index_path),
        source_view_root=root,
        expected_development_commitments=expected_commitments,
    )
    bindings = receipt["binding_hashes"]
    if (
        bindings["private_index_file_sha256"] != sha256_file(index_path)
        or bindings["private_index_hash"] != index["private_index_hash"]
        or bindings["source_private_pack_hash"] != index["source_private_pack_hash"]
        or bindings["source_view_input_set_hash"] != index["input_set_hash"]
        or bindings["development_commitment_set_hash"]
        != index["development_commitment_set_hash"]
    ):
        raise NoaaGsodError("public receipt differs from development source-view")
    if acquisition is not None and acquisition_file is not None:
        if (
            bindings["acquisition_receipt_file_sha256"]
            != sha256_file(acquisition_file)
            or bindings["acquisition_receipt_hash"] != acquisition["receipt_hash"]
            or bindings["acquisition_receipt_hash"]
            != index["source_acquisition_receipt_hash"]
            or bindings["source_private_pack_hash"] != acquisition["pack_hash"]
            or bindings["task_contract_hash"] != acquisition["task_contract_hash"]
        ):
            raise NoaaGsodError("development source-view differs from acquisition")
    return receipt


def export_development_source_view(
    *,
    private_pack_path: str | Path,
    private_pack_root: str | Path,
    acquisition_receipt_path: str | Path,
    source_view_root: str | Path,
    public_receipt_path: str | Path,
) -> dict[str, Any]:
    private_root = Path(private_pack_root).resolve(strict=True)
    private_pack_file = Path(private_pack_path).resolve(strict=True)
    if private_root not in private_pack_file.parents:
        raise NoaaGsodError("private pack file is outside private pack root")
    acquisition_file = Path(acquisition_receipt_path).resolve(strict=True)
    view_request = Path(source_view_root)
    public_request = Path(public_receipt_path)
    if view_request.is_symlink():
        raise NoaaGsodError("development source-view root must not be a symbolic link")
    if public_request.exists() or public_request.is_symlink():
        raise NoaaGsodError("public source receipt already exists; no-clobber required")
    view_root = view_request.resolve()
    public_file = public_request.resolve()
    if _paths_overlap(view_root, private_root):
        raise NoaaGsodError("development source-view overlaps private pack root")
    if _paths_overlap(public_file, private_root) or _paths_overlap(
        public_file, view_root
    ):
        raise NoaaGsodError("public source receipt overlaps a private root")
    _require_ignored_when_inside_repository(view_root)
    if view_root.exists():
        raise NoaaGsodError("development source-view root already exists")

    # This is the single trusted custodian boundary.  No downstream controller
    # should receive either of these in-memory objects or their source paths.
    private_pack = verify_private_pack(
        read_json(private_pack_file),
        private_root=private_root,
    )
    acquisition = verify_public_receipt(read_json(acquisition_file))
    if (
        acquisition.get("study_id") != STUDY_ID
        or acquisition.get("pack_hash") != private_pack.get("pack_hash")
        or acquisition.get("source_commitments")
        != private_pack.get("source_commitments")
        or acquisition.get("partition_counts") != PARTITION_COUNTS
        or acquisition.get("task_contract_hash") != payload_hash(TASK_CONTRACT)
    ):
        raise NoaaGsodError("acquisition receipt does not bind private pack")
    development_items = sorted(
        (
            item
            for item in private_pack["items"]
            if item.get("partition") == "development"
        ),
        key=lambda item: int(item["ordinal"]),
    )
    expected_commitments = acquisition["item_commitments_by_partition"][
        "development"
    ]
    if (
        len(development_items) != DEVELOPMENT_ITEM_COUNT
        or [item.get("ordinal") for item in development_items]
        != list(range(DEVELOPMENT_ITEM_COUNT))
        or [item.get("item_id") for item in development_items]
        != [
            f"noaa_gsod_development_{ordinal:02d}"
            for ordinal in range(DEVELOPMENT_ITEM_COUNT)
        ]
        or [item.get("item_commitment") for item in development_items]
        != expected_commitments
    ):
        raise NoaaGsodError(
            "private development layout differs from public acquisition commitments"
        )

    view_root.mkdir(parents=True, exist_ok=False)
    try:
        exported_items: list[dict[str, Any]] = []
        for ordinal, source_item in enumerate(development_items):
            anonymous_item_id = f"development_item_{ordinal:02d}"
            anonymous_token = f"DEVELOPMENT_STATION_{ordinal:02d}"
            relative_path = f"items/{anonymous_item_id}.csv"
            source_path = _resolve_beneath(
                private_root,
                str(source_item["raw_csv_relative_path"]),
            )
            destination = view_root / relative_path
            _write_anonymous_input(
                source_path,
                destination,
                source_station_id=str(source_item["station_id"]),
                anonymous_token=anonymous_token,
            )
            exported_items.append(
                {
                    "anonymous_item_id": anonymous_item_id,
                    "anonymized_station_token": anonymous_token,
                    "input_columns": list(ANONYMOUS_COLUMNS),
                    "input_relative_path": relative_path,
                    "input_sha256": sha256_file(destination),
                    "ordinal": ordinal,
                    "source_item_commitment": str(source_item["item_commitment"]),
                }
            )
        index_body: dict[str, Any] = {
            "content_boundary": dict(INDEX_CONTENT_BOUNDARY),
            "development_commitment_set_hash": _commitment_set_hash(
                exported_items
            ),
            "development_item_count": DEVELOPMENT_ITEM_COUNT,
            "input_set_hash": _input_set_hash(exported_items),
            "items": exported_items,
            "partition": "development",
            "private_index_version": DEVELOPMENT_SOURCE_INDEX_VERSION,
            "role": "gold_free_anonymous_generation_source",
            "source_acquisition_receipt_hash": acquisition["receipt_hash"],
            "source_private_pack_hash": private_pack["pack_hash"],
            "study_id": STUDY_ID,
        }
        private_index = with_self_hash(index_body, "private_index_hash")
        index_path = view_root / PRIVATE_INDEX_NAME
        write_json(index_path, private_index)
        verify_development_source_index(
            read_json(index_path),
            source_view_root=view_root,
            expected_development_commitments=expected_commitments,
        )

        public_body: dict[str, Any] = {
            "binding_hashes": {
                "acquisition_receipt_file_sha256": sha256_file(acquisition_file),
                "acquisition_receipt_hash": acquisition["receipt_hash"],
                "development_commitment_set_hash": private_index[
                    "development_commitment_set_hash"
                ],
                "development_source_schema_set_hash": (
                    DEVELOPMENT_SOURCE_SCHEMA_SET_HASH
                ),
                "private_index_file_sha256": sha256_file(index_path),
                "private_index_hash": private_index["private_index_hash"],
                "source_private_pack_hash": private_pack["pack_hash"],
                "source_view_input_set_hash": private_index["input_set_hash"],
                "task_contract_hash": payload_hash(TASK_CONTRACT),
            },
            "call_ledger": dict(ZERO_CALL_LEDGER),
            "content_boundary": dict(PUBLIC_CONTENT_BOUNDARY),
            "custody_boundary": dict(CUSTODY_BOUNDARY),
            "development_input_materialized": True,
            "development_source_receipt_version": (
                DEVELOPMENT_SOURCE_RECEIPT_VERSION
            ),
            "partition_counts": {"development": DEVELOPMENT_ITEM_COUNT},
            "study_id": STUDY_ID,
        }
        public_receipt = with_self_hash(
            public_body,
            "development_source_receipt_hash",
        )
        verify_development_source_bundle(
            public_receipt,
            source_view_root=view_root,
            acquisition_receipt_path=acquisition_file,
        )
        _write_json_no_clobber(public_file, public_receipt)
    except Exception:
        shutil.rmtree(view_root, ignore_errors=True)
        raise

    return {
        "development_item_count": DEVELOPMENT_ITEM_COUNT,
        "development_source_receipt_hash": public_receipt[
            "development_source_receipt_hash"
        ],
        "input_set_hash": private_index["input_set_hash"],
        "private_index_hash": private_index["private_index_hash"],
        "source_private_pack_hash": private_pack["pack_hash"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export a gold-free anonymous NOAA development source view through "
            "the acquisition-custodian boundary."
        )
    )
    parser.add_argument("--private-pack", required=True)
    parser.add_argument("--private-pack-root", required=True)
    parser.add_argument("--acquisition-receipt", required=True)
    parser.add_argument("--source-view-root", required=True)
    parser.add_argument("--public-receipt", required=True)
    arguments = parser.parse_args(argv)
    summary = export_development_source_view(
        private_pack_path=arguments.private_pack,
        private_pack_root=arguments.private_pack_root,
        acquisition_receipt_path=arguments.acquisition_receipt,
        source_view_root=arguments.source_view_root,
        public_receipt_path=arguments.public_receipt,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
