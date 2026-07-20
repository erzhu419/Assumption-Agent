"""Gold-separated acquisition of a fresh BRIGHT reserve measurement cohort."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from assumption_agent.benchmarks import bright_reasoning_retrieval_acquisition_v1 as source


VERSION = "bright_reasoning_retrieval_reserve_acquisition_v1"
VIEW_SCHEMA = f"{VERSION}_block_view"
LABEL_SCHEMA = f"{VERSION}_block_labels"
RESULT_SCHEMA = f"{VERSION}_result"
ATTEMPT_SCHEMA = f"{VERSION}_attempt"
FREEZE_SCHEMA = "bright_reasoning_retrieval_reserve_measurement_v1_implementation_freeze"
BLOCK = "R_search"
COUNT_PER_FAMILY = 15
ITEM_COUNT = 45

DESIGN_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_reserve_measurement_design_v1.json"
)
FREEZE_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_reserve_measurement_implementation_freeze_v1.json"
)
ORIGINAL_RESULT_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_acquisition_result_v1.json"
)
ORIGINAL_PRIVATE_RELATIVE = Path(
    "artifacts/bright_reasoning_retrieval_acquisition_v1/private"
)
ROOT_RELATIVE = Path("artifacts/bright_reasoning_retrieval_reserve_acquisition_v1")
PRIVATE_RELATIVE = ROOT_RELATIVE / "private"
RESULT_RELATIVE = Path(
    "manifests/bright_reasoning_retrieval_reserve_acquisition_result_v1.json"
)

DESIGN_FILE_SHA256 = "e33bc5d432b037336d21fe3c77a11c2d6e3e2b3ce76995da2d7eab34ac96b59b"
DESIGN_SELF_SHA256 = "0a7eb606c31fcd1f634443e22a0d6868f9acee55bd869795f960e5c0b4a5c08d"
ORIGINAL_RESULT_FILE_SHA256 = (
    "f637c369015b0e8d991a1d43373360e0292cf8362ba801347006bd297e7a8e1b"
)
ORIGINAL_RESULT_SHA256 = (
    "5736847df8a9a57f674ee02dc1fbc1fdf08120faa358631358d7d80498092ce7"
)
ORIGINAL_RESERVE_VIEW_FILE_SHA256 = (
    "6019129c48302f5f26440ad285efb4c684c1ca938bd75518f4679142fcc435e8"
)
ORIGINAL_RESERVE_VIEW_SHA256 = (
    "9a8ce1db47632c042ce2308c1153477a8ecc504dd774a52ab889bf48a0aabba1"
)
ORIGINAL_SELECTION_SECRET_SHA256 = (
    "00f64a57001fea3d2922db0e807d92920cc880dd1d5cf214d79e364e7ec8d046"
)


class BrightReserveAcquisitionError(RuntimeError):
    """Reserve acquisition failed closed."""


def _read_canonical(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BrightReserveAcquisitionError(f"{field} is unavailable")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BrightReserveAcquisitionError(f"{field} is invalid") from exc
    if (
        not isinstance(value, dict)
        or source.canonical_json_bytes(value) + b"\n" != raw
    ):
        raise BrightReserveAcquisitionError(f"{field} is not canonical")
    return value


def _verify_self(payload: Mapping[str, Any], field: str, expected: str) -> None:
    try:
        observed = source.verify_self_hash(payload, field)
    except source.BrightAcquisitionError as exc:
        raise BrightReserveAcquisitionError(str(exc)) from exc
    if observed != expected:
        raise BrightReserveAcquisitionError(f"{field} binding drifted")


def _verify_freeze(project_root: Path) -> dict[str, Any]:
    design_path = project_root / DESIGN_RELATIVE
    if source.file_sha256(design_path) != DESIGN_FILE_SHA256:
        raise BrightReserveAcquisitionError("reserve design file drifted")
    design = _read_canonical(design_path, "reserve design")
    _verify_self(design, "self_sha256", DESIGN_SELF_SHA256)
    freeze = _read_canonical(project_root / FREEZE_RELATIVE, "implementation freeze")
    if (
        freeze.get("schema") != FREEZE_SCHEMA
        or freeze.get("design_self_sha256") != DESIGN_SELF_SHA256
    ):
        raise BrightReserveAcquisitionError("implementation freeze identity drifted")
    source.verify_self_hash(freeze, "self_sha256")
    bindings = freeze.get("implementation_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise BrightReserveAcquisitionError("implementation bindings are invalid")
    for row in bindings:
        if not isinstance(row, Mapping) or set(row) != {"relative_path", "sha256"}:
            raise BrightReserveAcquisitionError("implementation binding row drifted")
        path = project_root / str(row["relative_path"])
        if source.file_sha256(path) != row["sha256"]:
            raise BrightReserveAcquisitionError("implementation file drifted")
    return freeze


def select_measurement_rows(
    reserve_rows: Sequence[source.SourceItem],
) -> tuple[source.SourceItem, ...]:
    selected: list[source.SourceItem] = []
    for family in source.FAMILY_ORDER:
        family_rows = [row for row in reserve_rows if row.family == family]
        if len(family_rows) < COUNT_PER_FAMILY:
            raise BrightReserveAcquisitionError("reserve family capacity drifted")
        selected.extend(family_rows[:COUNT_PER_FAMILY])
    if (
        len(selected) != ITEM_COUNT
        or len({row.commitment_sha256 for row in selected}) != ITEM_COUNT
        or Counter(row.family for row in selected)
        != Counter({family: COUNT_PER_FAMILY for family in source.FAMILY_ORDER})
    ):
        raise BrightReserveAcquisitionError("reserve selection drifted")
    return tuple(selected)


def measurement_view(rows: Sequence[source.SourceItem]) -> dict[str, Any]:
    body = {
        "block": BLOCK,
        "excluded_fields": [
            "source_example_id",
            "reasoning",
            "gold_ids_long",
            "gold_ids",
            "gold_answer",
        ],
        "item_count": len(rows),
        "items": [
            {
                "excluded_ids": list(item.excluded_ids),
                "family": item.family,
                "item_commitment_sha256": item.commitment_sha256,
                "ordinal": ordinal,
                "query": item.query,
            }
            for ordinal, item in enumerate(rows)
        ],
        "schema": VIEW_SCHEMA,
    }
    return source.self_hashed(body, "pack_sha256")


def measurement_labels(rows: Sequence[source.SourceItem]) -> dict[str, Any]:
    body = {
        "block": BLOCK,
        "item_count": len(rows),
        "items": [
            {
                "gold_ids": list(item.gold_ids),
                "item_commitment_sha256": item.commitment_sha256,
                "ordinal": ordinal,
            }
            for ordinal, item in enumerate(rows)
        ],
        "schema": LABEL_SCHEMA,
    }
    return source.self_hashed(body, "pack_sha256")


def run(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    freeze = _verify_freeze(project_root)
    result_path = project_root / RESULT_RELATIVE
    root = project_root / ROOT_RELATIVE
    if result_path.exists() or root.exists():
        raise BrightReserveAcquisitionError("reserve acquisition is one-shot")
    root.mkdir(mode=0o700)
    private = root / "private"
    private.mkdir(mode=0o700)
    marker = source.self_hashed(
        {
            "design_self_sha256": DESIGN_SELF_SHA256,
            "implementation_freeze_self_sha256": freeze["self_sha256"],
            "schema": ATTEMPT_SCHEMA,
        },
        "attempt_sha256",
    )
    source._write_json(root / "attempt.marker", marker)

    original_path = project_root / ORIGINAL_RESULT_RELATIVE
    if source.file_sha256(original_path) != ORIGINAL_RESULT_FILE_SHA256:
        raise BrightReserveAcquisitionError("original acquisition result file drifted")
    original = _read_canonical(original_path, "original acquisition result")
    _verify_self(original, "result_sha256", ORIGINAL_RESULT_SHA256)
    secret_path = project_root / ORIGINAL_PRIVATE_RELATIVE / "selection.secret.bin"
    secret = secret_path.read_bytes()
    if hashlib.sha256(secret).hexdigest() != ORIGINAL_SELECTION_SECRET_SHA256:
        raise BrightReserveAcquisitionError("selection secret binding drifted")
    items_by_family = source._source_items(project_root)
    assignments = source.assign_blocks(items_by_family, secret)
    reserve_rows = assignments["RESERVE"]
    reconstructed = source.block_view("RESERVE", reserve_rows)
    original_view_path = project_root / ORIGINAL_PRIVATE_RELATIVE / "RESERVE.view.json"
    if (
        source.file_sha256(original_view_path) != ORIGINAL_RESERVE_VIEW_FILE_SHA256
        or reconstructed["pack_sha256"] != ORIGINAL_RESERVE_VIEW_SHA256
        or original_view_path.read_bytes()
        != source.canonical_json_bytes(reconstructed) + b"\n"
    ):
        raise BrightReserveAcquisitionError("original RESERVE reconstruction drifted")

    selected = select_measurement_rows(reserve_rows)
    view = measurement_view(selected)
    labels = measurement_labels(selected)
    view_path = private / f"{BLOCK}.view.json"
    label_path = private / f"{BLOCK}.labels.json"
    source._write_json(view_path, view)
    source._write_json(label_path, labels)
    body = {
        "claim_boundary": {
            "document_content_reasoning_gold_answer_or_gold_ids_long_read": false,
            "model_retrieval_or_score_count": 0,
            "network_call_count": 0,
            "selection_used_gold_or_outcome": false,
        },
        "cohort": {
            "family_counts": dict(sorted(Counter(row.family for row in selected).items())),
            "item_count": len(selected),
            "label_pack_file_sha256": source.file_sha256(label_path),
            "label_pack_sha256": labels["pack_sha256"],
            "view_pack_file_sha256": source.file_sha256(view_path),
            "view_pack_sha256": view["pack_sha256"],
        },
        "formal_binding": {
            "attempt_marker_file_sha256": source.file_sha256(root / "attempt.marker"),
            "design_self_sha256": DESIGN_SELF_SHA256,
            "implementation_freeze_self_sha256": freeze["self_sha256"],
            "original_acquisition_result_sha256": ORIGINAL_RESULT_SHA256,
            "original_RESERVE_view_pack_sha256": ORIGINAL_RESERVE_VIEW_SHA256,
            "selection_secret_sha256": ORIGINAL_SELECTION_SECRET_SHA256,
        },
        "schema": RESULT_SCHEMA,
        "status": "fresh_RESERVE_R_search_acquired_labels_sealed",
    }
    result = source.self_hashed(body, "result_sha256")
    source._write_json(result_path, result, mode=0o644)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run(arguments.project_root)
    print(
        json.dumps(
            {"result_sha256": result["result_sha256"], "status": result["status"]},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
