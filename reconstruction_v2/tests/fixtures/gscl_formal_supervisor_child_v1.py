"""Stdlib-only source-free child used by formal-supervisor qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError("fixture input is not an object")
    return value


def _write_once(path: Path, value: dict[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write(_canonical(value))
        handle.flush()


def _arm(input_path: Path, output_path: Path) -> None:
    predictor = _read(input_path)
    rows = predictor["rows"]
    predictions = [
        {
            "opaque_item_id": row["opaque_item_id"],
            "disposition": "ANSWER",
            "selected_choice": "first_choice",
            "error_code": None,
        }
        for row in rows
    ]
    _write_once(output_path, {"predictions": predictions})


def _score(
    label_path: Path, prediction_paths: str, output_path: Path
) -> None:
    labels = _read(label_path)
    predictions = [_read(Path(path)) for path in prediction_paths.split(",")]
    item_count = len(labels["rows"])
    if len(predictions) != 4:
        raise ValueError("fixture scorer requires four arms")
    _write_once(
        output_path,
        {
            "status": "SYNTHETIC_AGGREGATE_ONLY",
            "arm_aggregates": {
                pack["arm_id"]: {"item_count": item_count}
                for pack in predictions
            },
            "paired_aggregate_differences": {
                "synthetic_zero": 0
            },
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    arm = subparsers.add_parser("arm")
    arm.add_argument("input", type=Path)
    arm.add_argument("output", type=Path)
    scorer = subparsers.add_parser("score")
    scorer.add_argument("labels", type=Path)
    scorer.add_argument("predictions")
    scorer.add_argument("output", type=Path)
    arguments = parser.parse_args()
    if arguments.mode == "arm":
        _arm(arguments.input, arguments.output)
    else:
        _score(arguments.labels, arguments.predictions, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
