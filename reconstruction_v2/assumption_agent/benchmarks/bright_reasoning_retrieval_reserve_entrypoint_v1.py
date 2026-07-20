"""Frozen entrypoint for the BRIGHT reserve acquisition and measurement phases.

This module installs two narrow phase-boundary adapters before delegating:
pretty-printed design/freeze manifests are read as JSON and still verified by
their bound file/self hashes, and a valid terminal HippoRAG output survives a
controller SIGTERM even when its optional captured stdout/stderr logs were not
written during the final parent-process micro-window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from assumption_agent.benchmarks import bright_reasoning_retrieval_reserve_acquisition_v1 as acquisition
from assumption_agent.benchmarks import bright_reasoning_retrieval_reserve_measurement_v1 as measurement
from assumption_agent.benchmarks import bright_reasoning_retrieval_study_v1 as base
from replication_runtime.bright_official_hipporag_v1.contract import (
    parse_output as parse_hipporag_output,
)


def _read_bound_json(path: Path, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise acquisition.BrightReserveAcquisitionError(f"{field} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise acquisition.BrightReserveAcquisitionError(f"{field} is invalid") from exc
    if not isinstance(value, dict):
        raise acquisition.BrightReserveAcquisitionError(f"{field} root is invalid")
    return value


def _optional_regular_file_sha256(path: Path) -> str | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise measurement.BrightReserveMeasurementError(
            "recovered HippoRAG log path is not a regular file"
        )
    return base.file_sha256(path)


def _recoverable_existing_hipporag(
    item_root: Path, candidate_rows: Sequence[int]
) -> dict[str, Any]:
    output_path = item_root / "output.json"
    if output_path.is_symlink() or not output_path.is_file():
        raise measurement.BrightReserveMeasurementError(
            "terminal HippoRAG output is unavailable"
        )
    payload = parse_hipporag_output(output_path.read_bytes())
    if (
        payload["graph_node_count"] <= base.core.POOL_SIZE
        or payload["graph_edge_count"] <= 0
    ):
        raise measurement.BrightReserveMeasurementError(
            "existing HippoRAG graph is invalid"
        )
    try:
        top_rows = [candidate_rows[index] for index in payload["top_ordinals"]]
    except (IndexError, TypeError) as exc:
        raise measurement.BrightReserveMeasurementError(
            "existing HippoRAG candidate mapping drifted"
        ) from exc
    if len(top_rows) != base.core.TOP_K or len(set(top_rows)) != base.core.TOP_K:
        raise measurement.BrightReserveMeasurementError(
            "existing HippoRAG top-k drifted"
        )
    return {
        "graph_edge_count": payload["graph_edge_count"],
        "graph_node_count": payload["graph_node_count"],
        "output_file_sha256": base.file_sha256(output_path),
        "stderr_sha256": _optional_regular_file_sha256(item_root / "stderr.log"),
        "stdout_sha256": _optional_regular_file_sha256(item_root / "stdout.log"),
        "top_rows": top_rows,
    }


def _activate() -> None:
    acquisition._read_canonical = _read_bound_json
    measurement._existing_hipporag = _recoverable_existing_hipporag
    measurement.false = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("acquire", "prepare", "execute-actions", "resume-actions", "score"),
    )
    parser.add_argument("--project-root", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    _activate()
    if arguments.command == "acquire":
        result = acquisition.run(arguments.project_root)
    elif arguments.command == "prepare":
        result = measurement.prepare(arguments.project_root)
    elif arguments.command == "execute-actions":
        result = measurement.execute_actions(arguments.project_root, resume=False)
    elif arguments.command == "resume-actions":
        result = measurement.execute_actions(arguments.project_root, resume=True)
    else:
        result = measurement.score(arguments.project_root)
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
