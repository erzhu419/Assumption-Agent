#!/usr/bin/env python3
"""Write the latest GSCL Phase-0 safe receipt by atomic replacement."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


PROJECT_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_PACKAGE_ROOT))

from assumption_agent.benchmarks.gscl_phase0_offline_qualification_v1 import (
    run_extended_qualification,
    run_narrative_source_free_qualification,
    run_qualification,
)
from assumption_agent.generalized_structural_correspondence_v1 import (
    strict_canonical_bytes,
)


DEFAULT_OUTPUT = Path(
    "artifacts/gscl_phase0_offline_qualification_v1/latest.safe.json"
)
DEFAULT_EXTENDED_OUTPUT = Path(
    "artifacts/gscl_phase0_offline_qualification_v1/"
    "latest.unified.safe.json"
)
DEFAULT_NARRATIVE_OUTPUT = Path(
    "artifacts/gscl_phase0_offline_qualification_v1/"
    "latest.narrative-source-free.safe.json"
)


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(strict_canonical_bytes(payload))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--extended",
        action="store_true",
        help=(
            "run the same-lineage controlled evidence and source-free "
            "narrative extensions in addition to the Phase-0 checks"
        ),
    )
    modes.add_argument(
        "--narrative-source-free",
        action="store_true",
        help=(
            "run only the source-free narrative extension with test "
            "stubs and no benchmark/model/network access"
        ),
    )
    arguments = parser.parse_args()
    if arguments.extended:
        receipt = run_extended_qualification()
        default_output = DEFAULT_EXTENDED_OUTPUT
    elif arguments.narrative_source_free:
        receipt = run_narrative_source_free_qualification()
        default_output = DEFAULT_NARRATIVE_OUTPUT
    else:
        receipt = run_qualification()
        default_output = DEFAULT_OUTPUT
    output = arguments.output or default_output
    _write_atomic(output, receipt)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": receipt["status"],
                "self_hash": receipt["self_hash"],
                "issue_ids": receipt["issue_ids"],
            },
            sort_keys=True,
        )
    )
    return (
        0
        if receipt["status"]
        in {
            "PASS_PHASE0_KERNEL_ONLY",
            "PASS_GSCL_UNIFIED_NONSCORING_HARNESS",
            "PASS_GSCL_NARRATIVE_SOURCE_FREE_QUALIFICATION",
        }
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
