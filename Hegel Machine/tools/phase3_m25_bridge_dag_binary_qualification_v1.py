#!/usr/bin/env python3
"""Emit the non-authoritative offline Rust bridge-DAG binary qualification."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hegel_machine.phase3_m25_bridge_dag_binary_qualification_v1 import (  # noqa: E402
    DEFAULT_RUST_BRIDGE_DAG_QUALIFICATION_REPORT,
    canonical_qualification_report_bytes_v1,
    qualify_rust_bridge_dag_binary_v1,
)


def _atomic_write(path: Path, payload: bytes) -> None:
    path = path.resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.pending-", dir=path.parent)
    pending = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        pending.chmod(0o644)
        os.replace(pending, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if pending.exists():
            pending.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-basis-commit")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RUST_BRIDGE_DAG_QUALIFICATION_REPORT,
    )
    parser.add_argument("--stdout", action="store_true")
    arguments = parser.parse_args()
    report = qualify_rust_bridge_dag_binary_v1(
        implementation_basis_commit=arguments.implementation_basis_commit
    )
    payload = canonical_qualification_report_bytes_v1(report)
    if arguments.stdout:
        sys.stdout.buffer.write(payload)
    else:
        _atomic_write(arguments.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
