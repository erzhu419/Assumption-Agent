"""One-shot publisher for the offline container-actor qualification report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .phase3_container_actor_runtime_v1 import (
    run_live_qualification,
    validate_qualification_report,
)


def _publish_exclusive(path: Path, payload: bytes) -> None:
    """Publish one immutable public report without overwriting prior evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.fchmod(descriptor, 0o644)
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("short qualification-report write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run four offline technical actors and publish their public receipt."
    )
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)

    report = validate_qualification_report(run_live_qualification())
    encoded = (
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    _publish_exclusive(arguments.output, encoded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
