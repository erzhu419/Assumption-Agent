"""Bind the qualified HippoRAG source before invoking the frozen worker."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys
from typing import Sequence


PATCHED_SOURCE_SHA256 = (
    "6d0938da96757504e88ec15ea88f15bc6a6605e006eeb00c780598330b4c698b"
)


class BoundWorkerError(RuntimeError):
    """The patched source was not the source imported by the worker."""


def assert_bound_source() -> Path:
    """Return the exact imported source path or fail closed."""

    from hipporag import HippoRAG

    source = Path(inspect.getfile(HippoRAG)).resolve(strict=True)
    if hashlib.sha256(source.read_bytes()).hexdigest() != PATCHED_SOURCE_SHA256:
        raise BoundWorkerError("qualified patched HippoRAG source is not bound")
    return source


def main(argv: Sequence[str] | None = None) -> int:
    assert_bound_source()
    from replication_runtime.wikisql_uao_official_v1 import worker

    return worker.main(argv)


if __name__ == "__main__":  # pragma: no cover - exercised by remote service.
    raise SystemExit(main(sys.argv[1:]))

