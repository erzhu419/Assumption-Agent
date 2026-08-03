"""Explicit CLI for offline M3 implementation-binding qualification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from .phase3_m3_implementation_qualification_v1 import (
    M3ImplementationQualificationError,
    build_qualified_formal_static_basis_v1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hegel-m3-implementation-qualification-v1"
    )
    parser.add_argument("--basis-commit", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        # ``mode`` passed to os.open is filtered by the caller's umask.  The
        # formal consumer requires one exact public-artifact mode, so publish
        # that mode explicitly instead of inheriting ambient process policy.
        os.fchmod(descriptor, 0o644)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short qualification receipt write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    basis = build_qualified_formal_static_basis_v1(args.basis_commit)
    receipt = basis.implementation_inputs["m3_implementation_qualification_receipt"]
    payload = (
        json.dumps(dict(receipt), ensure_ascii=True, sort_keys=True, indent=2)
        + "\n"
    ).encode("ascii")
    if args.output is not None:
        _exclusive_write(args.output, payload)
    sys.stdout.buffer.write(payload)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (M3ImplementationQualificationError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
