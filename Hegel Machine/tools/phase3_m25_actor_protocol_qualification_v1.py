#!/usr/bin/env python3
"""Run the Commit-A-bound, non-formal live Docker actor qualification."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hegel_machine.phase3_m25_actor_protocol_qualification_v1 import (  # noqa: E402
    ActorProtocolQualificationError,
    consume_live_actor_protocol_admission_v1,
    qualify_live_actor_protocol_v1,
)


def _exclusive_write(path: Path, payload: bytes) -> None:
    path = path.resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise OSError("diagnostic output parent must already be a real directory")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        os.fchmod(descriptor, 0o644)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short diagnostic report write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hegel-m25-live-actor-protocol-qualification-v1",
        description=(
            "Run four offline Docker actors with ephemeral keys and public "
            "synthetic split/marker inputs. Never instantiates a real seed."
        ),
    )
    parser.add_argument("--basis-commit", required=True)
    parser.add_argument(
        "--custody-directory",
        required=True,
        type=Path,
        help="existing empty repo-external, non-/tmp, Linux-local mode-0700 directory",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    admission = qualify_live_actor_protocol_v1(
        basis_commit=arguments.basis_commit,
        custody_directory=arguments.custody_directory,
    )
    consumed = consume_live_actor_protocol_admission_v1(
        admission,
        expected_basis_commit=arguments.basis_commit,
    )
    _exclusive_write(
        arguments.output,
        consumed.canonical_bundle_bytes,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ActorProtocolQualificationError, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
