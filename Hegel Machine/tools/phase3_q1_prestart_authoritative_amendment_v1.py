#!/usr/bin/env python3
"""CLI for the zero-execution Q1 prestart authoritative amendment."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
import types
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "hegel_machine"
PACKAGE_NAME = "hegel_machine"
if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(PACKAGE_ROOT)]
    package.__package__ = PACKAGE_NAME
    sys.modules[PACKAGE_NAME] = package

_core = importlib.import_module(
    "hegel_machine.phase3_q1_prestart_authoritative_amendment_v1"
)


DEFAULT_CONTRACT = (
    PROJECT_ROOT
    / "config/phase3_q1_prestart_authoritative_amendment_v1.json"
)
PENDING_EXIT_CODE = 75


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish or replay the Q1 14/20 prestart amendment without "
            "starting Q1 or invoking an endpoint"
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--record-authorization", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--capsule-inputs", type=Path)
    parser.add_argument("--capsule-digest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.preflight:
        if args.capsule_inputs is None or args.capsule_digest is not None:
            parser.error(
                "--preflight requires --capsule-inputs and forbids --capsule-digest"
            )
        result = _core.preflight_v1(args.contract, args.capsule_inputs)
    elif args.record_authorization:
        if args.capsule_digest is None or args.capsule_inputs is not None:
            parser.error(
                "--record-authorization requires --capsule-digest and forbids "
                "--capsule-inputs"
            )
        result = _core.record_authorization_v1(
            args.contract, args.capsule_digest
        )
    else:
        if args.capsule_digest is None or args.capsule_inputs is not None:
            parser.error(
                "--run requires --capsule-digest and forbids --capsule-inputs"
            )
        result = _core.run_v1(args.contract, args.capsule_digest)
    sys.stdout.buffer.write(_core.canonical_json_bytes_v1(result))
    if result["status"] == _core.PENDING_NOT_CONSUMED:
        return PENDING_EXIT_CODE
    if result["status"] == _core.TERMINAL_NO_RESUME:
        return 2
    if result["status"] == _core.NOT_AUTHORIZED:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
