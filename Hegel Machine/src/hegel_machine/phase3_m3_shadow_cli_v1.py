"""Dedicated CLI for the non-authoritative Phase-3 M3 shadow track.

This module is intentionally separate from the formal project CLI.  Invoke it
as ``python -m hegel_machine.phase3_m3_shadow_cli_v1 admit|start``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .phase3_m3_shadow_admission_v1 import (
    DEFAULT_ADMISSION_ARTIFACT_PATH,
    DEFAULT_START_ARTIFACT_PATH,
    ShadowAdmissionError,
    admit_internal_shadow,
    load_admission_artifact,
    start_internal_shadow,
    write_json_exclusive,
)


def _absolute_existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("calculator endpoint path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise argparse.ArgumentTypeError("calculator endpoint file is absent") from error
    if not resolved.is_file() or path.is_symlink():
        raise argparse.ArgumentTypeError("calculator endpoint must be a real file")
    return resolved


def _add_calculator_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--python-calculator",
        type=_absolute_existing_file,
        required=True,
        help="absolute standalone Python FD3 calculator endpoint",
    )
    parser.add_argument(
        "--rust-calculator",
        type=_absolute_existing_file,
        required=True,
        help="absolute compiled Rust FD3 calculator endpoint",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase3-m3-shadow",
        description=(
            "Run the internal purpose-separated, non-authoritative shadow "
            "admission/start track. Formal status remains 14/24 / NOT_RUN."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    admit = subcommands.add_parser(
        "admit", help="run the no-key/no-seed 12/12 admission probes"
    )
    admit.add_argument(
        "--basis-commit",
        help="reachable committed basis (default: current HEAD)",
    )
    admit.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ADMISSION_ARTIFACT_PATH,
        help="exclusive admission JSON path",
    )
    _add_calculator_arguments(admit)

    start = subcommands.add_parser(
        "start", help="explicitly enter RUNNING_CANONICAL_ENUMERATION"
    )
    start.add_argument(
        "--admission",
        type=Path,
        default=DEFAULT_ADMISSION_ARTIFACT_PATH,
        help="validated 12/12 admission JSON",
    )
    start.add_argument(
        "--state-directory",
        type=Path,
        required=True,
        help="one-shot private state directory outside the repository",
    )
    start.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_START_ARTIFACT_PATH,
        help="exclusive start JSON path",
    )
    _add_calculator_arguments(start)
    return parser


def _summary(*, action: str, output: Path, report: dict[str, object]) -> str:
    shadow = report["shadow_track"]
    assert isinstance(shadow, dict)
    return json.dumps(
        {
            "ok": True,
            "action": action,
            "output": str(output),
            "artifact_kind": report["artifact_kind"],
            "formal_track_status": report["formal_track_status"],
            "shadow_run_id": report["shadow_run_id"],
            "shadow_state": shadow["state"],
            "formal_gate_delta": 0,
        },
        sort_keys=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "admit":
            report = admit_internal_shadow(
                basis_commit_id=arguments.basis_commit,
                python_calculator_path=arguments.python_calculator,
                rust_calculator_path=arguments.rust_calculator,
            )
            output = write_json_exclusive(arguments.output, report)
            action = "phase3-m3-shadow-admit"
        elif arguments.command == "start":
            admission = load_admission_artifact(arguments.admission)
            report = start_internal_shadow(
                admission,
                state_directory=arguments.state_directory,
                python_calculator_path=arguments.python_calculator,
                rust_calculator_path=arguments.rust_calculator,
            )
            output = write_json_exclusive(arguments.output, report)
            action = "phase3-m3-shadow-start"
        else:  # pragma: no cover - argparse rejects this branch.
            raise AssertionError("unreachable command")
    except ShadowAdmissionError as error:
        print(
            json.dumps(
                {"ok": False, "code": error.code, "detail": error.detail},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(_summary(action=action, output=output, report=report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
