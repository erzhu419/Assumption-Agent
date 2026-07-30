"""Command-line interface for the offline v0.1 kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .benchmark import run_phase2_benchmark
from .bootstrap import initial_theory
from .vertical_slice import run_controlled_vertical_slice


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def command_benchmark(output: Path | None) -> int:
    report = run_phase2_benchmark()
    if output is not None:
        _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["structural_accuracy"] == 1.0 else 1


def command_demo() -> int:
    theory = initial_theory()
    payload = {
        "version_id": theory.version_id,
        "schema_version": theory.schema_version,
        "law_count": len(theory.relation_laws),
        "law_kinds": [law.kind.value for law in theory.relation_laws],
        "universal_assumption_count": len(theory.hypothesis_families),
        "probe_count": len(theory.probes),
        "evaluator_epoch": theory.evaluator.epoch,
        "claim_boundary": (
            "known-law structural verification only; no relation invention claim"
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_vertical_slice(output: Path | None) -> int:
    report = run_controlled_vertical_slice()
    if output is not None:
        _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["decision"] == "candidate_framework" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hegel-machine")
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser(
        "benchmark", help="run the controlled Phase-2 benchmark"
    )
    benchmark.add_argument("--output", type=Path)
    vertical = subparsers.add_parser(
        "vertical-slice", help="run the controlled end-to-end qualification"
    )
    vertical.add_argument("--output", type=Path)
    subparsers.add_parser("demo", help="print the frozen initial theory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "benchmark":
        return command_benchmark(args.output)
    if args.command == "demo":
        return command_demo()
    if args.command == "vertical-slice":
        return command_vertical_slice(args.output)
    raise AssertionError(f"unhandled command: {args.command}")
