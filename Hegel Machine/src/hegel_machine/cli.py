"""CLI for Phase-2A replay and Phase-2B/Phase-3 preregistration artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .benchmark import run_phase2_benchmark
from .bootstrap import initial_theory
from .milestones import (
    CURRENT_SCALE_CAPABILITY_NAME,
    CURRENT_TYPED_SELECTION_CAPABILITY_NAME,
    PHASE2A,
    PHASE2B,
    PHASE3A,
)
from .phase2_exit import run_phase2_exit_benchmark
from .phase2b_protocol import (
    frozen_phase2b_protocol,
    phase2b_preregistration_report,
)
from .phase3_contract import (
    DEFAULT_PHASE3_PREREGISTRATION,
    phase3_preregistration_report,
)
from .phase3_closure_preflight import (
    CONDITIONAL_CAPACITY_STATUS,
    phase3_closure_capacity_preflight_report,
)
from .phase3_dsl_v1 import (
    OBSERVED_OMITTED_SINK_CONTROL,
    ODD_REDUCTION_TARGET,
)
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
        "current_milestone_id": PHASE2A.machine_id,
        "current_milestone_name": PHASE2A.name,
        "current_typed_capability": CURRENT_TYPED_SELECTION_CAPABILITY_NAME,
        "current_scale_capability": CURRENT_SCALE_CAPABILITY_NAME,
        "next_phase2b_milestone": PHASE2B.name,
        "phase2b_ready_for_holdout_generation": (
            frozen_phase2b_protocol().ready_for_holdout_generation
        ),
        "next_phase3a_milestone": PHASE3A.name,
        "phase3_ready_for_outside_certificate": (
            DEFAULT_PHASE3_PREREGISTRATION.ready_for_outside_certificate
        ),
        "phase3_capacity_preflight_status": CONDITIONAL_CAPACITY_STATUS,
        "phase3_executed_closure_status": "NOT_RUN",
        "phase3_target_universe_rows": ODD_REDUCTION_TARGET.universe_rows,
        "phase3_null_control_universe_rows": (
            OBSERVED_OMITTED_SINK_CONTROL.universe_rows
        ),
        "claim_boundary": (
            "Phase-2A controlled typed-selector mechanics plus Phase-2B/Phase-3 "
            "preregistration infrastructure; no raw extraction, formal Phase-2 "
            "exit, OUTSIDE_FROZEN_CLOSURE certificate, or relation invention "
            "claim"
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_phase2_exit(output: Path | None) -> int:
    report = run_phase2_exit_benchmark()
    if output is not None:
        _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "controlled_api_selector_qualified" else 1


def command_phase2b_preregister(output: Path | None) -> int:
    report = phase2b_preregistration_report()
    if output is not None:
        _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_phase3_preregister(output: Path | None) -> int:
    report = phase3_preregistration_report()
    if output is not None:
        _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_phase3_closure_preflight(output: Path | None) -> int:
    report = phase3_closure_capacity_preflight_report(replay_subset=True)
    if output is not None:
        _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_vertical_slice(output: Path | None) -> int:
    report = run_controlled_vertical_slice()
    if output is not None:
        _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["decision"] == "candidate_framework" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hegel-machine",
        description=(
            "Offline Phase-2A replay plus fail-closed Phase-2B/Phase-3 "
            "preregistration; no raw-extraction or formal exit claim."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser(
        "benchmark", help="run the provided-binding verifier qualification"
    )
    benchmark.add_argument("--output", type=Path)
    phase2_exit = subparsers.add_parser(
        "phase2-exit",
        help=(
            "legacy command for the Phase-2A controlled typed-selector mechanics "
            "qualification (not a formal Phase-2 exit)"
        ),
    )
    phase2_exit.add_argument("--output", type=Path)
    phase2a = subparsers.add_parser(
        "phase2a",
        help="run the Phase-2A controlled typed-selector mechanics qualification",
    )
    phase2a.add_argument("--output", type=Path)
    phase2b = subparsers.add_parser(
        "phase2b-preregister",
        help="emit the unsealed Phase-2B protocol/readiness artifact",
    )
    phase2b.add_argument("--output", type=Path)
    phase3 = subparsers.add_parser(
        "phase3-preregister",
        help="emit the fail-closed old-DSL freeze/readiness artifact",
    )
    phase3.add_argument("--output", type=Path)
    phase3_preflight = subparsers.add_parser(
        "phase3-closure-preflight",
        help=(
            "replay the constructive old-DSL capacity subset; this cannot "
            "issue an outside certificate"
        ),
    )
    phase3_preflight.add_argument("--output", type=Path)
    vertical = subparsers.add_parser(
        "vertical-slice", help="run the controlled candidate/shadow-only slice"
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
    if args.command in {"phase2-exit", "phase2a"}:
        return command_phase2_exit(args.output)
    if args.command == "phase2b-preregister":
        return command_phase2b_preregister(args.output)
    if args.command == "phase3-preregister":
        return command_phase3_preregister(args.output)
    if args.command == "phase3-closure-preflight":
        return command_phase3_closure_preflight(args.output)
    if args.command == "vertical-slice":
        return command_vertical_slice(args.output)
    raise AssertionError(f"unhandled command: {args.command}")
