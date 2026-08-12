"""CLI for Phase-2A replay and Phase-2B/Phase-3 preregistration artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
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
    DSL_TOO_LARGE_STATUS,
    phase3_closure_capacity_preflight_report,
)
from .phase3_dsl_v1 import (
    OBSERVED_OMITTED_SINK_CONTROL,
    ODD_REDUCTION_TARGET,
)
from .phase3_strict_replay_v1 import (
    DEFAULT_RUST_BINARY,
    dual_capacity_replay_report,
    dual_strict_gate_report,
)
from .phase3_shrink1_publication_v1 import (
    shrink1_publication_report,
    shrink_transition_report,
)
from .phase3_shrink1_replay_v1 import (
    DEFAULT_RUST_BINARY as DEFAULT_SHRINK1_RUST_BINARY,
    dual_shrink1_capacity_replay_report,
    dual_shrink1_strict_gate_report,
)
from .phase3_shrink1_registry_v1 import (
    DSL_VERSION as SHRINK1_DSL_VERSION,
    FREEZE_VERSION as SHRINK1_FREEZE_VERSION,
)
from .phase3_m25_readiness_v1 import (
    MACHINE_FREEZE_ID as M25_FREEZE_VERSION,
    phase3_m25_readiness_report,
    validate_phase3_m25_readiness_report,
)
from .phase3_m25_replay_v1 import (
    DEFAULT_RUST_BINARY as DEFAULT_M25_RUST_BINARY,
    dual_synthetic_replay_report,
    validate_dual_synthetic_replay_report,
)
from .phase3_m25_external_v1 import (
    external_genesis_preflight_report,
    validate_external_genesis_preflight_report,
)
from .phase3_m25_qualification_v112 import (
    DEFAULT_RUST_BINARY as DEFAULT_M25_V112_RUST_BINARY,
    dual_typed_rows_qualification_report,
    validate_dual_typed_rows_qualification_report,
)
from .phase3_m25_errata_qualification_v1 import (
    dual_errata_qualification_report,
    publish_errata_qualification_report_v1,
    validate_errata_qualification_output_path,
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
        "phase3_parent_capacity_status": DSL_TOO_LARGE_STATUS,
        "phase3_parent_dsl_version": "hegel-old-dsl-v1.0.0",
        "phase3_child_dsl_version": SHRINK1_DSL_VERSION,
        "phase3_child_freeze_version": SHRINK1_FREEZE_VERSION,
        "phase3_m25_parent_freeze_version": "hegel-freeze-p2b-p3-v1.1.1",
        "phase3_m25_freeze_version": M25_FREEZE_VERSION,
        "phase3_child_subset_status": "VERIFIED_WITHIN_BUDGET",
        "phase3_child_subset_accepted_unique_count": 25_872,
        "phase3_executed_closure_status": "NOT_RUN",
        "phase3_complete_closure_enumerated": False,
        "phase3_m25_gates_satisfied": 14,
        "phase3_m25_gates_total": 24,
        "phase3_required_next_action": (
            "M25_EXACT_ERRATA_THEN_INDEPENDENT_EXTERNAL_GENESIS"
        ),
        "phase3_target_universe_rows": ODD_REDUCTION_TARGET.universe_rows,
        "phase3_null_control_universe_rows": (
            OBSERVED_OMITTED_SINK_CONTROL.universe_rows
        ),
        "claim_boundary": (
            "Phase-2A controlled typed-selector mechanics plus Phase-2B/Phase-3 "
            "preregistration infrastructure, a bounded parent-DSL DSL_TOO_LARGE "
            "result, and a child shrink-1 subset qualification that is not "
            "COMPLETE; no raw extraction, formal Phase-2 "
            "exit, extensional target verdict, OUTSIDE_FROZEN_CLOSURE "
            "certificate, or relation invention claim"
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


def command_phase3_strict_gate(
    output: Path | None,
    rust_binary: Path,
) -> int:
    report = dual_strict_gate_report(rust_binary.resolve())
    if output is not None:
        _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "VERIFIED" else 1


def command_phase3_strict_capacity_replay(
    output: Path | None,
    rust_binary: Path,
) -> int:
    report = dual_capacity_replay_report(rust_binary.resolve())
    if output is not None:
        _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["executed_closure_status"] == "DSL_TOO_LARGE" else 1


def command_phase3_shrink1_strict_gate(
    output: Path | None,
    rust_binary: Path,
) -> int:
    report = dual_shrink1_strict_gate_report(rust_binary.resolve())
    if output is not None:
        _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "VERIFIED" else 1


def command_phase3_shrink1_subset_replay(
    output: Path | None,
    rust_binary: Path,
) -> int:
    report = dual_shrink1_capacity_replay_report(rust_binary.resolve())
    if output is not None:
        _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "VERIFIED_WITHIN_BUDGET" else 1


def command_phase3_shrink1_publish(output: Path | None) -> int:
    report = shrink1_publication_report()
    if output is not None:
        _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    expected = (
        report["status"] == "SHRINK1_SUBSET_QUALIFIED_M3_BLOCKED"
        and report["child_execution_state"] == "NOT_RUN"
        and report["complete_closure_enumerated"] is False
    )
    return 0 if expected else 1


def command_phase3_shrink1_transition(output: Path | None) -> int:
    report = shrink_transition_report()
    if output is not None:
        _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["child_initial_state"] == "NOT_RUN" else 1


def command_phase3_m25_readiness(output: Path | None) -> int:
    report = phase3_m25_readiness_report()
    validate_phase3_m25_readiness_report(report)
    if output is not None:
        _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_phase3_m25_synthetic_replay(
    output: Path | None,
    rust_binary: Path,
) -> int:
    report = dual_synthetic_replay_report(rust_binary.resolve())
    validate_dual_synthetic_replay_report(report)
    if output is not None:
        _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "SYNTHETIC_FOUNDATION_DUAL_REPLAY_PASS" else 1


def command_phase3_m25_v112_qualify(
    output: Path | None,
    rust_binary: Path,
) -> int:
    report = dual_typed_rows_qualification_report(rust_binary.resolve())
    validate_dual_typed_rows_qualification_report(report, rust_binary.resolve())
    if output is not None:
        _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "DUAL_TYPED_ROWS_AND_ROOTS_CANDIDATE_PASS" else 1


def command_phase3_m25_external_preflight(output: Path | None) -> int:
    report = external_genesis_preflight_report()
    validate_external_genesis_preflight_report(report)
    if output is not None:
        _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_phase3_m25_errata_qualify(
    output: Path | None,
) -> int:
    validated_output = (
        None
        if output is None
        else validate_errata_qualification_output_path(output)
    )
    report = dual_errata_qualification_report()
    if validated_output is not None:
        # The publisher repeats validation and holds the parent dirfd during
        # O_EXCL install, closing the qualification-time substitution window.
        publish_errata_qualification_report_v1(validated_output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "DUAL_EXACT_WIRE_ERRATA_GOLDEN_PASS" else 1


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
    strict_gate = subparsers.add_parser(
        "phase3-strict-gate",
        help="verify Python/Rust strict AST, CBOR, and RFC6962 golden vectors",
    )
    strict_gate.add_argument("--output", type=Path)
    strict_gate.add_argument(
        "--rust-binary",
        type=Path,
        default=DEFAULT_RUST_BINARY,
    )
    strict_capacity = subparsers.add_parser(
        "phase3-strict-capacity-replay",
        help=(
            "dual-replay the 64,680 strict capacity subset after the golden "
            "gate; this cannot issue an outside certificate"
        ),
    )
    strict_capacity.add_argument("--output", type=Path)
    strict_capacity.add_argument(
        "--rust-binary",
        type=Path,
        default=DEFAULT_RUST_BINARY,
    )
    shrink1_gate = subparsers.add_parser(
        "phase3-shrink1-strict-gate",
        help="verify child sparse-registry admission with Python and Rust",
    )
    shrink1_gate.add_argument("--output", type=Path)
    shrink1_gate.add_argument(
        "--rust-binary",
        type=Path,
        default=DEFAULT_SHRINK1_RUST_BINARY,
    )
    shrink1_subset = subparsers.add_parser(
        "phase3-shrink1-subset-replay",
        help=(
            "dual-replay the 25,872-source shrink-1 subset; this is never a "
            "complete-closure claim"
        ),
    )
    shrink1_subset.add_argument("--output", type=Path)
    shrink1_subset.add_argument(
        "--rust-binary",
        type=Path,
        default=DEFAULT_SHRINK1_RUST_BINARY,
    )
    shrink1_publish = subparsers.add_parser(
        "phase3-shrink1-publish",
        help="emit child publication, binding, null-root, and M3 gate state",
    )
    shrink1_publish.add_argument("--output", type=Path)
    shrink1_transition = subparsers.add_parser(
        "phase3-shrink1-transition",
        help="emit the post-subset diagnostic DSL shrink transition record",
    )
    shrink1_transition.add_argument("--output", type=Path)
    m25_readiness = subparsers.add_parser(
        "phase3-m25-readiness",
        help=(
            "emit fail-closed M2.5 foundation/specification/custody readiness; "
            "this never instantiates a seed or formal root"
        ),
    )
    m25_readiness.add_argument("--output", type=Path)
    m25_replay = subparsers.add_parser(
        "phase3-m25-synthetic-replay",
        help=(
            "dual-replay public M2.5 primitive vectors; this never creates "
            "an authoritative root, seed, signature, or gate pass"
        ),
    )
    m25_replay.add_argument("--output", type=Path)
    m25_replay.add_argument(
        "--rust-binary",
        type=Path,
        default=DEFAULT_M25_RUST_BINARY,
    )
    m25_v112 = subparsers.add_parser(
        "phase3-m25-v112-qualify",
        help=(
            "independently replay v1.1.2 typed rows and candidate roots in "
            "Python/Rust; this cannot claim formal roots or advance a gate"
        ),
    )
    m25_v112.add_argument("--output", type=Path)
    m25_v112.add_argument(
        "--rust-binary",
        type=Path,
        default=DEFAULT_M25_V112_RUST_BINARY,
    )
    m25_external = subparsers.add_parser(
        "phase3-m25-external-preflight",
        help=(
            "emit the exact-errata external-genesis stop report; this performs "
            "no CSPRNG call, marker creation, signing, or state transition"
        ),
    )
    m25_external.add_argument("--output", type=Path)
    m25_errata = subparsers.add_parser(
        "phase3-m25-errata-qualify",
        help=(
            "independently replay the Python/Rust E1-E12 exact-wire vectors; "
            "a pass only authorizes the separate external-genesis workflow"
        ),
    )
    m25_errata.add_argument("--output", type=Path)
    vertical = subparsers.add_parser(
        "vertical-slice", help="run the controlled candidate/shadow-only slice"
    )
    vertical.add_argument("--output", type=Path)
    subparsers.add_parser("demo", help="print the frozen initial theory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    if (
        raw_argv
        and type(raw_argv[0]) is str
        and raw_argv[0] == "phase2b-verify-v2-structure"
    ):
        from .phase2b_strict_recognizer_cli_v2 import main as strict_v2_main

        return strict_v2_main(raw_argv[1:])
    args = build_parser().parse_args(raw_argv)
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
    if args.command == "phase3-strict-gate":
        return command_phase3_strict_gate(args.output, args.rust_binary)
    if args.command == "phase3-strict-capacity-replay":
        return command_phase3_strict_capacity_replay(
            args.output,
            args.rust_binary,
        )
    if args.command == "phase3-shrink1-strict-gate":
        return command_phase3_shrink1_strict_gate(args.output, args.rust_binary)
    if args.command == "phase3-shrink1-subset-replay":
        return command_phase3_shrink1_subset_replay(args.output, args.rust_binary)
    if args.command == "phase3-shrink1-publish":
        return command_phase3_shrink1_publish(args.output)
    if args.command == "phase3-shrink1-transition":
        return command_phase3_shrink1_transition(args.output)
    if args.command == "phase3-m25-readiness":
        return command_phase3_m25_readiness(args.output)
    if args.command == "phase3-m25-synthetic-replay":
        return command_phase3_m25_synthetic_replay(args.output, args.rust_binary)
    if args.command == "phase3-m25-v112-qualify":
        return command_phase3_m25_v112_qualify(args.output, args.rust_binary)
    if args.command == "phase3-m25-external-preflight":
        return command_phase3_m25_external_preflight(args.output)
    if args.command == "phase3-m25-errata-qualify":
        return command_phase3_m25_errata_qualify(args.output)
    if args.command == "vertical-slice":
        return command_vertical_slice(args.output)
    raise AssertionError(f"unhandled command: {args.command}")
