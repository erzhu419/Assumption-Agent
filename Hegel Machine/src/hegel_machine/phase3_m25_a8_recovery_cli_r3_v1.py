"""Dedicated CLI for the fixed A8 -> R1 -> R2 -> R3 -> R3.1 chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .phase3_m25_a8_recovery_amendment_r3_v1 import (
    A8R3RecoveryAmendmentError,
    DEFAULT_MANIFEST_PATH,
    execute_fixed_a8_r3_recovery_v1,
    inspect_r3_source_preflight_v1,
    prepare_fixed_a8_r3_authorization_v1,
    write_fixed_a8_r3_owner_authorization_v1,
)
from .phase3_m25_formal_container_executor_v1 import FormalContainerExecutorError


def _add_transaction_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--custody-directory", type=Path, required=True)
    parser.add_argument("--public-evidence-output", type=Path, required=True)
    parser.add_argument("--promotion-output", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phase3-m25-a8-r31-recovery-v1")
    commands = parser.add_subparsers(dest="operation", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)

    prepare = commands.add_parser("prepare-authorization")
    prepare.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    prepare.add_argument("--audit-directory", type=Path, required=True)
    _add_transaction_paths(prepare)

    authorize = commands.add_parser("authorize-fixed-transaction")
    authorize.add_argument("--audit-directory", type=Path, required=True)
    authorize.add_argument("--owner-confirmation", required=True)

    recover = commands.add_parser(
        "recover-fixed-complete-seed",
        help=(
            "consume still-unconsumed recovery attempt 3 exactly once under "
            "R3.1 canonical-byte admission and resume only the fixed A8 "
            "run/ledger with REAL_PENDING_RESUME"
        ),
    )
    recover.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    _add_transaction_paths(recover)
    recover.add_argument("--rust-formal-replay-binary", type=Path, required=True)
    recover.add_argument("--rust-bridge-dag-replay-binary", type=Path, required=True)
    recover.add_argument(
        "--rust-bridge-dag-qualification-report", type=Path, required=True
    )
    recover.add_argument("--audit-directory", type=Path, required=True)
    return parser


def _write_report(report: object) -> None:
    sys.stdout.buffer.write(
        (
            json.dumps(
                report,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "preflight":
            _write_report(inspect_r3_source_preflight_v1(manifest_path=args.manifest))
            return 0
        if args.operation == "prepare-authorization":
            prepare_fixed_a8_r3_authorization_v1(
                audit_directory=args.audit_directory,
                custody_directory=args.custody_directory,
                public_evidence_path=args.public_evidence_output,
                public_promotion_path=args.promotion_output,
                manifest_path=args.manifest,
            )
            return 0
        if args.operation == "authorize-fixed-transaction":
            write_fixed_a8_r3_owner_authorization_v1(
                audit_directory=args.audit_directory,
                owner_confirmation=args.owner_confirmation,
            )
            return 0
        if args.operation == "recover-fixed-complete-seed":
            execute_fixed_a8_r3_recovery_v1(
                custody_directory=args.custody_directory,
                rust_formal_replay_binary=args.rust_formal_replay_binary,
                rust_bridge_dag_replay_binary=args.rust_bridge_dag_replay_binary,
                rust_bridge_dag_qualification_report=(
                    args.rust_bridge_dag_qualification_report
                ),
                public_evidence_path=args.public_evidence_output,
                public_promotion_path=args.promotion_output,
                audit_directory=args.audit_directory,
                manifest_path=args.manifest,
            )
            return 0
    except (
        A8R3RecoveryAmendmentError,
        FormalContainerExecutorError,
        OSError,
        ValueError,
    ) as exc:
        code = getattr(exc, "code", "FAIL_M25_A8_R31_RECOVERY_CLI")
        detail = getattr(exc, "detail", str(exc))
        sys.stderr.write(
            json.dumps(
                {"ok": False, "error_code": code, "detail": detail},
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )
        return 2
    raise AssertionError("unreachable R3.1 recovery CLI operation")


if __name__ == "__main__":
    raise SystemExit(main())
