"""Dedicated CLI for the exact formal ``phase3-m3-start`` boundary.

``prepare`` is the default and performs no write.  ``start`` requires an
explicit mode plus an absolute state path.  ``verify`` only replays an already
persisted state.  None of the modes can invoke M3 closure enumeration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .phase3_m3_local_admission_v1 import (
    M3LocalAdmissionError,
    validate_live_local_admission_v1,
)
from .phase3_m3_start_v1 import (
    FORMAL_RUN_ID_HEX,
    M3StartError,
    PUBLICATION_COMMIT_B,
    canonical_start_state_path_v1,
    load_publication_blobs_v1,
    prepare_authoritative_m3_start_v1,
    prepare_m3_start_v1,
    read_state_file_v1,
    require_canonical_start_state_path_v1,
    verify_m3_start_v1,
    write_state_exact_once_v1,
)


_DIRECT_ENTRYPOINT_SEAL = object()
FAIL_DIRECT_ENTRYPOINT = "FAIL_M3_START_DIRECT_ENTRYPOINT_REQUIRED"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase3-m3-start",
        description=(
            "Replay committed 24/24 evidence and prepare, explicitly persist, "
            "or verify the unique index-zero M3 start record. Never runs closure."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("prepare", "start", "verify"),
        default="prepare",
        help="safe default is side-effect-free prepare",
    )
    parser.add_argument(
        "--publication-revision",
        default=PUBLICATION_COMMIT_B,
        help="frozen authoritative Commit-B (override is still exact-identity checked)",
    )
    parser.add_argument(
        "--recorded-at-unix-seconds",
        type=int,
        help="required for prepare/start; exact timestamp carried by the formal record",
    )
    parser.add_argument(
        "--state",
        type=Path,
        help="absolute exact-once state path; required for start/verify",
    )
    parser.add_argument(
        "--admission-revision",
        help="required for start; explicit full local approval Commit D",
    )
    return parser


def _summary(report: dict[str, object], *, mode: str, status: str) -> str:
    return json.dumps(
        {
            "ok": True,
            "action": "phase3-m3-start",
            "mode": mode,
            "status": status,
            "publication_commit": report["publication_commit"],
            "basis_commit": report["basis_commit"],
            "formal_gate_count": 24,
            "run_id_hex": report["run_id_hex"],
            "state_record_root_hex": report["state_record_root_hex"],
            "child_state": report["child_state_after"],
            "running_phase": report["running_phase_after"],
            "closure_invoked": False,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    _launch_capability: object | None = None,
) -> int:
    if _launch_capability is not _DIRECT_ENTRYPOINT_SEAL:
        print(
            json.dumps(
                {
                    "ok": False,
                    "code": FAIL_DIRECT_ENTRYPOINT,
                    "detail": (
                        "invoke the committed direct entrypoint with "
                        "python -I -S -B"
                    ),
                },
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.mode in {"prepare", "start"} and arguments.recorded_at_unix_seconds is None:
        parser.error("--recorded-at-unix-seconds is required for prepare/start")
    if arguments.mode in {"start", "verify"} and arguments.state is None:
        parser.error("--state is required for start/verify")
    if arguments.mode == "start" and arguments.admission_revision is None:
        parser.error("--admission-revision is required for start")
    if arguments.state is not None and not arguments.state.is_absolute():
        parser.error("--state must be absolute")
    if (
        arguments.state is not None
        and arguments.state != canonical_start_state_path_v1(FORMAL_RUN_ID_HEX)
    ):
        parser.error("--state must equal the frozen run's canonical state path")

    try:
        if arguments.mode == "start":
            local_admission = validate_live_local_admission_v1(
                arguments.admission_revision
            )
        commit, evidence, promotion = load_publication_blobs_v1(
            publication_revision=arguments.publication_revision
        )
        if arguments.mode == "verify":
            state_bytes = read_state_file_v1(arguments.state)
            report = verify_m3_start_v1(
                state_bytes,
                evidence,
                promotion,
                publication_commit=commit,
            )
            require_canonical_start_state_path_v1(
                arguments.state, report["run_id_hex"]
            )
            status = "VERIFIED_EXISTING"
        else:
            if arguments.mode == "prepare":
                report = prepare_m3_start_v1(
                    evidence,
                    promotion,
                    publication_commit=commit,
                    recorded_at_unix_seconds=arguments.recorded_at_unix_seconds,
                )
                print(
                    _summary(
                        report,
                        mode="prepare",
                        status="PREPARED_DIAGNOSTIC_ONLY_NOT_PERSISTABLE",
                    )
                )
                return 0
            prepared = prepare_authoritative_m3_start_v1(
                evidence,
                promotion,
                publication_commit=commit,
                recorded_at_unix_seconds=arguments.recorded_at_unix_seconds,
            )
            report = dict(prepared.document)
            status = write_state_exact_once_v1(
                arguments.state,
                prepared,
                local_admission=local_admission,
            )
    except (M3StartError, M3LocalAdmissionError) as exc:
        print(
            json.dumps(
                {"ok": False, "code": exc.code, "detail": exc.detail},
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(_summary(report, mode=arguments.mode, status=status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FAIL_DIRECT_ENTRYPOINT", "main"]
