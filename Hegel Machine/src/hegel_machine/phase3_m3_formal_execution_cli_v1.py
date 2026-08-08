"""Canonical command boundary for formal M3 preflight and execution.

The safe default is ``preflight``.  Neither subcommand accepts a caller-
selected state, run, attempt, or outcome path.  Those paths are derived only
from the frozen formal run ID by the canonical path helpers.

``preflight`` verifies the checked-out runtime source identity, replays the
two Commit-B public blobs, and constructs a side-effect-free start candidate.
It never persists the candidate and never imports or invokes a Docker runner.

``execute`` repeats the runtime-source verification, consumes the already-
persisted canonical start state, and delegates the unique attempt and outcome
paths to the formal execution module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Final, NoReturn, Sequence

from .phase3_m25_commit_b_publication_audit_v1 import canonical_json_v1
from .phase3_m3_formal_execution_v1 import (
    CANONICAL_ATTEMPT_ID,
    M3FormalExecutionError,
    execute_formal_m3_v1,
    prepare_formal_execution_v1,
)
from .phase3_m3_local_admission_v1 import (
    M3LocalAdmissionError,
    M3_RUNTIME_SOURCE_PATHS,
    LocalTwoCommitAdmissionResultV1,
    canonical_json_v1 as canonical_local_admission_json_v1,
    validate_live_local_admission_v1,
)
from .phase3_m3_start_v1 import (
    FORMAL_RUN_ID_HEX,
    M3StartError,
    PUBLICATION_COMMIT_B,
    canonical_run_root_v1,
    canonical_start_state_path_v1,
    canonical_terminal_outcome_path_v1,
    load_publication_blobs_v1,
    prepare_m3_start_v1,
)


SCHEMA: Final = "hegel-phase3-m3-formal-execution-cli/1"
CLI_REPOSITORY_PATH: Final = (
    "Hegel Machine/src/hegel_machine/phase3_m3_formal_execution_cli_v1.py"
)
RUNTIME_SOURCE_PATHS: Final = tuple(
    M3_RUNTIME_SOURCE_PATHS
)
FAIL_USAGE: Final = "FAIL_M3_FORMAL_EXECUTION_CLI_USAGE"
FAIL_CLI: Final = "FAIL_M3_FORMAL_EXECUTION_CLI"
FAIL_DIRECT_ENTRYPOINT: Final = (
    "FAIL_M3_FORMAL_EXECUTION_DIRECT_ENTRYPOINT_REQUIRED"
)
_DIRECT_ENTRYPOINT_SEAL = object()


class M3FormalExecutionCliError(RuntimeError):
    """Stable error for CLI-only argument and output-boundary failures."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise M3FormalExecutionCliError(code, detail)


class _CanonicalArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        _fail(FAIL_USAGE, message)


def _parser() -> argparse.ArgumentParser:
    parser = _CanonicalArgumentParser(
        prog="phase3-m3-formal-execution",
        description=(
            "Verify the committed M3 runtime and either prepare a side-effect-free "
            "candidate (default) or explicitly execute the sole canonical formal run."
        ),
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=("preflight", "execute"),
        default="preflight",
        help="safe default is preflight; execute is the only state-consuming action",
    )
    parser.add_argument(
        "--admission-revision",
        required=True,
        help="explicit full local approval Commit D for the exact M3 runtime",
    )
    return parser


def _runtime_identity_summary(
    result: LocalTwoCommitAdmissionResultV1,
) -> dict[str, object]:
    def plain(value: object) -> object:
        return json.loads(canonical_local_admission_json_v1(value))

    return {
        "claim_level": result.receipt_fields["claim_level"],
        "runtime_commit_c": result.runtime_commit_c,
        "approval_commit_d": result.approval_commit_d,
        "artifact": plain(result.artifact_fields),
        "manifest": plain(result.manifest_fields),
        "receipt": plain(result.receipt_fields),
    }


def _canonical_paths() -> tuple[Path, Path, Path]:
    run_root = canonical_run_root_v1(FORMAL_RUN_ID_HEX)
    state_path = canonical_start_state_path_v1(FORMAL_RUN_ID_HEX)
    outcome_path = canonical_terminal_outcome_path_v1(FORMAL_RUN_ID_HEX)
    if state_path.parent != run_root or outcome_path.parent != run_root:
        _fail(FAIL_CLI, "canonical formal path relationship differs")
    return run_root, state_path, outcome_path


def _preflight(
    runtime: LocalTwoCommitAdmissionResultV1,
) -> dict[str, object]:
    publication_commit, evidence, promotion = load_publication_blobs_v1(
        publication_revision=PUBLICATION_COMMIT_B
    )
    if publication_commit != PUBLICATION_COMMIT_B:
        _fail(FAIL_CLI, "public evidence did not resolve to the frozen Commit-B")
    candidate = prepare_m3_start_v1(
        evidence,
        promotion,
        publication_commit=publication_commit,
        recorded_at_unix_seconds=int(time.time()),
    )
    if candidate.get("run_id_hex") != FORMAL_RUN_ID_HEX:
        _fail(FAIL_CLI, "prepared start candidate has the wrong formal run ID")
    run_root, state_path, outcome_path = _canonical_paths()
    return {
        "schema": SCHEMA,
        "ok": True,
        "action": "preflight",
        "runtime_source_identity": _runtime_identity_summary(runtime),
        "publication_commit_b": publication_commit,
        "basis_commit_a": candidate["basis_commit"],
        "run_id_hex": candidate["run_id_hex"],
        "candidate_start_state_record_root_hex": candidate[
            "state_record_root_hex"
        ],
        "candidate_start_state_artifact_sha256": candidate[
            "state_artifact_sha256"
        ],
        "canonical_run_root": run_root.as_posix(),
        "canonical_persisted_start_path": state_path.as_posix(),
        "canonical_terminal_outcome_path": outcome_path.as_posix(),
        "start_candidate_prepared": True,
        "start_written": False,
        "persisted_start_read": False,
        "docker_invoked": False,
        "formal_execution_invoked": False,
        "state_changed": False,
    }


def _execute(
    runtime: LocalTwoCommitAdmissionResultV1,
) -> dict[str, object]:
    publication_commit, evidence, promotion = load_publication_blobs_v1(
        publication_revision=PUBLICATION_COMMIT_B
    )
    if publication_commit != PUBLICATION_COMMIT_B:
        _fail(FAIL_CLI, "public evidence did not resolve to the frozen Commit-B")
    run_root, state_path, outcome_path = _canonical_paths()
    prepared = prepare_formal_execution_v1(
        state_path,
        evidence,
        promotion,
        publication_commit=publication_commit,
        expected_admission_revision=runtime.approval_commit_d,
    )
    publication = execute_formal_m3_v1(
        prepared,
        run_root=run_root,
        attempt_id=CANONICAL_ATTEMPT_ID,
        outcome_path=outcome_path,
    )
    document = publication.document
    terminal_status = document.get("closure_status")
    if terminal_status is None:
        terminal_status = document.get("terminal_status")
    if type(terminal_status) is not str:
        _fail(FAIL_CLI, "formal execution returned no terminal status")
    artifact_sha256 = document.get("outcome_artifact_sha256")
    if type(artifact_sha256) is not str:
        _fail(FAIL_CLI, "formal execution returned no outcome artifact digest")
    return {
        "schema": SCHEMA,
        "ok": True,
        "action": "execute",
        "runtime_source_identity": _runtime_identity_summary(runtime),
        "publication_commit_b": publication_commit,
        "run_id_hex": FORMAL_RUN_ID_HEX,
        "canonical_run_root": run_root.as_posix(),
        "canonical_persisted_start_path": state_path.as_posix(),
        "canonical_terminal_outcome_path": outcome_path.as_posix(),
        "attempt_id": CANONICAL_ATTEMPT_ID,
        "persisted_start_read": True,
        "formal_execution_invoked": True,
        "publication_status": publication.status,
        "terminal_status": terminal_status,
        "outcome_schema": document.get("schema"),
        "outcome_artifact_sha256": artifact_sha256,
    }


def _error_payload(error: BaseException) -> dict[str, object]:
    code = getattr(error, "code", FAIL_CLI)
    detail = getattr(error, "detail", str(error))
    if type(code) is not str or not code or "\x00" in code:
        code = FAIL_CLI
    if type(detail) is not str:
        detail = type(error).__name__
    return {
        "schema": SCHEMA,
        "ok": False,
        "code": code,
        "detail": detail,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    _launch_capability: object | None = None,
) -> int:
    try:
        if _launch_capability is not _DIRECT_ENTRYPOINT_SEAL:
            _fail(
                FAIL_DIRECT_ENTRYPOINT,
                "invoke the committed direct entrypoint with python -I -S -B",
            )
        arguments = _parser().parse_args(argv)
        runtime = validate_live_local_admission_v1(
            arguments.admission_revision,
        )
        if arguments.action == "preflight":
            payload = _preflight(runtime)
        elif arguments.action == "execute":
            payload = _execute(runtime)
        else:  # pragma: no cover - argparse choices and the branch are both frozen.
            _fail(FAIL_USAGE, "unknown formal action")
    except (
        M3FormalExecutionCliError,
        M3FormalExecutionError,
        M3LocalAdmissionError,
        M3StartError,
        OSError,
        ValueError,
    ) as exc:
        sys.stderr.buffer.write(canonical_json_v1(_error_payload(exc)))
        return 2
    sys.stdout.buffer.write(canonical_json_v1(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLI_REPOSITORY_PATH",
    "FAIL_CLI",
    "FAIL_USAGE",
    "M3FormalExecutionCliError",
    "RUNTIME_SOURCE_PATHS",
    "SCHEMA",
    "main",
]
