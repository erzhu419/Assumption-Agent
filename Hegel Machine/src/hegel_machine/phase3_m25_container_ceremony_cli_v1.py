"""Public CLI for M2.5 readiness, explicit ceremony, and public replay.

There are no seed/key input flags.  ``execution-status`` and ``readiness`` run
the network-disabled implementation qualifier and may verify or install the
commit-bound Rust enumerator, but cannot create ceremony keys, seed, markers,
formal M3 output roots, or authority.  ``execute`` is the sole real-genesis
operation.  No command starts M3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

from .phase3_m25_container_ceremony_v1 import (
    FAIL_PUBLIC_BASIS_REGISTRY_PREIMAGE,
    FAIL_SPLIT_FULL_ENDPOINT_REQUIRED,
    M25ContainerCeremonyError,
    build_committed_public_basis_candidates_v1,
    read_marker_snapshot_v1,
    require_full_split_response_agreement_v2,
)
from .phase3_m25_formal_container_executor_v1 import (
    DockerCeremonyActorsV1,
    FormalContainerExecutorError,
    execute_formal_container_ceremony_v1,
    inspect_formal_ceremony_readiness_v1,
    replay_public_gate_evidence_v1,
)
from .phase3_m25_bridge_dag_binary_qualification_v1 import (
    DEFAULT_RUST_BRIDGE_DAG_QUALIFICATION_REPORT,
)
from .phase3_m25_formal_static_basis_v1 import FormalStaticBasisError
from .phase3_m3_implementation_qualification_v1 import (
    M3ImplementationQualificationError,
)


def _transport(value: object) -> object:
    if type(value) is bytes:
        return {"bytes_hex": value.hex()}
    if isinstance(value, Mapping):
        return {str(key): _transport(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_transport(item) for item in value]
    if value is None or type(value) in {bool, int, str}:
        return value
    raise TypeError(f"unsupported public transport value {type(value).__name__}")


def _write_json(path: Path | None, value: object) -> None:
    payload = (
        json.dumps(_transport(value), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    )
    if path is None:
        sys.stdout.write(payload)
        return
    flags = "x"  # O_EXCL semantics; never overwrite ceremony evidence.
    with path.open(flags, encoding="ascii", newline="\n") as stream:
        stream.write(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phase3-m25-container-ceremony-v1")
    subparsers = parser.add_subparsers(dest="operation", required=True)

    basis = subparsers.add_parser(
        "public-basis",
        help="construct the committed, non-secret candidate preimage subset",
    )
    basis.add_argument("--basis-commit", required=True)
    basis.add_argument("--output", type=Path)

    split = subparsers.add_parser(
        "validate-split-response",
        help="require exact Python/Rust full FD-5 public response agreement",
    )
    split.add_argument("--python-frame", type=Path, required=True)
    split.add_argument("--rust-frame", type=Path, required=True)
    split.add_argument("--output", type=Path)

    marker = subparsers.add_parser("marker-status", help="read a public marker snapshot")
    marker.add_argument("--marker", type=Path, required=True)
    marker.add_argument("--output", type=Path)

    status = subparsers.add_parser(
        "execution-status",
        help="run offline qualification and report basis-specific readiness",
    )
    status.add_argument("--basis-commit", required=True)
    status.add_argument("--output", type=Path)

    readiness = subparsers.add_parser(
        "readiness",
        help="run offline qualification and check all pre-genesis blockers",
    )
    readiness.add_argument("--basis-commit", required=True)
    readiness.add_argument("--output", type=Path)

    replay = subparsers.add_parser(
        "replay-public",
        help="reconstruct GateEvidenceInputsV1 and replay 24/24 public evidence",
    )
    replay.add_argument("--input", required=True, type=Path)
    replay.add_argument("--output", type=Path)

    execute = subparsers.add_parser(
        "execute",
        help="explicit one-shot real container genesis; never starts M3",
    )
    execute.add_argument("--basis-commit", required=True)
    execute.add_argument("--actor-qualification", required=True, type=Path)
    execute.add_argument("--errata-qualification", required=True, type=Path)
    execute.add_argument("--custody-directory", required=True, type=Path)
    execute.add_argument(
        "--qualification-custody-directory",
        required=True,
        type=Path,
        help="separate empty 0700 repo-external custody for same-process no-seed admission",
    )
    execute.add_argument("--rust-formal-replay-binary", required=True, type=Path)
    execute.add_argument(
        "--rust-bridge-dag-qualification-report",
        type=Path,
        default=DEFAULT_RUST_BRIDGE_DAG_QUALIFICATION_REPORT,
        help="stable post-Commit-A offline Rust bridge qualification report",
    )
    execute.add_argument("--public-evidence-output", required=True, type=Path)
    execute.add_argument("--promotion-output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "public-basis":
            report = build_committed_public_basis_candidates_v1(args.basis_commit)
            _write_json(args.output, report)
            return 0
        if args.operation == "validate-split-response":
            response = require_full_split_response_agreement_v2(
                args.python_frame.read_bytes(), args.rust_frame.read_bytes()
            )
            _write_json(
                args.output,
                {
                    "schema": "hegel-phase3-split-fd5-dual-agreement/1",
                    "seed_commitment_hex": response.seed_commitment.hex(),
                    "partitions": [
                        {
                            "role_id": row.role_id,
                            "partition_id": row.partition_id,
                            "row_count": row.row_count,
                            "root_hex": row.root.hex(),
                        }
                        for row in response.partitions
                    ],
                    "python_rust_exact_frame_equal": True,
                    "gate_22_claimed": False,
                },
            )
            return 0
        if args.operation == "marker-status":
            snapshot = read_marker_snapshot_v1(args.marker)
            _write_json(
                args.output,
                {
                    "state": snapshot.state,
                    "split_version_digest_hex": snapshot.split_version_digest.hex(),
                    "seed_commitment_manifest_root_hex_or_null": (
                        None
                        if snapshot.seed_commitment_manifest_root is None
                        else snapshot.seed_commitment_manifest_root.hex()
                    ),
                    "custodian_key_id_hex": snapshot.custodian_key_id.hex(),
                    "created_at_unix_seconds": snapshot.created_at_unix_seconds,
                },
            )
            return 0
        if args.operation == "execution-status":
            readiness = inspect_formal_ceremony_readiness_v1(args.basis_commit)
            report = readiness.public_report()
            report.update(
                {
                    "schema": "hegel-phase3-m25-execution-status/2",
                    "ceremony_execution_enabled_for_basis": readiness.ready,
                    "external_genesis_executed": False,
                    "blocking_prerequisites": [
                        {
                            "failure_code": code,
                            "required": "resolve this basis-specific fail-closed prerequisite",
                        }
                        for code in readiness.blockers
                    ],
                }
            )
            _write_json(
                args.output,
                report,
            )
            return 0
        if args.operation == "readiness":
            _write_json(
                args.output,
                inspect_formal_ceremony_readiness_v1(args.basis_commit).public_report(),
            )
            return 0
        if args.operation == "replay-public":
            value = json.loads(args.input.read_text(encoding="ascii"))
            _write_json(args.output, replay_public_gate_evidence_v1(value))
            return 0
        if args.operation == "execute":
            # This call is intentionally after the basis-specific offline
            # qualification guard.  Qualification containers and the durable
            # commit-bound Rust binary are allowed here; any unresolved control
            # still stops before actor key/marker/seed side effects.
            readiness = inspect_formal_ceremony_readiness_v1(args.basis_commit)
            if not readiness.ready:
                from .phase3_m25_formal_container_executor_v1 import FAIL_EXECUTION_BINDINGS
                raise FormalContainerExecutorError(
                    FAIL_EXECUTION_BINDINGS, ",".join(readiness.blockers)
                )
            actor_report = json.loads(
                args.actor_qualification.read_text(encoding="ascii")
            )
            errata_report = json.loads(
                args.errata_qualification.read_text(encoding="ascii")
            )
            timestamp = int(time.time())
            actors = DockerCeremonyActorsV1(
                basis_commit=args.basis_commit,
                custody_directory=args.custody_directory,
                rust_formal_replay_binary=args.rust_formal_replay_binary,
                rust_bridge_dag_qualification_report=(
                    args.rust_bridge_dag_qualification_report
                ),
                timestamp=timestamp,
            )
            execute_formal_container_ceremony_v1(
                basis_commit=args.basis_commit,
                actor_qualification_report=actor_report,
                errata_qualification_report=errata_report,
                custody_directory=args.custody_directory,
                qualification_custody_directory=(
                    args.qualification_custody_directory
                ),
                public_evidence_path=args.public_evidence_output,
                public_promotion_path=args.promotion_output,
                actors=actors,
            )
            return 0
    except (
        M25ContainerCeremonyError,
        FormalContainerExecutorError,
        FormalStaticBasisError,
        M3ImplementationQualificationError,
        OSError,
        ValueError,
        TypeError,
    ) as exc:
        if isinstance(
            exc,
            (
                M25ContainerCeremonyError,
                FormalContainerExecutorError,
                FormalStaticBasisError,
                M3ImplementationQualificationError,
            ),
        ):
            code = exc.code
            detail = exc.detail
        else:
            code = "FAIL_M25_CONTAINER_CEREMONY_CLI"
            detail = str(exc)
        sys.stderr.write(json.dumps({"ok": False, "error_code": code, "detail": detail}) + "\n")
        return 2
    raise AssertionError("unreachable CLI operation")


if __name__ == "__main__":
    raise SystemExit(main())
