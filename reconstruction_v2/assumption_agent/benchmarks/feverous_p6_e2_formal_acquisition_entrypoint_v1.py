"""Safe module entry point for the one-shot FEVEROUS formal acquisition.

The command line accepts only the project root.  Implementation-freeze and
identity/compiler qualification hashes are read from the committed, verified
freeze; neither a selection secret nor a caller-supplied binding can enter via
argv.  The acquisition module creates the secret internally after the freeze
check.  Standard output contains only an aggregate completion receipt.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from assumption_agent.benchmarks import (
    feverous_p6_e2_formal_acquisition_v1 as formal_acquisition,
)
from assumption_agent.benchmarks import (
    feverous_p6_e2_implementation_freeze_v1 as implementation_freeze,
)


VERSION = "feverous_p6_e2_formal_acquisition_entrypoint_v1"
SUMMARY_SCHEMA = f"{VERSION}_aggregate_summary"


class FeverousFormalAcquisitionEntrypointError(RuntimeError):
    """Committed prerequisites or aggregate completion bindings drifted."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FeverousFormalAcquisitionEntrypointError(
            "aggregate summary is not canonical JSON"
        ) from exc


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def run_from_committed_freeze(project: str | Path) -> Mapping[str, Any]:
    """Verify the committed freeze, acquire once, and return no private data."""

    freeze = implementation_freeze.verify_committed_implementation_freeze(
        project
    )
    if not isinstance(freeze, Mapping):
        raise FeverousFormalAcquisitionEntrypointError(
            "implementation freeze verifier returned no binding"
        )
    freeze_sha = freeze.get("implementation_freeze_sha256")
    qualification_sha = freeze.get(
        "identity_compiler_qualification_sha256"
    )
    if not _is_sha256(freeze_sha) or not _is_sha256(qualification_sha):
        raise FeverousFormalAcquisitionEntrypointError(
            "committed freeze lacks acquisition prerequisite hashes"
        )
    receipt = formal_acquisition.perform_formal_acquisition_once(
        project=project,
        implementation_freeze_sha256=str(freeze_sha),
        identity_full_compile_equivalence_qualification_sha256=str(
            qualification_sha
        ),
    )
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("status")
        != "all_four_train_blocks_acquired_before_any_action_or_outcome"
        or receipt.get("implementation_freeze_sha256") != freeze_sha
        or receipt.get(
            "identity_full_compile_equivalence_qualification_sha256"
        )
        != qualification_sha
        or not _is_sha256(receipt.get("acquisition_receipt_sha256"))
        or receipt.get("all_blocks_one_acquisition") is not True
        or receipt.get("action_retrieval_utility_or_evaluator_calls") != 0
        or receipt.get("development_or_test_source_accessed") is not False
        or receipt.get("online_evaluator_calls") != 0
    ):
        raise FeverousFormalAcquisitionEntrypointError(
            "formal acquisition completion binding drifted"
        )
    body: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "version": VERSION,
        "status": "formal_acquisition_completed_from_committed_freeze",
        "implementation_freeze_sha256": freeze_sha,
        "identity_compiler_qualification_sha256": qualification_sha,
        "acquisition_receipt_sha256": receipt["acquisition_receipt_sha256"],
        "block_counts": dict(receipt.get("block_counts", {})),
        "all_blocks_one_acquisition": True,
        "private_selection_secret_logged": False,
        "private_pack_content_logged": False,
        "development_or_test_source_accessed": False,
        "online_evaluator_calls": 0,
    }
    return {**body, "entrypoint_summary_sha256": _stable_hash(body)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run FEVEROUS formal acquisition from its committed freeze."
        )
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Frozen reconstruction_v2 project root.",
    )
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = run_from_committed_freeze(arguments.project)
    print(
        json.dumps(
            summary,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "FeverousFormalAcquisitionEntrypointError",
    "SUMMARY_SCHEMA",
    "VERSION",
    "run_from_committed_freeze",
]
