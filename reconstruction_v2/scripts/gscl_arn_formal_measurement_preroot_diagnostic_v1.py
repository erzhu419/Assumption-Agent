"""Source-content-free pre-root qualification for formal ARN measurement.

This diagnostic executes the exact frozen test/runtime/receipt/systemd
preconditions used by the formal runner.  It never opens the staged ARN or
metadata bytes, never freezes an action, never begins an attempt, never opens
labels, and never invokes the scorer.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from assumption_agent.benchmarks import (
    gscl_arn_formal_measurement_v1 as measurement,
)
from assumption_agent.benchmarks import (
    gscl_arn_formal_supervisor_v1 as supervisor,
)
from assumption_agent.benchmarks import (
    gscl_arn_internal_factory_qualification_v1 as qualification,
)


OUTPUT_PATH = Path(
    "/var/tmp/gscl_unified_nonscoring_harness_20260730/work/"
    "arn_formal_measurement_preroot_diagnostic_v1/terminal.safe.json"
)


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _content_hash(value: Any) -> str:
    return hashlib.sha256(
        _canonical_bytes(value).rstrip(b"\n")
    ).hexdigest()


def _write_once(path: Path, value: dict[str, Any]) -> None:
    raw = _canonical_bytes(value)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            offset += os.write(descriptor, view[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    if OUTPUT_PATH.exists():
        raise RuntimeError("formal_preroot_output_not_fresh")
    measurement._validate_source_constants()  # noqa: SLF001
    measurement._preflight_staged_source_without_opening_content()  # noqa: SLF001
    measurement._preflight_never_started()  # noqa: SLF001
    qwen_receipts = measurement._validate_fixed_assets_and_receipts()  # noqa: SLF001
    qualification._preflight_frozen_main_runtime()  # noqa: SLF001
    qualification._preflight_frozen_runtime_binding_manifest()  # noqa: SLF001
    qualification._preflight_fixed_source_free_test_runtime()  # noqa: SLF001
    qualification._preflight_exactly_two_idle_gpus()  # noqa: SLF001
    closure, tests = measurement._source_free_runtime_closure(  # noqa: SLF001
        qwen_receipts=qwen_receipts
    )
    qualification._preflight_exactly_two_idle_gpus()  # noqa: SLF001
    measurement._preflight_staged_source_without_opening_content()  # noqa: SLF001
    measurement._preflight_never_started()  # noqa: SLF001
    outer = supervisor._attest_current_outer_systemd_service(  # noqa: SLF001
        writable_root=measurement.FORMAL_ROOT
    )
    commitments = measurement._freeze_commitments(  # noqa: SLF001
        closure=closure
    )
    runner_path = Path(measurement.__file__).resolve()
    runner_sha256 = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    if (
        closure.file_hashes.get(str(runner_path)) != runner_sha256
        or len(commitments) != 5
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in commitments.values()
        )
        or (measurement.FORMAL_ROOT / "state/action.freeze.json").exists()
        or (measurement.FORMAL_ROOT / "state/attempts").exists()
        or (
            measurement.FORMAL_ROOT
            / "control/outer_terminal.safe.json"
        ).exists()
    ):
        raise RuntimeError("formal_preroot_boundary_invalid")
    body: dict[str, Any] = {
        "api_evaluation_count": 0,
        "effect_gate_count": 0,
        "formal_action_freeze_count": 0,
        "formal_attempt_count": 0,
        "formal_measurement": False,
        "label_open_count": 0,
        "official_source_content_read_count": 0,
        "outer_systemd_attestation_self_hash": outer["self_hash"],
        "outer_systemd_stable_binding_sha256": outer[
            "stable_binding_sha256"
        ],
        "qualification_receipt_count": 3,
        "runtime_closure_self_hash": closure.manifest["self_hash"],
        "runner_sha256": runner_sha256,
        "schema": "gscl_arn_formal_measurement_preroot_diagnostic_v1",
        "scorer_call_count": 0,
        "source_free_test_attestation_self_hash": tests.receipt[
            "self_hash"
        ],
        "staged_dataset_expected_sha256": (
            measurement.OFFICIAL_DATASET_SHA256
        ),
        "staged_metadata_expected_sha256": (
            measurement.OFFICIAL_METADATA_SHA256
        ),
        "status": "PASS_FORMAL_MEASUREMENT_SOURCE_FREE_PREROOT",
    }
    _write_once(
        OUTPUT_PATH,
        {**body, "self_sha256": _content_hash(body)},
    )
    print(
        json.dumps(
            {
                "runtime_closure_self_hash": body[
                    "runtime_closure_self_hash"
                ],
                "status": body["status"],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
