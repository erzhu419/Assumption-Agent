"""Source-free diagnostic for the GSCL outer user-service contract.

This command is qualification-only.  It reads only the current cgroup and
systemd metadata, opens no benchmark source or labels, loads no model, invokes
no scorer, and cannot authorize a formal measurement.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from assumption_agent.benchmarks import (
    gscl_arn_formal_supervisor_v1 as supervisor,
)


ROOT = Path(
    "/var/tmp/gscl_unified_nonscoring_harness_20260730/work/"
    "outer_systemd_contract_diagnostic_r1"
)
OUTPUT = ROOT / "terminal.safe.json"


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
    if OUTPUT.exists():
        raise RuntimeError("diagnostic_output_not_fresh")
    control_group = supervisor._current_unified_cgroup()  # noqa: SLF001
    unit_id = supervisor._outer_service_unit_from_cgroup(  # noqa: SLF001
        control_group
    )
    command = [
        str(supervisor._SYSTEMCTL),  # noqa: SLF001
        "--user",
        "show",
        unit_id,
        "--no-pager",
    ]
    for key in supervisor._OUTER_SYSTEMD_LIVE_PROPERTIES:  # noqa: SLF001
        command.extend(("--property", key))
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=15,
    )
    properties = supervisor._parse_systemd_show(  # noqa: SLF001
        completed.stdout
    )
    original_validator = (
        supervisor._validate_outer_systemd_attestation  # noqa: SLF001
    )
    try:
        supervisor._validate_outer_systemd_attestation = (  # type: ignore[attr-defined]  # noqa: SLF001
            lambda receipt, **_: receipt
        )
        receipt = supervisor._attest_current_outer_systemd_service(  # noqa: SLF001
            writable_root=ROOT
        )
    finally:
        supervisor._validate_outer_systemd_attestation = (  # type: ignore[attr-defined]  # noqa: SLF001
            original_validator
        )
    validation_error = ""
    try:
        original_validator(receipt, expected_writable_root=ROOT)
    except Exception as exc:  # diagnostic must expose the exact fail-closed id
        validation_error = f"{type(exc).__name__}:{exc}"
    expected_contract = supervisor._outer_systemd_full_contract(  # noqa: SLF001
        ROOT
    )
    property_mismatches = {
        key: {
            "actual": properties.get(key),
            "expected": value,
        }
        for key, value in expected_contract.items()
        if properties.get(key) != value
    }
    body: dict[str, Any] = {
        "api_evaluation_count": 0,
        "effect_gate_count": 0,
        "formal_measurement": False,
        "label_open_count": 0,
        "model_weight_load_count": 0,
        "network_endpoint_contacted": False,
        "official_source_access_count": 0,
        "property_mismatches": property_mismatches,
        "receipt": receipt,
        "schema": "gscl_outer_systemd_contract_diagnostic_v1",
        "scorer_call_count": 0,
        "source_content_supplied": False,
        "status": (
            "PASS_OUTER_SYSTEMD_CONTRACT"
            if not validation_error
            else "FAIL_OUTER_SYSTEMD_CONTRACT"
        ),
        "systemctl_returncode": completed.returncode,
        "unit_id": unit_id,
        "validation_error": validation_error,
    }
    _write_once(
        OUTPUT,
        {**body, "self_sha256": _content_hash(body)},
    )
    print(
        json.dumps(
            {
                "property_mismatch_keys": sorted(property_mismatches),
                "status": body["status"],
                "validation_error": validation_error,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if not validation_error else 2


if __name__ == "__main__":
    raise SystemExit(main())
