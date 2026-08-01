"""Source-free diagnostic for MiniLM target-manifest drift.

It reproduces the item-factory import order, constructs the public-synthetic
portable encoder once, and emits only differing manifest field paths and
scalar/hash summaries.  No ARN source, label, scorer, network, or efficacy
measurement is available to this command.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

# Reproduce the exact import order that precedes encoder construction.
from assumption_agent.benchmarks import (  # noqa: F401
    gscl_arn_formal_item_factory_v1 as item_factory,
)
from replication_runtime.gscl_minilm_portable_v1 import (
    binding as gscl_binding,
)
from replication_runtime.qasper_minilm_portable_v2.binding import (
    PortableOfflineMiniLMEncoder,
)


BASE = Path("/var/tmp/gscl_unified_nonscoring_harness_20260730")
TARGET = (
    BASE
    / "assets/minilm_target_qualification_ext4_r1/target_manifest.json"
)
ASSET = BASE / "assets/minilm_asset_manifest.json"
MODEL = BASE / "assets/minilm_model"
ROOT = BASE / "work/minilm_target_drift_diagnostic_r1"
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


def _summary(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str) and len(value) <= 240:
        return value
    return {
        "canonical_sha256": _content_hash(value),
        "type": type(value).__name__,
    }


def _diff(
    expected: Any,
    observed: Any,
    *,
    path: str = "$",
) -> list[dict[str, Any]]:
    if type(expected) is not type(observed):
        return [
            {
                "path": path,
                "expected": _summary(expected),
                "observed": _summary(observed),
            }
        ]
    if isinstance(expected, dict):
        rows: list[dict[str, Any]] = []
        for key in sorted(set(expected) | set(observed)):
            if key not in expected or key not in observed:
                rows.append(
                    {
                        "path": f"{path}.{key}",
                        "expected": _summary(expected.get(key)),
                        "observed": _summary(observed.get(key)),
                    }
                )
            else:
                rows.extend(
                    _diff(
                        expected[key],
                        observed[key],
                        path=f"{path}.{key}",
                    )
                )
        return rows
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return [
                {
                    "path": f"{path}.length",
                    "expected": len(expected),
                    "observed": len(observed),
                }
            ]
        rows = []
        for index, (left, right) in enumerate(zip(expected, observed)):
            rows.extend(
                _diff(
                    left,
                    right,
                    path=f"{path}[{index}]",
                )
            )
        return rows
    if expected != observed:
        return [
            {
                "path": path,
                "expected": _summary(expected),
                "observed": _summary(observed),
            }
        ]
    return []


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
    if ROOT.exists():
        raise RuntimeError("diagnostic_root_not_fresh")
    ROOT.mkdir(mode=0o700)
    ROOT.chmod(0o700)
    target_raw = TARGET.read_bytes()
    target = gscl_binding._decode_target_manifest(  # noqa: SLF001
        target_raw
    )
    encoder = PortableOfflineMiniLMEncoder(
        asset_manifest_path=ASSET,
        model_root=MODEL,
        run_canary=True,
    )
    observed = gscl_binding._manifest_body(encoder)  # noqa: SLF001
    differences = _diff(target, observed)
    body: dict[str, Any] = {
        "api_evaluation_count": 0,
        "difference_count": len(differences),
        "differences": differences,
        "effect_gate_count": 0,
        "formal_measurement": False,
        "label_open_count": 0,
        "model_weight_load_count": 1,
        "network_call_count": 0,
        "official_source_access_count": 0,
        "observed_manifest_sha256": hashlib.sha256(
            gscl_binding._encode_manifest(observed)  # noqa: SLF001
        ).hexdigest(),
        "schema": "gscl_minilm_target_drift_diagnostic_v1",
        "scorer_call_count": 0,
        "source_content_supplied": False,
        "status": (
            "PASS_TARGET_MATCH"
            if not differences
            else "DIAGNOSED_TARGET_DRIFT"
        ),
        "target_manifest_file_sha256": hashlib.sha256(
            target_raw
        ).hexdigest(),
    }
    _write_once(
        OUTPUT,
        {**body, "self_sha256": _content_hash(body)},
    )
    print(
        json.dumps(
            {
                "difference_count": len(differences),
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
