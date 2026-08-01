"""Launch the public-synthetic MiniLM target diff inside exact Landlock."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from assumption_agent.benchmarks import (
    gscl_arn_formal_supervisor_v1 as supervisor,
)


BASE = Path("/var/tmp/gscl_unified_nonscoring_harness_20260730")
ROOT = BASE / "work/minilm_landlock_target_drift_diagnostic_r3"
PRIVATE_ROOT = (
    BASE / "work/minilm_landlock_target_drift_diagnostic_r3_private"
)
TARGET_SOURCE = (
    BASE
    / "assets/minilm_target_qualification_ext4_r1/target_manifest.json"
)
TARGET_FILE_SHA256 = (
    "ea4054549bd22fe53029568b69e8652589f585a54985c54f09207ee3e6149d0e"
)
WORKER = (
    BASE
    / "code/reconstruction_v2/scripts/"
    "gscl_minilm_target_drift_worker_v1.py"
)
WORKER_SHA256 = (
    "55ee22695d6c622072baec85d2729647bd30ac921c88de6d30afcb8771cf7dea"
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


def _write_bytes_once(path: Path, raw: bytes) -> None:
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


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    _write_bytes_once(path, _canonical_bytes(value))


def main() -> int:
    if ROOT.exists() or PRIVATE_ROOT.exists():
        raise RuntimeError("diagnostic_topology_not_fresh")
    ROOT.mkdir(mode=0o700)
    PRIVATE_ROOT.mkdir(mode=0o700)
    ROOT.chmod(0o700)
    PRIVATE_ROOT.chmod(0o700)
    target_raw = TARGET_SOURCE.read_bytes()
    if hashlib.sha256(target_raw).hexdigest() != TARGET_FILE_SHA256:
        raise RuntimeError("target_manifest_changed")
    target = ROOT / "minilm.target.json"
    _write_bytes_once(target, target_raw)
    label_probe = PRIVATE_ROOT / "labels.json"
    linkage_probe = PRIVATE_ROOT / "linkage.json"
    _write_json_once(label_probe, {"probe": "label"})
    _write_json_once(linkage_probe, {"probe": "linkage"})
    output = ROOT / "drift.safe.json"
    receipt = ROOT / "sandbox.safe.json"
    spec_path = ROOT / "sandbox.spec.json"
    command = [
        str(
            BASE
            / "assets/gscl_runtime_ext4_v1/typed_venv/bin/python"
        ),
        "-B",
        str(WORKER),
        "--asset",
        str(BASE / "assets/minilm_asset_manifest.json"),
        "--model",
        str(BASE / "assets/minilm_model"),
        "--target",
        str(target),
        "--output",
        str(output),
    ]
    spec_body: dict[str, Any] = {
        "schema": supervisor.SANDBOX_SPEC_SCHEMA,
        "arm_id": "minilm_target_drift_diagnostic",
        "one_shot_key": "0" * 64,
        "action_self_hash": "1" * 64,
        "command": command,
        "implementation_sha256": WORKER_SHA256,
        "code_roots": [
            str(BASE / "code"),
            str(BASE / "code/reconstruction_v2"),
            str(BASE / "code/assumption_os"),
            str(BASE / "assets/gscl_runtime_ext4_v1/python310"),
            str(BASE / "assets/gscl_runtime_ext4_v1/typed_venv"),
        ],
        "model_roots": [
            str(BASE / "assets/minilm_model"),
            str(BASE / "assets/minilm_asset_manifest.json"),
        ],
        "work_root": str(ROOT),
        "predictor_pack_sha256": "2" * 64,
        "prediction_output_path": str(output),
        "sandbox_receipt_path": str(receipt),
        "private_denial_probes": {
            "label_pack": str(label_probe),
            "linkage_pack": str(linkage_probe),
        },
        "gpu_device_index": None,
        "environment_overrides": {"CUDA_VISIBLE_DEVICES": ""},
    }
    spec = {**spec_body, "self_hash": _content_hash(spec_body)}
    _write_json_once(spec_path, spec)
    exit_code = supervisor._sandbox_child(spec_path)  # noqa: SLF001
    safe = json.loads(receipt.read_text(encoding="ascii"))
    drift = (
        json.loads(output.read_text(encoding="ascii"))
        if output.exists()
        else None
    )
    body: dict[str, Any] = {
        "api_evaluation_count": 0,
        "difference_count": (
            drift.get("difference_count")
            if isinstance(drift, dict)
            else None
        ),
        "drift_receipt_file_sha256": (
            hashlib.sha256(output.read_bytes()).hexdigest()
            if output.exists()
            else None
        ),
        "drift_receipt_self_sha256": (
            drift.get("self_sha256")
            if isinstance(drift, dict)
            else None
        ),
        "effect_gate_count": 0,
        "formal_measurement": False,
        "label_open_count": 0,
        "model_weight_load_count": 1,
        "network_call_count": 0,
        "official_source_access_count": 0,
        "sandbox_receipt_self_hash": safe.get("self_hash"),
        "sandbox_status": safe.get("status"),
        "schema": "gscl_minilm_landlock_drift_launcher_v1",
        "scorer_call_count": 0,
        "source_content_supplied": False,
        "status": (
            "PASS_LANDLOCK_TARGET_MATCH"
            if exit_code == 0
            and isinstance(drift, dict)
            and drift.get("difference_count") == 0
            else "DIAGNOSED_LANDLOCK_TARGET_DRIFT"
        ),
        "target_manifest_file_sha256": TARGET_FILE_SHA256,
    }
    _write_json_once(
        ROOT / "terminal.safe.json",
        {**body, "self_sha256": _content_hash(body)},
    )
    print(
        json.dumps(
            {
                "difference_count": body["difference_count"],
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
