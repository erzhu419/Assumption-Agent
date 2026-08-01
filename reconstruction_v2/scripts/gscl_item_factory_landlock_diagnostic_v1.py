"""Exact source-free Landlock canary for the GSCL ARN item factory.

The inputs are the public-synthetic extractor outputs already sealed by the
failed qualification attempt.  The command opens no official ARN source or
labels, invokes no scorer, and writes only a safe aggregate terminal plus
private diagnostic output under a fresh ext4 root.
"""

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
FAILED_ROOT = (
    BASE
    / "work/source_free_internal_factory_qualification_ext4_repair_r2"
)
ONE_SHOT_KEY = (
    "ac5306008885f8f9f79f48dd05055cd1f672a554862c50728d354c4d09cf4546"
)
FAILED_ITEM_ROOT = (
    FAILED_ROOT / f"work/formal_factory/{ONE_SHOT_KEY}/item_factory"
)
ROOT = BASE / "work/item_factory_landlock_ext4_target_diagnostic_r3"
PRIVATE_ROOT = (
    BASE / "work/item_factory_landlock_ext4_target_diagnostic_r3_private"
)
TARGET_SOURCE = (
    BASE
    / "assets/minilm_target_qualification_ext4_r1/target_manifest.json"
)
TARGET_FILE_SHA256 = (
    "ea4054549bd22fe53029568b69e8652589f585a54985c54f09207ee3e6149d0e"
)
ITEM_FACTORY_SHA256 = (
    "7711c0e52b58a916a8985140a3844adfa6b55a227d8cc06ff10096ac8cea28b3"
)
PREDICTOR_PACK_SHA256 = (
    "ef998c7cb8105792fa78a2a5c782c42f9d8adc673fa1c1a7f0acbd69d94b3187"
)
ACTION_SELF_HASH = (
    "23523283d53df1ce3a1d67264075155208cf40f6a1b7c7789f06a5bce8bbea41"
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
    output = ROOT / "private_four_arm.json"
    receipt = ROOT / "item_factory.sandbox.safe.json"
    spec_path = ROOT / "item_factory.sandbox.spec.json"
    command = [
        str(
            BASE
            / "assets/gscl_runtime_ext4_v1/typed_venv/bin/python"
        ),
        "-m",
        "assumption_agent.benchmarks.gscl_arn_formal_item_factory_v1",
        "--predictor",
        str(FAILED_ITEM_ROOT / "predictor.json"),
        "--batch-manifest",
        str(FAILED_ITEM_ROOT / "extractor_batches.json"),
        "--minilm-manifest",
        str(BASE / "assets/minilm_asset_manifest.json"),
        "--minilm-model",
        str(BASE / "assets/minilm_model"),
        "--minilm-target-manifest",
        str(target),
        "--output",
        str(output),
    ]
    spec_body: dict[str, Any] = {
        "schema": supervisor.SANDBOX_SPEC_SCHEMA,
        "arm_id": "item_factory_diagnostic",
        "one_shot_key": ONE_SHOT_KEY,
        "action_self_hash": ACTION_SELF_HASH,
        "command": command,
        "implementation_sha256": ITEM_FACTORY_SHA256,
        "code_roots": [
            str(BASE / "code"),
            str(BASE / "code/reconstruction_v2"),
            str(BASE / "code/assumption_os"),
            str(BASE / "assets/gscl_runtime_ext4_v1/python310"),
            str(BASE / "assets/gscl_runtime_ext4_v1/typed_venv"),
            str(FAILED_ITEM_ROOT),
        ],
        "model_roots": [
            str(BASE / "assets/minilm_model"),
            str(BASE / "assets/minilm_asset_manifest.json"),
        ],
        "work_root": str(ROOT),
        "predictor_pack_sha256": PREDICTOR_PACK_SHA256,
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
    output_sha256 = (
        hashlib.sha256(output.read_bytes()).hexdigest()
        if output.exists()
        else None
    )
    terminal_body: dict[str, Any] = {
        "api_evaluation_count": 0,
        "effect_gate_count": 0,
        "formal_measurement": False,
        "item_content_emitted": False,
        "label_open_count": 0,
        "model_weight_load_count": 1,
        "network_call_count": 0,
        "official_source_access_count": 0,
        "output_file_sha256": output_sha256,
        "sandbox_receipt_self_hash": safe.get("self_hash"),
        "sandbox_status": safe.get("status"),
        "schema": "gscl_item_factory_landlock_diagnostic_v1",
        "scorer_call_count": 0,
        "source_content_supplied": False,
        "status": (
            "PASS_ITEM_FACTORY_EXT4_TARGET_LANDLOCK"
            if exit_code == 0
            else "FAIL_ITEM_FACTORY_EXT4_TARGET_LANDLOCK"
        ),
        "target_manifest_file_sha256": TARGET_FILE_SHA256,
    }
    terminal = {
        **terminal_body,
        "self_sha256": _content_hash(terminal_body),
    }
    _write_json_once(ROOT / "terminal.safe.json", terminal)
    print(
        json.dumps(
            {
                "sandbox_status": terminal["sandbox_status"],
                "status": terminal["status"],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
