"""Exact source-free pre-root diagnostic for the GSCL internal factory.

This command is qualification-only.  It cannot create the future qualification
root, open the official ARN source, invoke a scorer, or authorize an effect
measurement.
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
from assumption_agent.benchmarks import (
    gscl_arn_internal_factory_qualification_v1 as runner,
)
from assumption_agent.benchmarks import (
    gscl_arn_formal_item_factory_v1 as item_factory,
)
from assumption_agent.benchmarks import (
    gscl_arn_intrinsic_protocol_v1 as protocol,
)


FUTURE_ROOT = Path(
    "/var/tmp/gscl_unified_nonscoring_harness_20260730/work/"
    "source_free_internal_factory_qualification_ext4_repair_r4"
)
OUTPUT_PATH = Path(
    "/var/tmp/gscl_unified_nonscoring_harness_20260730/work/"
    "source_free_internal_factory_preroot_ext4_diagnostic_r4/"
    "terminal.safe.json"
)
R3_FACTORY_OUTPUT = Path(
    "/var/tmp/gscl_unified_nonscoring_harness_20260730/work/"
    "source_free_internal_factory_qualification_ext4_repair_r3/"
    "work/formal_factory/"
    "d948b551da1e1247551146ad81476579c011fd88a6562d38a445696296853e57/"
    "item_factory/private_four_arm.json"
)
R3_FACTORY_OUTPUT_FILE_SHA256 = (
    "6dbea75d99a9c6208489150c9ddb1cadb4807800729928eb6394095f3e5d0466"
)
ITEM_FACTORY_LANDLOCK_TERMINAL = Path(
    "/var/tmp/gscl_unified_nonscoring_harness_20260730/work/"
    "item_factory_landlock_ext4_target_diagnostic_r3/terminal.safe.json"
)
ITEM_FACTORY_LANDLOCK_TERMINAL_FILE_SHA256 = (
    "7e7b9015a1d3cdc22c2cebca791115d1143b96cf90530e4552d0597a76f7d83e"
)
ITEM_FACTORY_LANDLOCK_TERMINAL_SELF_SHA256 = (
    "86f50cfba2737d6cf14ff277e136be3655f61d5a5fb3c8a884b47dc182288cff"
)
ITEM_FACTORY_LANDLOCK_SANDBOX = Path(
    "/var/tmp/gscl_unified_nonscoring_harness_20260730/work/"
    "item_factory_landlock_ext4_target_diagnostic_r3/"
    "item_factory.sandbox.safe.json"
)
ITEM_FACTORY_LANDLOCK_SANDBOX_FILE_SHA256 = (
    "a5239eead878dd5a746f5e1cc154993c75fc7eb206aa1e5b1b4c5d78389e3272"
)
ITEM_FACTORY_LANDLOCK_SANDBOX_SELF_SHA256 = (
    "9e705852f5801f67b8193a7522a634a5a0b48dd3d4c28026113b82862bd6270f"
)
QWEN_MODEL_ROOT = Path(
    "/var/tmp/gscl_closed_choice_actual_qualification_20260730/"
    "model_snapshot"
)
QWEN_MODEL_MANIFEST = Path(
    "/var/tmp/gscl_closed_choice_actual_qualification_20260730/"
    "qwen_manifest.json"
)
QWEN_CANARY_LINEAGE = (
    runner.FROZEN_CODE_ROOT
    / (
        "manifests/"
        "gscl_closed_choice_actual_canary_"
        "lineage_terminal_ext4_20260730.json"
    )
)
MINILM_MODEL_ROOT = (
    runner.FROZEN_DEPLOYMENT_ROOT / "assets/minilm_model"
)
MINILM_ASSET_MANIFEST = (
    runner.FROZEN_DEPLOYMENT_ROOT
    / "assets/minilm_asset_manifest.json"
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
    if FUTURE_ROOT.exists() or OUTPUT_PATH.exists():
        raise RuntimeError("preroot_topology_not_fresh")
    r3_factory_raw = supervisor._read_regular_absolute_exact(  # noqa: SLF001
        R3_FACTORY_OUTPUT,
        expected_sha256=R3_FACTORY_OUTPUT_FILE_SHA256,
        maximum_bytes=2 * 1024 * 1024,
    )
    r3_factory = item_factory._decode_canonical_object(  # noqa: SLF001
        r3_factory_raw,
        issue_id="r3_factory_output_invalid",
    )
    r3_item_ids = {
        row["opaque_item_id"]
        for predictions in r3_factory.get("by_arm", {}).values()
        for row in predictions
    }
    r3_normalized = supervisor._validate_factory_output_receipt(  # noqa: SLF001
        r3_factory,
        expected_schema=item_factory.PRIVATE_OUTPUT_SCHEMA,
        expected_lineage="formal_frozen_assets",
        expected_predictor_file_sha256=r3_factory[
            "predictor_pack_file_sha256"
        ],
        expected_batch_receipts=r3_factory[
            "extractor_batch_receipts"
        ],
        expected_item_ids=r3_item_ids,
    )
    supervisor._validate_factory_encoder_binding(  # noqa: SLF001
        r3_factory["encoder_binding"],
        expected_target_file_sha256=(
            runner.FROZEN_MINILM_TARGET_FILE_SHA256
        ),
        expected_target_self_sha256=(
            runner.FROZEN_MINILM_TARGET_SELF_SHA256
        ),
    )
    post_factory_pack_hashes = [
        supervisor._content_hash(  # noqa: SLF001
            {
                "arm_id": arm_id,
                "predictions": r3_normalized[arm_id],
            }
        )
        for arm_id in protocol.ARM_IDS
    ]
    item_terminal_raw = supervisor._read_regular_absolute_exact(  # noqa: SLF001
        ITEM_FACTORY_LANDLOCK_TERMINAL,
        expected_sha256=ITEM_FACTORY_LANDLOCK_TERMINAL_FILE_SHA256,
        maximum_bytes=64 * 1024,
    )
    item_terminal = supervisor._parse_json(  # noqa: SLF001
        item_terminal_raw,
        issue_id="item_factory_landlock_terminal_invalid",
    )
    item_terminal_body = dict(item_terminal)
    item_terminal_claimed = item_terminal_body.pop("self_sha256", None)
    if (
        item_terminal_claimed
        != ITEM_FACTORY_LANDLOCK_TERMINAL_SELF_SHA256
        or supervisor._content_hash(item_terminal_body)  # noqa: SLF001
        != item_terminal_claimed
        or item_terminal.get("status")
        != "PASS_ITEM_FACTORY_EXT4_TARGET_LANDLOCK"
        or item_terminal.get("sandbox_status")
        != "LANDLOCK_ARM_COMPLETED"
        or item_terminal.get("source_content_supplied") is not False
        or item_terminal.get("formal_measurement") is not False
        or item_terminal.get("official_source_access_count") != 0
        or item_terminal.get("scorer_call_count") != 0
        or item_terminal.get("api_evaluation_count") != 0
    ):
        raise RuntimeError("item_factory_landlock_terminal_invalid")
    item_sandbox_raw = supervisor._read_regular_absolute_exact(  # noqa: SLF001
        ITEM_FACTORY_LANDLOCK_SANDBOX,
        expected_sha256=ITEM_FACTORY_LANDLOCK_SANDBOX_FILE_SHA256,
        maximum_bytes=64 * 1024,
    )
    item_sandbox = supervisor._parse_json(  # noqa: SLF001
        item_sandbox_raw,
        issue_id="item_factory_landlock_sandbox_invalid",
    )
    item_sandbox_body = dict(item_sandbox)
    item_sandbox_claimed = item_sandbox_body.pop("self_hash", None)
    if (
        item_sandbox_claimed
        != ITEM_FACTORY_LANDLOCK_SANDBOX_SELF_SHA256
        or supervisor._content_hash(item_sandbox_body)  # noqa: SLF001
        != item_sandbox_claimed
        or item_sandbox.get("status") != "LANDLOCK_ARM_COMPLETED"
        or item_sandbox.get("arm_exit_code") != 0
        or item_sandbox.get("label_denial_errno") not in {1, 13}
        or item_sandbox.get("linkage_denial_errno") not in {1, 13}
    ):
        raise RuntimeError("item_factory_landlock_sandbox_invalid")
    runner._preflight_exact_assets(  # noqa: SLF001
        root=FUTURE_ROOT,
        qwen_model_root=QWEN_MODEL_ROOT,
        qwen_model_manifest=QWEN_MODEL_MANIFEST,
        qwen_actual_canary_lineage_terminal=QWEN_CANARY_LINEAGE,
        minilm_model_root=MINILM_MODEL_ROOT,
        minilm_asset_manifest=MINILM_ASSET_MANIFEST,
        minilm_target_manifest=runner.FROZEN_MINILM_TARGET_MANIFEST,
    )
    runner._preflight_frozen_main_runtime()  # noqa: SLF001
    runner._preflight_frozen_runtime_binding_manifest()  # noqa: SLF001
    runner._preflight_fixed_source_free_test_runtime()  # noqa: SLF001
    runner._preflight_exactly_two_idle_gpus()  # noqa: SLF001
    test_attestation = supervisor.run_source_free_tests(
        code_root=supervisor._RECONSTRUCTION_ROOT,  # noqa: SLF001
        test_files=tuple(
            supervisor._INTERNAL_QUALIFICATION_TEST_PATHS.values()  # noqa: SLF001
        ),
        deselected_test_nodes=runner.SOURCE_FREE_DESELECTED_TEST_NODES,
        test_python=runner.FROZEN_TEST_PYTHON,
        pytest_wheel_bundle_manifest=(
            runner.FROZEN_PYTEST_WHEEL_BUNDLE_MANIFEST
        ),
    )
    runner._validate_fixed_test_attestation(  # noqa: SLF001
        test_attestation
    )
    outer_systemd_attestation = (
        supervisor._attest_current_outer_systemd_service(  # noqa: SLF001
            writable_root=OUTPUT_PATH.parent
        )
    )
    qwen_manifest_sha256 = supervisor._hash_regular_absolute(  # noqa: SLF001
        QWEN_MODEL_MANIFEST
    )
    lineage_raw = supervisor._read_regular_absolute_exact(  # noqa: SLF001
        QWEN_CANARY_LINEAGE,
        expected_sha256=(
            supervisor.
            CLOSED_CHOICE_ACTUAL_CANARY_LINEAGE_FILE_SHA256
        ),
        maximum_bytes=64 * 1024,
    )
    lineage = supervisor._validate_closed_choice_actual_canary_lineage(  # noqa: SLF001
        lineage_raw,
        expected_model_manifest_sha256=qwen_manifest_sha256,
    )
    closure = supervisor.attest_runtime_closure(
        code_roots=(supervisor._RECONSTRUCTION_ROOT,),  # noqa: SLF001
        entry_files=(
            Path(runner.__file__).resolve(),
            *tuple(
                supervisor.
                _INTERNAL_FORMAL_IMPLEMENTATION_PATHS.values()  # noqa: SLF001
            ),
        ),
        config_files=(
            QWEN_MODEL_MANIFEST,
            QWEN_CANARY_LINEAGE,
            MINILM_ASSET_MANIFEST,
            runner.FROZEN_MINILM_TARGET_MANIFEST,
            ITEM_FACTORY_LANDLOCK_TERMINAL,
            ITEM_FACTORY_LANDLOCK_SANDBOX,
            R3_FACTORY_OUTPUT,
            runner.FROZEN_RUNTIME_BINDING_MANIFEST,
            runner.FROZEN_PYTEST_WHEEL_BUNDLE_MANIFEST,
        ),
        asset_roots=(QWEN_MODEL_ROOT, MINILM_MODEL_ROOT),
        test_attestation=test_attestation,
        support_module_files=(  # noqa: SLF001
            supervisor._INTERNAL_SUPPORT_MODULE_PATHS
        ),
    )
    serialized_closure = json.dumps(
        closure.manifest,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    if (
        "/home/" in serialized_closure
        or FUTURE_ROOT.exists()
        or closure.manifest.get("source_content_supplied") is not False
        or closure.manifest.get("formal_measurement_run") is not False
    ):
        raise RuntimeError("preroot_closure_boundary_invalid")
    body: dict[str, Any] = {
        "api_evaluation_count": 0,
        "effect_gate_count": 0,
        "formal_measurement": False,
        "future_qualification_root_created": False,
        "item_factory_landlock_sandbox_self_hash": (
            item_sandbox_claimed
        ),
        "item_factory_landlock_terminal_self_hash": (
            item_terminal_claimed
        ),
        "lineage_model_weight_load_count": lineage[
            "lineage_model_weight_load_count"
        ],
        "official_source_access_count": 0,
        "outer_systemd_attestation_self_hash": (
            outer_systemd_attestation["self_hash"]
        ),
        "outer_systemd_stable_binding_sha256": (
            outer_systemd_attestation["stable_binding_sha256"]
        ),
        "post_factory_canonicalization_pack_count": len(
            post_factory_pack_hashes
        ),
        "public_synthetic_factory_output_validated": True,
        "runtime_closure_self_hash": closure.manifest["self_hash"],
        "runtime_root_count": len(closure.manifest["runtime_roots"]),
        "schema": "gscl_internal_factory_preroot_ext4_diagnostic_v1",
        "scorer_call_count": 0,
        "source_content_supplied": False,
        "status": "PASS_EXACT_SOURCE_FREE_PREROOT_CLOSURE",
        "test_attestation_self_hash": test_attestation.receipt[
            "self_hash"
        ],
        "worker_sha256": lineage["worker_sha256"],
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
