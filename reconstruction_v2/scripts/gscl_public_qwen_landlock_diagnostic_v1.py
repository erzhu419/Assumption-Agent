"""Fixed public-only diagnostic for the GSCL Qwen Landlock execution path."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import traceback

from replication_runtime.gscl_narrative_extractor_v1 import (
    closed_choice_multi_pack_worker as worker,
)
from replication_runtime.gscl_narrative_extractor_v1 import contract


ROOT = Path(
    "/var/tmp/"
    "gscl_closed_choice_public_landlock_diagnostic_20260730_r9"
)
INPUT_MANIFEST = ROOT / "manifest.json"
MODEL_ROOT = Path(
    "/var/tmp/"
    "gscl_closed_choice_actual_qualification_20260730/model_snapshot"
)
MODEL_MANIFEST = ROOT / "qwen.model.json"
RUNTIME_SAFE_RECEIPT = ROOT / "runtime.safe.json"
DIAGNOSTIC_SAFE_RECEIPT = ROOT / "diagnostic.safe.json"


def _write_once(path: Path, value: dict[str, object]) -> None:
    raw = contract.canonical_json_bytes(value)
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
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise RuntimeError("diagnostic_write_failed")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    if (
        not stat.S_ISDIR(ROOT.lstat().st_mode)
        or not stat.S_ISDIR(MODEL_ROOT.lstat().st_mode)
        or any(
            not stat.S_ISREG(path.lstat().st_mode)
            for path in (INPUT_MANIFEST, MODEL_MANIFEST)
        )
        or RUNTIME_SAFE_RECEIPT.exists()
        or DIAGNOSTIC_SAFE_RECEIPT.exists()
    ):
        raise RuntimeError("public_diagnostic_topology_invalid")
    try:
        # ``torch.cuda.is_available`` intentionally suppresses the underlying
        # driver exception.  Initialising explicitly preserves that public
        # infrastructure cause if the Landlock allowlist is incomplete.
        import torch

        torch.cuda.init()
        receipt = worker.run_formal_multi_pack(
            input_manifest_path=INPUT_MANIFEST,
            model_root=MODEL_ROOT,
            model_manifest_path=MODEL_MANIFEST,
            safe_receipt_path=RUNTIME_SAFE_RECEIPT,
        )
    except BaseException as exc:  # diagnostic must preserve the root cause
        causes: list[dict[str, object]] = []
        current: BaseException | None = exc
        for depth in range(8):
            if current is None:
                break
            causes.append(
                {
                    "depth": depth,
                    "frames": [
                        {
                            "filename": frame.filename,
                            "function": frame.name,
                            "line": frame.lineno,
                        }
                        for frame in traceback.extract_tb(
                            current.__traceback__
                        )
                    ],
                    "message": str(current),
                    "module": type(current).__module__,
                    "type": type(current).__name__,
                }
            )
            current = current.__cause__
        body: dict[str, object] = {
            "causes": causes,
            "effect_gate_added": False,
            "formal_measurement": False,
            "official_source_access_count": 0,
            "public_synthetic_only": True,
            "runtime_safe_receipt_created": (
                RUNTIME_SAFE_RECEIPT.exists()
            ),
            "schema": (
                "gscl_public_qwen_landlock_diagnostic_v1."
                "safe_receipt.v1"
            ),
            "status": "PUBLIC_SOURCE_FREE_CAUSE_CAPTURED",
        }
    else:
        body = {
            "batch_count": receipt["batch_count"],
            "effect_gate_added": False,
            "formal_measurement": False,
            "free_form_generation_count": 0,
            "official_source_access_count": 0,
            "public_synthetic_only": True,
            "runtime_safe_receipt_file_sha256": hashlib.sha256(
                RUNTIME_SAFE_RECEIPT.read_bytes()
            ).hexdigest(),
            "runtime_safe_receipt_self_sha256": receipt[
                "self_sha256"
            ],
            "schema": (
                "gscl_public_qwen_landlock_diagnostic_v1."
                "safe_receipt.v1"
            ),
            "selection_receipt_count": receipt[
                "selection_receipt_count"
            ],
            "status": "PUBLIC_SOURCE_FREE_LANDLOCK_PATH_PASS",
        }
    safe = {
        **body,
        "self_sha256": contract.semantic_sha256(body),
    }
    _write_once(DIAGNOSTIC_SAFE_RECEIPT, safe)
    print(
        json.dumps(
            {
                "self_sha256": safe["self_sha256"],
                "status": safe["status"],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
    # Avoid interpreter-shutdown CUDA finalizers retaining a diagnostic
    # process after its safe receipt has been durably published.
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
