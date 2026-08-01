"""Source-free actual-asset qualification for the closed-choice Qwen path.

The command has no story/source argument.  It loads the verified local model
once, runs the runtime-owned fixed synthetic double-run canary, and writes a
safe commitment-only receipt with ``O_EXCL``.  It is not a study, measurement,
or efficacy gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Mapping, Sequence

from . import closed_choice_qwen_runtime as qwen_closed
from . import closed_choice_worker as closed
from . import contract
from . import worker


VERSION = "gscl_narrative_closed_choice_actual_qualification_v1"
SAFE_RECEIPT_SCHEMA = f"{VERSION}.safe_receipt.v1"


def _publish_once(path: Path, raw: bytes) -> str:
    """Publish a complete same-directory pending inode without replacement."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    if (
        not absolute.parent.exists()
        or absolute.parent.is_symlink()
        or not stat.S_ISDIR(absolute.parent.stat().st_mode)
        or absolute.exists()
        or absolute.is_symlink()
    ):
        raise closed.ClosedChoiceError(
            "closed_choice_qualification_output_invalid"
        )
    pending: Path | None = None
    descriptor: int | None = None
    for _ in range(32):
        candidate = absolute.with_name(
            f".{absolute.name}.pending-{secrets.token_hex(16)}"
        )
        try:
            descriptor = os.open(
                candidate,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError:
            continue
        pending = candidate
        break
    if pending is None or descriptor is None:
        raise closed.ClosedChoiceError(
            "closed_choice_qualification_pending_unavailable"
        )
    expected = (hashlib.sha256(raw).hexdigest(), len(raw))
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise closed.ClosedChoiceError(
                    "closed_choice_qualification_write_failed"
                )
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        read_descriptor = os.open(
            pending,
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            observed = worker._stable_file_hash_from_fd(
                read_descriptor,
                maximum=worker.MAXIMUM_MODEL_MANIFEST_BYTES,
            )
        finally:
            os.close(read_descriptor)
        if observed != expected:
            raise closed.ClosedChoiceError(
                "closed_choice_qualification_pending_changed"
            )
        pending_metadata = pending.stat(follow_symlinks=False)
        try:
            os.link(
                pending,
                absolute,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise closed.ClosedChoiceError(
                "closed_choice_qualification_output_invalid"
            ) from exc
        try:
            directory_descriptor = os.open(
                absolute.parent,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except Exception:
            try:
                final_metadata = absolute.stat(
                    follow_symlinks=False
                )
                if (
                    final_metadata.st_dev == pending_metadata.st_dev
                    and final_metadata.st_ino
                    == pending_metadata.st_ino
                ):
                    absolute.unlink()
            except FileNotFoundError:
                pass
            raise
        return expected[0]
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            pending.unlink()
        except FileNotFoundError:
            pass


def run_source_free_actual_qualification(
    *,
    model_root: Path,
    model_manifest_path: Path,
    output_path: Path,
) -> Mapping[str, object]:
    """Run the fixed internal canary; no caller-supplied text or scorer."""

    manifest = worker.load_model_asset_manifest(
        manifest_path=model_manifest_path,
        model_root=model_root,
    )
    runtime = qwen_closed.LocalQwenClosedChoiceRuntime(
        model_root=model_root,
        manifest=manifest,
    )
    runtime._validate_formal_binding()
    runtime_receipt = dict(runtime.runtime_receipt)
    double_run_receipt = dict(
        runtime.target_double_run_receipt
    )
    if (
        runtime_receipt.get("free_form_generation_count") != 0
        or double_run_receipt.get("free_form_generation_count")
        != 0
        or double_run_receipt.get("repeat_exact") is not True
        or double_run_receipt.get("repeat_count") != 2
    ):
        raise closed.ClosedChoiceError(
            "closed_choice_qualification_canary_invalid"
        )
    body: dict[str, object] = {
        "claim_scope": (
            "source_free_runtime_qualification_only_no_efficacy"
        ),
        "critical_dependency_closure_sha256": (
            contract.semantic_sha256(
                runtime_receipt["critical_dependency_closure"]
            )
        ),
        "execution_closure": runtime.execution_closure.payload(),
        "formal_measurement": False,
        "free_form_generation_count": 0,
        "model_asset_manifest_file_sha256": (
            manifest.manifest_file_sha256
        ),
        "runtime_receipt_sha256": hashlib.sha256(
            contract.canonical_json_bytes(runtime_receipt)
        ).hexdigest(),
        "schema": SAFE_RECEIPT_SCHEMA,
        "official_source_access_count": 0,
        "official_source_content_supplied_to_model": False,
        "public_synthetic_content_supplied_to_model": True,
        "status": "PASS_CLOSED_CHOICE_ACTUAL_SOURCE_FREE_CANARY",
        "target_double_run_receipt_sha256": hashlib.sha256(
            contract.canonical_json_bytes(double_run_receipt)
        ).hexdigest(),
        "teacher_forced_backend_commitment": (
            runtime._teacher_forced_backend_commitment
        ),
        "version": VERSION,
    }
    receipt = {
        **body,
        "self_sha256": contract.semantic_sha256(body),
    }
    raw = contract.canonical_json_bytes(receipt)
    # This is the final operation capable of invalidating the PASS claim.
    # Publish only after it succeeds; no later runtime assertion may leave a
    # misleading final receipt behind.
    runtime._validate_formal_binding()
    _publish_once(output_path, raw)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--model-manifest", required=True, type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    receipt = run_source_free_actual_qualification(
        model_root=arguments.model,
        model_manifest_path=arguments.model_manifest,
        output_path=arguments.output,
    )
    print(
        json.dumps(
            {
                "free_form_generation_count": 0,
                "self_sha256": receipt["self_sha256"],
                "status": receipt["status"],
                "target_double_run_receipt_sha256": receipt[
                    "target_double_run_receipt_sha256"
                ],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        closed.ClosedChoiceError,
        contract.NarrativeExtractorRuntimeError,
    ) as exc:
        issue_id = getattr(exc, "issue_id", "runtime_error")
        print(
            f"{VERSION} failed closed: {issue_id}",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


__all__ = [
    "SAFE_RECEIPT_SCHEMA",
    "VERSION",
    "run_source_free_actual_qualification",
]
