"""Single-load, custody-bound closed-choice Qwen multi-pack execution.

The caller supplies only a canonical sequence of securely opened story-only
packs and exact local assets.  This module constructs the frozen
``LocalQwenClosedChoiceRuntime`` exactly once, keeps it inside this process,
and permits the model to rank only program-enumerated alternatives through
teacher-forced conditional log-likelihood.  It never calls free-form
generation and accepts no runtime, scorer, parser, completion, prediction, or
receipt from its caller.

The input envelope intentionally reuses the mechanism-neutral secure
multi-pack manifest schema.  The output schema, runtime preimages, selection
commitments, and execution closure are closed-choice specific.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

from . import closed_choice_qwen_runtime as qwen_closed
from . import closed_choice_worker as closed
from . import contract
from . import multi_pack_worker as support
from . import worker


VERSION = "gscl_narrative_closed_choice_multi_pack_worker_v1"
INPUT_MANIFEST_SCHEMA = support.INPUT_MANIFEST_SCHEMA
SAFE_RECEIPT_SCHEMA = f"{VERSION}.private_runtime_receipt.v1"
MAXIMUM_BATCH_COUNT = support.MAXIMUM_BATCH_COUNT
MultiPackWorkerError = support.MultiPackWorkerError


def _selection_commitment(
    commitments: Sequence[str],
) -> str:
    if (
        not isinstance(commitments, Sequence)
        or isinstance(commitments, (str, bytes))
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in commitments
        )
    ):
        raise MultiPackWorkerError(
            "closed_choice_selection_commitments_invalid"
        )
    return support._semantic_sha256(list(commitments))  # noqa: SLF001


def _teacher_forced_backend_commitment(
    runtime: qwen_closed.LocalQwenClosedChoiceRuntime,
) -> str:
    runtime._validate_formal_binding()
    value = runtime._teacher_forced_backend_commitment  # noqa: SLF001
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise MultiPackWorkerError(
            "closed_choice_teacher_forced_binding_invalid"
        )
    return value


def _batch_selection_commitments_sha256(
    batch_receipts: Sequence[Mapping[str, object]],
) -> str:
    rows = [
        {
            "selection_receipt_commitment": row[
                "selection_receipt_commitment"
            ],
            "selection_receipt_count": row[
                "selection_receipt_count"
            ],
            "sequence": row["sequence"],
        }
        for row in batch_receipts
    ]
    return support._semantic_sha256(rows)  # noqa: SLF001


def run_formal_multi_pack(
    *,
    input_manifest_path: Path,
    model_root: Path,
    model_manifest_path: Path,
    safe_receipt_path: Path,
) -> Mapping[str, object]:
    """Execute the exact closed-choice multi-pack ABI with one model load."""

    (
        work_root,
        batches,
        manifest_file_sha256,
        lineage,
    ) = support._decode_manifest(input_manifest_path)  # noqa: SLF001
    support._require_beneath(  # noqa: SLF001
        safe_receipt_path,
        work_root=work_root,
        final_must_exist=False,
    )
    manifest = worker.load_model_asset_manifest(
        manifest_path=model_manifest_path,
        model_root=model_root,
    )
    runtime = qwen_closed.LocalQwenClosedChoiceRuntime(
        model_root=model_root,
        manifest=manifest,
    )
    batch_receipts: list[dict[str, object]] = []
    all_selection_commitments: list[str] = []
    for row in batches:
        input_path = row["input_path"]
        output_path = row["output_path"]
        if (
            not isinstance(input_path, Path)
            or not isinstance(output_path, Path)
        ):
            raise MultiPackWorkerError(
                "multi_pack_internal_path_invalid"
            )
        pack = contract.load_trusted_story_only_input_pack(
            input_path
        )
        if (
            pack.sequence != row["sequence"]
            or pack.input_file_sha256 != row["input_file_sha256"]
        ):
            raise MultiPackWorkerError(
                "multi_pack_input_binding_changed"
            )
        selection_start = len(
            runtime.selection_receipt_commitments
        )
        started_ns = time.monotonic_ns()
        results = qwen_closed.process_formal_pack(
            pack,
            runtime=runtime,
        )
        elapsed_ns = time.monotonic_ns() - started_ns
        selection_commitments = (
            runtime.selection_receipt_commitments[
                selection_start:
            ]
        )
        contract.write_private_output_once(
            output_path,
            pack=pack,
            execution_closure=runtime.execution_closure,
            results=results,
        )
        output_read = contract.secure_read_file(
            output_path,
            maximum=contract.MAXIMUM_OUTPUT_BYTES,
        )
        decoded_output = contract.decode_private_output(
            output_read.raw,
            expected_pack=pack,
        )
        if decoded_output["execution_closure"] != (
            support._execution_closure_payload(  # noqa: SLF001
                runtime.execution_closure
            )
        ):
            raise MultiPackWorkerError(
                "multi_pack_execution_closure_changed"
            )
        valid_results = [
            result
            for result in decoded_output["results"]
            if result["generation_valid"] is True
        ]
        invalid_count = sum(
            result["generation_valid"] is False
            for result in decoded_output["results"]
        )
        if len(selection_commitments) != len(valid_results):
            raise MultiPackWorkerError(
                "closed_choice_selection_count_mismatch"
            )
        all_selection_commitments.extend(
            selection_commitments
        )
        token_counts = [
            result["completion_token_count"]
            for result in valid_results
        ]
        batch_receipts.append(
            {
                "batch_id": pack.batch_id,
                "decision_elapsed_ns": elapsed_ns,
                "decision_invalid_count": invalid_count,
                "decision_valid_count": len(valid_results),
                "input_file_sha256": pack.input_file_sha256,
                "input_pack_commitment": (
                    pack.input_pack_commitment
                ),
                "output_file_sha256": output_read.sha256,
                "selection_receipt_commitment": (
                    _selection_commitment(
                        selection_commitments
                    )
                ),
                "selection_receipt_count": len(
                    selection_commitments
                ),
                "sequence": pack.sequence,
                "story_count": len(pack.requests),
                "valid_wire_completion_token_count_maximum": (
                    max(token_counts, default=0)
                ),
                "valid_wire_completion_token_count_sum": sum(
                    token_counts
                ),
            }
        )

    runtime._validate_formal_binding()
    runtime_receipt = dict(runtime.runtime_receipt)
    double_run_receipt = dict(
        runtime.target_double_run_receipt
    )
    runtime_receipt_sha256 = (
        runtime._runtime_receipt_sha256  # noqa: SLF001
    )
    double_run_receipt_sha256 = (
        runtime._double_run_receipt_sha256  # noqa: SLF001
    )
    execution_closure = (
        support._execution_closure_payload(  # noqa: SLF001
            runtime.execution_closure
        )
    )
    logical_gpu_binding = (
        support._logical_gpu_binding(runtime)  # noqa: SLF001
    )
    if (
        runtime_receipt.get("free_form_generation_count") != 0
        or runtime_receipt.get("score_operation")
        != "teacher_forced_forward_log_softmax"
        or hashlib.sha256(
            contract.canonical_json_bytes(runtime_receipt)
        ).hexdigest()
        != runtime_receipt_sha256
        or double_run_receipt.get("free_form_generation_count")
        != 0
        or double_run_receipt.get("repeat_exact") is not True
        or double_run_receipt.get("repeat_count") != 2
        or hashlib.sha256(
            contract.canonical_json_bytes(double_run_receipt)
        ).hexdigest()
        != double_run_receipt_sha256
        or execution_closure[
            "target_double_run_receipt_sha256"
        ]
        != double_run_receipt_sha256
        or double_run_receipt.get("runtime_receipt_sha256")
        != runtime_receipt_sha256
        or tuple(all_selection_commitments)
        != runtime.selection_receipt_commitments
    ):
        raise MultiPackWorkerError(
            "multi_pack_runtime_preimage_mismatch"
        )
    distribution_rows = (
        support._loaded_distribution_rows()  # noqa: SLF001
    )
    body: dict[str, object] = {
        "batch_count": len(batch_receipts),
        "batches": batch_receipts,
        "claim_scope": closed.CLAIM_SCOPE,
        "execution_closure": execution_closure,
        "free_form_generation_count": 0,
        "input_manifest_file_sha256": (
            manifest_file_sha256
        ),
        "lineage": lineage,
        "loaded_distribution_closure_sha256": (
            support._semantic_sha256(  # noqa: SLF001
                distribution_rows
            )
        ),
        "loaded_distributions": distribution_rows,
        "logical_gpu_binding": logical_gpu_binding,
        "model_asset_manifest_file_sha256": (
            manifest.manifest_file_sha256
        ),
        "runtime_receipt": runtime_receipt,
        "runtime_receipt_sha256": runtime_receipt_sha256,
        "schema": SAFE_RECEIPT_SCHEMA,
        "score_operation": (
            "teacher_forced_forward_log_softmax"
        ),
        "selection_receipt_commitments_sha256": (
            _batch_selection_commitments_sha256(
                batch_receipts
            )
        ),
        "selection_receipt_count": len(
            all_selection_commitments
        ),
        "single_model_load_count": 1,
        "source_content_supplied": (
            lineage == "formal_measurement"
        ),
        "target_double_run_receipt": (
            double_run_receipt
        ),
        "target_double_run_receipt_sha256": (
            double_run_receipt_sha256
        ),
        "teacher_forced_backend_commitment": (
            _teacher_forced_backend_commitment(runtime)
        ),
        "worker_version": VERSION,
    }
    receipt = {
        **body,
        "self_sha256": support._semantic_sha256(body),  # noqa: SLF001
    }
    support._write_once(  # noqa: SLF001
        safe_receipt_path,
        contract.canonical_json_bytes(receipt),
        work_root=work_root,
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-manifest", required=True, type=Path
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--model-manifest", required=True, type=Path
    )
    parser.add_argument(
        "--safe-receipt", required=True, type=Path
    )
    arguments = parser.parse_args(argv)
    receipt = run_formal_multi_pack(
        input_manifest_path=arguments.input_manifest,
        model_root=arguments.model,
        model_manifest_path=arguments.model_manifest,
        safe_receipt_path=arguments.safe_receipt,
    )
    print(
        json.dumps(
            {
                "batch_count": receipt["batch_count"],
                "free_form_generation_count": 0,
                "runtime_receipt_sha256": receipt[
                    "runtime_receipt_sha256"
                ],
                "selection_receipt_count": receipt[
                    "selection_receipt_count"
                ],
                "status": "completed_closed_choice",
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
        MultiPackWorkerError,
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
    "INPUT_MANIFEST_SCHEMA",
    "MAXIMUM_BATCH_COUNT",
    "MultiPackWorkerError",
    "SAFE_RECEIPT_SCHEMA",
    "VERSION",
    "main",
    "run_formal_multi_pack",
]
