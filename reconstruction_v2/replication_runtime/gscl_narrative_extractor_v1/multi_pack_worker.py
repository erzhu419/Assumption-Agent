"""Single-load, custody-bound multi-pack Qwen execution.

This wrapper is part of the trusted formal implementation.  It loads the
frozen Qwen runtime exactly once, processes a canonical sequence of securely
opened story-only packs, writes every private output once, and persists the
complete non-story runtime/double-run preimages plus a content closure for
all loaded Python distributions.  It accepts no runtime object, parser,
completion, prediction, or receipt from its caller.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time
from typing import Mapping, Sequence

from . import contract
from . import worker


VERSION = "gscl_narrative_extractor_multi_pack_worker_v1"
INPUT_MANIFEST_SCHEMA = f"{VERSION}.input_manifest.v1"
SAFE_RECEIPT_SCHEMA = f"{VERSION}.private_runtime_receipt.v1"
MAXIMUM_MANIFEST_BYTES = 4 * 1024 * 1024
MAXIMUM_BATCH_COUNT = 4_096
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CRITICAL_DISTRIBUTIONS = frozenset(
    {
        "huggingface-hub",
        "numpy",
        "safetensors",
        "tokenizers",
        "torch",
        "transformers",
    }
)


class MultiPackWorkerError(RuntimeError):
    def __init__(self, issue_id: str) -> None:
        self.issue_id = issue_id
        super().__init__(issue_id)


def _normalise_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _semantic_sha256(value: object) -> str:
    return hashlib.sha256(
        contract.canonical_json_bytes(value, newline=False)
    ).hexdigest()


def _exact_object(
    value: object, keys: frozenset[str], issue_id: str
) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise MultiPackWorkerError(issue_id)
    return dict(value)


def _safe_absolute(value: object, *, issue_id: str) -> Path:
    if (
        not isinstance(value, str)
        or "\x00" in value
        or not Path(value).is_absolute()
        or os.path.abspath(value) != value
    ):
        raise MultiPackWorkerError(issue_id)
    return Path(value)


def _require_beneath(
    path: Path,
    *,
    work_root: Path,
    final_must_exist: bool,
) -> None:
    try:
        relative = path.relative_to(work_root)
    except ValueError as exc:
        raise MultiPackWorkerError(
            "multi_pack_path_outside_work"
        ) from exc
    if not relative.parts:
        raise MultiPackWorkerError("multi_pack_path_is_work_root")
    current = work_root
    components = relative.parts if final_must_exist else relative.parts[:-1]
    for component in components:
        if component in {"", ".", ".."}:
            raise MultiPackWorkerError(
                "multi_pack_path_component_invalid"
            )
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise MultiPackWorkerError(
                "multi_pack_path_component_missing"
            ) from exc
        is_final = current == path
        if current.is_symlink():
            raise MultiPackWorkerError("multi_pack_path_symlink")
        if is_final:
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise MultiPackWorkerError(
                    "multi_pack_input_topology_invalid"
                )
        elif (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise MultiPackWorkerError(
                "multi_pack_parent_topology_invalid"
            )
    if not final_must_exist and (
        path.exists() or path.is_symlink()
    ):
        raise MultiPackWorkerError("multi_pack_output_already_exists")


def _decode_manifest(
    path: Path,
) -> tuple[Path, tuple[dict[str, object], ...], str, str]:
    read = contract.secure_read_file(
        path, maximum=MAXIMUM_MANIFEST_BYTES
    )
    try:
        decoded = json.loads(
            read.raw.decode("ascii"),
            parse_float=lambda _: (_ for _ in ()).throw(
                MultiPackWorkerError("multi_pack_float_forbidden")
            ),
            parse_constant=lambda _: (_ for _ in ()).throw(
                MultiPackWorkerError("multi_pack_constant_forbidden")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise MultiPackWorkerError(
            "multi_pack_manifest_invalid"
        ) from exc
    if contract.canonical_json_bytes(decoded) != read.raw:
        raise MultiPackWorkerError(
            "multi_pack_manifest_not_canonical"
        )
    envelope = _exact_object(
        decoded,
        frozenset(
            {
                "batches",
                "lineage",
                "schema",
                "self_sha256",
                "work_root",
            }
        ),
        "multi_pack_manifest_fields_invalid",
    )
    body = {
        key: value
        for key, value in envelope.items()
        if key != "self_sha256"
    }
    if (
        envelope["schema"] != INPUT_MANIFEST_SCHEMA
        or envelope["lineage"]
        not in {"formal_measurement", "source_free_qualification"}
        or not isinstance(envelope["self_sha256"], str)
        or _SHA256.fullmatch(envelope["self_sha256"]) is None
        or _semantic_sha256(body) != envelope["self_sha256"]
    ):
        raise MultiPackWorkerError(
            "multi_pack_manifest_identity_invalid"
        )
    work_root = _safe_absolute(
        envelope["work_root"], issue_id="multi_pack_work_root_invalid"
    )
    root_metadata = work_root.lstat()
    if (
        work_root.is_symlink()
        or not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid != os.getuid()
        or stat.S_IMODE(root_metadata.st_mode) != 0o700
    ):
        raise MultiPackWorkerError("multi_pack_work_root_invalid")
    try:
        path.relative_to(work_root)
    except ValueError as exc:
        raise MultiPackWorkerError(
            "multi_pack_manifest_outside_work"
        ) from exc
    batches_raw = envelope["batches"]
    if (
        type(batches_raw) is not list
        or not 1 <= len(batches_raw) <= MAXIMUM_BATCH_COUNT
    ):
        raise MultiPackWorkerError("multi_pack_batch_count_invalid")
    batches: list[dict[str, object]] = []
    input_paths: set[Path] = set()
    output_paths: set[Path] = set()
    previous_sequence = -1
    for raw_row in batches_raw:
        row = _exact_object(
            raw_row,
            frozenset(
                {
                    "input_file_sha256",
                    "input_path",
                    "output_path",
                    "sequence",
                }
            ),
            "multi_pack_batch_fields_invalid",
        )
        if (
            isinstance(row["sequence"], bool)
            or not isinstance(row["sequence"], int)
            or not 0 <= row["sequence"] <= 9_999_999_999
            or row["sequence"] <= previous_sequence
            or not isinstance(row["input_file_sha256"], str)
            or _SHA256.fullmatch(row["input_file_sha256"]) is None
        ):
            raise MultiPackWorkerError(
                "multi_pack_batch_binding_invalid"
            )
        input_path = _safe_absolute(
            row["input_path"], issue_id="multi_pack_input_path_invalid"
        )
        output_path = _safe_absolute(
            row["output_path"],
            issue_id="multi_pack_output_path_invalid",
        )
        _require_beneath(
            input_path, work_root=work_root, final_must_exist=True
        )
        _require_beneath(
            output_path, work_root=work_root, final_must_exist=False
        )
        if (
            input_path in input_paths
            or output_path in output_paths
            or input_path == output_path
        ):
            raise MultiPackWorkerError(
                "multi_pack_path_duplicate"
            )
        input_paths.add(input_path)
        output_paths.add(output_path)
        previous_sequence = row["sequence"]
        batches.append(
            {
                "input_file_sha256": row["input_file_sha256"],
                "input_path": input_path,
                "output_path": output_path,
                "sequence": row["sequence"],
            }
        )
    return (
        work_root,
        tuple(batches),
        read.sha256,
        envelope["lineage"],
    )


def _loaded_distribution_rows() -> list[dict[str, object]]:
    """Hash every currently loaded non-stdlib distribution, plus criticals."""

    package_map = importlib.metadata.packages_distributions()
    top_level_modules = {
        name.partition(".")[0]
        for name, module in tuple(sys.modules.items())
        if name
        and module is not None
        and isinstance(getattr(module, "__file__", None), str)
    }
    by_distribution: dict[str, set[str]] = {}
    for top_level in sorted(top_level_modules):
        for candidate in package_map.get(top_level, ()):
            normalised = _normalise_distribution_name(candidate)
            by_distribution.setdefault(normalised, set()).add(top_level)
    for required in _CRITICAL_DISTRIBUTIONS:
        by_distribution.setdefault(required, set())

    rows: list[dict[str, object]] = []
    for normalised, modules in sorted(by_distribution.items()):
        try:
            distribution = importlib.metadata.distribution(normalised)
        except importlib.metadata.PackageNotFoundError as exc:
            raise MultiPackWorkerError(
                "critical_runtime_distribution_missing"
                if normalised in _CRITICAL_DISTRIBUTIONS
                else "loaded_runtime_distribution_missing"
            ) from exc
        declared_name = distribution.metadata.get("Name")
        if not isinstance(declared_name, str) or not declared_name:
            raise MultiPackWorkerError(
                "runtime_distribution_name_invalid"
            )
        actual_name = _normalise_distribution_name(declared_name)
        if actual_name != normalised:
            raise MultiPackWorkerError(
                "runtime_distribution_name_ambiguous"
            )
        module_origins: list[Path] = []
        for module_name in sorted(modules):
            module = sys.modules.get(module_name)
            origin = getattr(module, "__file__", None)
            if not isinstance(origin, str) or not origin:
                raise MultiPackWorkerError(
                    "loaded_runtime_module_origin_unavailable"
                )
            module_origins.append(Path(origin))
        rows.append(
            {
                "closure_sha256": worker._distribution_closure_sha256(
                    declared_name,
                    required_module_origins=tuple(module_origins),
                ),
                "critical": normalised in _CRITICAL_DISTRIBUTIONS,
                "distribution": normalised,
                "loaded_top_level_modules": sorted(modules),
                "version": distribution.version,
            }
        )
    observed = {row["distribution"] for row in rows}
    if not _CRITICAL_DISTRIBUTIONS.issubset(observed):
        raise MultiPackWorkerError(
            "critical_runtime_distribution_missing"
        )
    return rows


def _execution_closure_payload(
    value: contract.ExecutionClosure,
) -> dict[str, str]:
    if type(value) is not contract.ExecutionClosure:
        raise MultiPackWorkerError("execution_closure_invalid")
    return {
        "model_asset_manifest_sha256": (
            value.model_asset_manifest_sha256
        ),
        "model_runtime_closure_sha256": (
            value.model_runtime_closure_sha256
        ),
        "parser_closure_sha256": value.parser_closure_sha256,
        "prompt_sha256": value.prompt_sha256,
        "target_double_run_receipt_sha256": (
            value.target_double_run_receipt_sha256
        ),
    }


def _logical_gpu_binding(
    runtime: worker.LocalQwenRuntime,
) -> dict[str, object]:
    runtime._validate_formal_binding()
    torch = runtime._torch
    if torch.cuda.device_count() != 1:
        raise MultiPackWorkerError(
            "logical_gpu_visibility_not_single"
        )
    properties = torch.cuda.get_device_properties(0)
    uuid_value = getattr(properties, "uuid", None)
    uuid = str(uuid_value) if uuid_value is not None else ""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    parameter_indices = sorted(
        {
            int(parameter.device.index)
            for parameter in runtime._model.parameters()
            if parameter.device.type == "cuda"
            and parameter.device.index is not None
        }
    )
    if (
        not isinstance(visible, str)
        or re.fullmatch(r"[0-9]{1,2}", visible) is None
        or not uuid
        or parameter_indices != [0]
    ):
        raise MultiPackWorkerError(
            "logical_gpu_binding_invalid"
        )
    return {
        "cuda_visible_devices": visible,
        "logical_compute_capability": [
            int(properties.major),
            int(properties.minor),
        ],
        "logical_device_count": 1,
        "logical_device_index": 0,
        "logical_device_name": str(properties.name),
        "logical_device_uuid": uuid,
        "model_parameter_logical_device_indices": parameter_indices,
    }


def _write_once(path: Path, raw: bytes, *, work_root: Path) -> str:
    _require_beneath(
        path, work_root=work_root, final_must_exist=False
    )
    try:
        contract._write_bytes_once(path, raw)  # noqa: SLF001
    except contract.NarrativeExtractorRuntimeError as exc:
        raise MultiPackWorkerError(
            "multi_pack_receipt_write_failed"
        ) from exc
    return hashlib.sha256(raw).hexdigest()


def run_formal_multi_pack(
    *,
    input_manifest_path: Path,
    model_root: Path,
    model_manifest_path: Path,
    safe_receipt_path: Path,
) -> Mapping[str, object]:
    """Execute the only formal multi-pack path; no dependency injection."""

    (
        work_root,
        batches,
        manifest_file_sha256,
        lineage,
    ) = _decode_manifest(input_manifest_path)
    _require_beneath(
        safe_receipt_path,
        work_root=work_root,
        final_must_exist=False,
    )
    manifest = worker.load_model_asset_manifest(
        manifest_path=model_manifest_path, model_root=model_root
    )
    runtime = worker.LocalQwenRuntime(
        model_root=model_root, manifest=manifest
    )
    batch_receipts: list[dict[str, object]] = []
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
        started_ns = time.monotonic_ns()
        results = worker.process_trusted_pack(
            pack, runtime=runtime
        )
        elapsed_ns = time.monotonic_ns() - started_ns
        contract.write_private_output_once(
            output_path,
            pack=pack,
            execution_closure=runtime.execution_closure,
            results=results,
        )
        output_read = contract.secure_read_file(
            output_path, maximum=contract.MAXIMUM_OUTPUT_BYTES
        )
        decoded_output = contract.decode_private_output(
            output_read.raw, expected_pack=pack
        )
        if decoded_output["execution_closure"] != (
            _execution_closure_payload(runtime.execution_closure)
        ):
            raise MultiPackWorkerError(
                "multi_pack_execution_closure_changed"
            )
        batch_receipts.append(
            {
                "batch_id": pack.batch_id,
                "generation_invalid_count": sum(
                    result["generation_valid"] is False
                    for result in decoded_output["results"]
                ),
                "generation_valid_count": sum(
                    result["generation_valid"] is True
                    for result in decoded_output["results"]
                ),
                "generation_elapsed_ns": elapsed_ns,
                "valid_completion_token_count_maximum": max(
                    (
                        result["completion_token_count"]
                        for result in decoded_output["results"]
                        if result["generation_valid"] is True
                    ),
                    default=0,
                ),
                "valid_completion_token_count_sum": sum(
                    result["completion_token_count"]
                    for result in decoded_output["results"]
                    if result["generation_valid"] is True
                ),
                "input_file_sha256": pack.input_file_sha256,
                "input_pack_commitment": pack.input_pack_commitment,
                "output_file_sha256": output_read.sha256,
                "sequence": pack.sequence,
                "story_count": len(pack.requests),
            }
        )

    runtime._validate_formal_binding()
    runtime_receipt = dict(runtime.runtime_receipt)
    double_run_receipt = dict(runtime.target_double_run_receipt)
    runtime_receipt_sha256 = _semantic_sha256(runtime_receipt)
    double_run_receipt_sha256 = _semantic_sha256(
        double_run_receipt
    )
    execution_closure = _execution_closure_payload(
        runtime.execution_closure
    )
    logical_gpu_binding = _logical_gpu_binding(runtime)
    if (
        execution_closure["target_double_run_receipt_sha256"]
        != double_run_receipt_sha256
        or double_run_receipt.get("runtime_receipt_sha256")
        != runtime_receipt_sha256
    ):
        raise MultiPackWorkerError(
            "multi_pack_runtime_preimage_mismatch"
        )
    distribution_rows = _loaded_distribution_rows()
    body: dict[str, object] = {
        "batch_count": len(batch_receipts),
        "batches": batch_receipts,
        "claim_scope": contract.CLAIM_SCOPE,
        "execution_closure": execution_closure,
        "input_manifest_file_sha256": manifest_file_sha256,
        "lineage": lineage,
        "logical_gpu_binding": logical_gpu_binding,
        "loaded_distribution_closure_sha256": _semantic_sha256(
            distribution_rows
        ),
        "loaded_distributions": distribution_rows,
        "model_asset_manifest_file_sha256": (
            manifest.manifest_file_sha256
        ),
        "runtime_receipt": runtime_receipt,
        "runtime_receipt_sha256": runtime_receipt_sha256,
        "schema": SAFE_RECEIPT_SCHEMA,
        "single_model_load_count": 1,
        "source_content_supplied": lineage == "formal_measurement",
        "target_double_run_receipt": double_run_receipt,
        "target_double_run_receipt_sha256": (
            double_run_receipt_sha256
        ),
        "worker_version": VERSION,
    }
    receipt = {**body, "self_sha256": _semantic_sha256(body)}
    _write_once(
        safe_receipt_path,
        contract.canonical_json_bytes(receipt),
        work_root=work_root,
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--safe-receipt", required=True, type=Path)
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
                "loaded_distribution_closure_sha256": receipt[
                    "loaded_distribution_closure_sha256"
                ],
                "runtime_receipt_sha256": receipt[
                    "runtime_receipt_sha256"
                ],
                "status": "completed",
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
        MultiPackWorkerError,
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
    "run_formal_multi_pack",
]
