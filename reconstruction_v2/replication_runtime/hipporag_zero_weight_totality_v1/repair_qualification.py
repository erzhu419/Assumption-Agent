"""Offline qualification continuation for the zero-weight totality repair.

The original qualification stopped before launching a cached worker because
its per-source parent directory was never created.  This runner keeps that
terminal attempt immutable, uses the same 60 frozen label-free fixtures, and
changes only writable-root materialization.  It accepts explicit external
runtime assets so the heavy cached retrieval can run on a shared remote node
without copying or modifying its frozen HippoRAG installation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from reconstruction_v2.assumption_agent.benchmarks import (
    hipporag_zero_weight_totality_qualification_v1 as qualification,
)
from reconstruction_v2.replication_runtime.hipporag_zero_weight_totality_v1 import (
    backport,
)


SCHEMA = "hipporag_zero_weight_totality_repair_qualification_v1"


class HippoRAGTotalityRepairQualificationError(RuntimeError):
    """The non-scoring repair qualification failed closed."""


def _direct_file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise HippoRAGTotalityRepairQualificationError(
            "required direct file is unavailable"
        )
    return qualification.file_sha256(path)


def _prepare_writable_root(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True)
    for name in ("home", "hf", "tmp"):
        (path / name).mkdir(mode=0o700)


def _materialize_source(input_source: Path, root: Path) -> Mapping[str, Any]:
    raw = input_source.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != backport.INPUT_SOURCE_SHA256:
        raise HippoRAGTotalityRepairQualificationError(
            "qualified input source drifted"
        )
    patched = backport.apply_totality_hardening(raw)
    patch = backport.unified_patch_bytes(raw, patched)
    source_root = root / "patched_source"
    source_root.mkdir(mode=0o700)
    input_copy = source_root / "HippoRAG.qualified_nonfinite.py"
    patched_path = source_root / "HippoRAG.py"
    patch_path = source_root / "HippoRAG.zero_weight_totality.patch"
    qualification._write_exclusive(input_copy, raw)
    qualification._write_exclusive(patched_path, patched)
    qualification._write_exclusive(patch_path, patch)
    return {
        "input_source_sha256": observed,
        "patched_path": patched_path,
        "patched_source_sha256": qualification.file_sha256(patched_path),
        "unified_patch_sha256": qualification.file_sha256(patch_path),
    }


def _bind_external_runtime(
    *,
    runtime_root: Path,
    llm_model: Path,
    embedding_model: Path,
    runtime_python: Path,
) -> Path:
    source_repo = (
        runtime_root
        / "reference/self_evo_continual_20260707/repos/HippoRAG"
    ).resolve(strict=True)
    source = (
        source_repo / qualification.BASELINE_SOURCE_WITHIN_REPO
    ).resolve(strict=True)
    if _direct_file_sha256(source) != backport.INPUT_SOURCE_SHA256:
        raise HippoRAGTotalityRepairQualificationError(
            "external HippoRAG source drifted"
        )
    for model in (llm_model, embedding_model):
        if model.is_symlink() or not model.is_dir():
            raise HippoRAGTotalityRepairQualificationError(
                "external offline model is unavailable"
            )
    if not runtime_python.resolve(strict=True).is_file():
        raise HippoRAGTotalityRepairQualificationError(
            "external runtime Python is unavailable"
        )
    qualification.BASELINE_REPO_RELATIVE = source_repo
    qualification.LLM_MODEL_RELATIVE = llm_model.resolve(strict=True)
    qualification.EMBEDDING_MODEL_RELATIVE = embedding_model.resolve(
        strict=True
    )
    qualification.RUNTIME_PYTHON_RELATIVE = runtime_python.resolve(
        strict=True
    )
    qualification._prepare_writable_root = _prepare_writable_root
    return source


def run(
    *,
    fixture_base: Path,
    scratch_root: Path,
    output: Path,
    runtime_root: Path,
    llm_model: Path,
    embedding_model: Path,
    runtime_python: Path,
    process_concurrency: int,
) -> Mapping[str, Any]:
    fixture_base = fixture_base.resolve(strict=True)
    runtime_root = runtime_root.resolve(strict=True)
    if (
        scratch_root.exists()
        or scratch_root.is_symlink()
        or output.exists()
        or output.is_symlink()
    ):
        raise HippoRAGTotalityRepairQualificationError(
            "repair qualification output has already been consumed"
        )
    if not 1 <= process_concurrency <= qualification.PROCESS_CONCURRENCY:
        raise HippoRAGTotalityRepairQualificationError(
            "process concurrency is outside the frozen maximum"
        )
    source_path = _bind_external_runtime(
        runtime_root=runtime_root,
        llm_model=llm_model,
        embedding_model=embedding_model,
        runtime_python=runtime_python,
    )
    artifact_binding = qualification.verify_frozen_artifact_sets(fixture_base)
    scratch_root.mkdir(mode=0o700)
    source = _materialize_source(source_path, scratch_root)
    synthetic = qualification._run_synthetic(
        fixture_base, scratch_root, source["patched_path"]
    )
    qualification.PROCESS_CONCURRENCY = process_concurrency
    rows, peak = qualification._run_cached_set(
        fixture_base, scratch_root, source["patched_path"]
    )
    parity_count = sum(
        row["disposition"] == "previous_success_byte_identical"
        for row in rows
    )
    repair_count = sum(
        row["disposition"]
        == "previous_failure_now_structurally_valid"
        for row in rows
    )
    if len(rows) != 60 or parity_count != 58 or repair_count != 2:
        raise HippoRAGTotalityRepairQualificationError(
            "cached qualification disposition count drifted"
        )
    public_source = {
        key: value for key, value in source.items() if key != "patched_path"
    }
    result = qualification.self_hashed(
        {
            "artifact_binding": artifact_binding,
            "cached_retrieval": {
                "item_count": len(rows),
                "peak_process_concurrency": peak,
                "previous_failure_structurally_valid_count": repair_count,
                "previous_success_byte_identical_count": parity_count,
                "row_set_sha256": qualification.stable_hash(rows),
            },
            "claim_boundary": {
                "current_failed_effect_cohort_replayed_or_scored": False,
                "external_network_call_count": 0,
                "label_or_qrel_open_count": 0,
                "online_evaluator_call_count": 0,
                "performance_score_count": 0,
                "qualification_is_comparator_performance_evidence": False,
            },
            "original_qualification_disposition": {
                "cached_worker_launch_count": 0,
                "failure_class": (
                    "missing_per_source_parent_before_cached_worker_launch"
                ),
                "original_root_reused_or_modified": False,
            },
            "recorded_date": "2026-07-29",
            "runtime_binding": {
                "embedding_model_path": str(embedding_model),
                "input_source_path": str(source_path),
                "llm_model_path": str(llm_model),
                "runtime_python_path": str(runtime_python),
            },
            "schema": SCHEMA,
            "source_hardening": public_source,
            "status": (
                "passed_zero_weight_totality_repair_qualified_for_"
                "future_frozen_comparator_use"
            ),
            "synthetic": synthetic,
        }
    )
    qualification._write_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-base", required=True, type=Path)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--llm-model", required=True, type=Path)
    parser.add_argument("--embedding-model", required=True, type=Path)
    parser.add_argument("--runtime-python", required=True, type=Path)
    parser.add_argument(
        "--process-concurrency",
        required=True,
        type=int,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    value = run(
        fixture_base=arguments.fixture_base,
        scratch_root=arguments.scratch_root,
        output=arguments.output,
        runtime_root=arguments.runtime_root,
        llm_model=arguments.llm_model,
        embedding_model=arguments.embedding_model,
        runtime_python=arguments.runtime_python,
        process_concurrency=arguments.process_concurrency,
    )
    print(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
