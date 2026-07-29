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
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from reconstruction_v2.assumption_agent.benchmarks import (
    hipporag_zero_weight_totality_qualification_v1 as qualification,
)
from reconstruction_v2.replication_runtime.hipporag_zero_weight_totality_v1 import (
    backport,
)

if __package__:
    from .landlock_runtime import landlock_abi_version
else:
    from landlock_runtime import landlock_abi_version


SCHEMA = "hipporag_zero_weight_totality_repair_qualification_v1"
CANARY_SCHEMA = "hipporag_zero_weight_totality_repair_canary_v1"
LANDLOCK_EXEC_SCHEMA = "hipporag_zero_weight_totality_landlock_exec_v1"
SYNTHETIC_MODULE = (
    "replication_runtime.hipporag_zero_weight_totality_v1.synthetic_worker"
)
CACHED_MODULE = (
    "replication_runtime.hipporag_zero_weight_totality_v1.cached_worker"
)
SYSTEM_READ_PATHS = tuple(
    Path(row)
    for row in (
        "/usr",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/etc",
        "/proc",
        "/sys",
    )
)
DEVICE_PATHS = (Path("/dev/null"), Path("/dev/urandom"))


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


def _prepare_short_model_aliases(
    *, llm_model: Path, embedding_model: Path
) -> tuple[Path, Path]:
    """Recreate the exact short model labels encoded in the frozen indexes."""

    alias_root = Path("/tmp/models")
    llm_alias = alias_root / "llm"
    embedding_alias = alias_root / "embed"
    if not alias_root.exists() and not alias_root.is_symlink():
        alias_root.mkdir(mode=0o700)
        llm_alias.symlink_to(llm_model, target_is_directory=True)
        embedding_alias.symlink_to(embedding_model, target_is_directory=True)
    else:
        metadata = alias_root.lstat()
        entries = {path.name for path in alias_root.iterdir()}
        if (
            alias_root.is_symlink()
            or not alias_root.is_dir()
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o777 != 0o700
            or entries != {"embed", "llm"}
            or not llm_alias.is_symlink()
            or not embedding_alias.is_symlink()
        ):
            raise HippoRAGTotalityRepairQualificationError(
                "shared short model alias root drifted"
            )
    if (
        llm_alias.resolve(strict=True) != llm_model.resolve(strict=True)
        or embedding_alias.resolve(strict=True)
        != embedding_model.resolve(strict=True)
    ):
        raise HippoRAGTotalityRepairQualificationError(
            "private short model alias binding failed"
        )
    return llm_alias, embedding_alias


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
    source_root.mkdir(mode=0o700, parents=True)
    input_copy = source_root / "HippoRAG.qualified_nonfinite.py"
    patched_path = source_root / "HippoRAG.py"
    patch_path = source_root / "HippoRAG.zero_weight_totality.patch"
    qualification._write_exclusive(input_copy, raw)
    qualification._write_exclusive(patched_path, patched)
    qualification._write_exclusive(patch_path, patch)
    package_source = input_source.parent
    if any(path.is_symlink() for path in package_source.rglob("*")):
        raise HippoRAGTotalityRepairQualificationError(
            "external HippoRAG package contains a symlink"
        )
    import_root = root / "patched_import"
    package_target = import_root / "hipporag"
    shutil.copytree(
        package_source,
        package_target,
        ignore=shutil.ignore_patterns("HippoRAG.py", "__pycache__", "*.pyc"),
    )
    import_source = package_target / "HippoRAG.py"
    qualification._write_exclusive(import_source, patched)
    import_rows = [
        {
            "relative_path": path.relative_to(import_root).as_posix(),
            "sha256": qualification.file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(import_root.rglob("*"))
        if path.is_file()
    ]
    if (
        not import_rows
        or qualification.file_sha256(import_source)
        != backport.PATCHED_SOURCE_SHA256
    ):
        raise HippoRAGTotalityRepairQualificationError(
            "patched import package materialization failed"
        )
    return {
        "input_source_sha256": observed,
        "patched_import_file_count": len(import_rows),
        "patched_import_root": import_root,
        "patched_import_set_sha256": qualification.stable_hash(import_rows),
        "patched_path": patched_path,
        "patched_source_sha256": qualification.file_sha256(patched_path),
        "unified_patch_sha256": qualification.file_sha256(patch_path),
    }


def _worker_environment(
    *,
    writable_root: Path,
    patched_import_root: Path,
    runtime_root: Path,
) -> dict[str, str]:
    python_path = os.pathsep.join(
        str(path)
        for path in (
            patched_import_root,
            runtime_root,
            runtime_root / "reconstruction_v2",
            runtime_root / "p16_site",
        )
    )
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(writable_root / "hf"),
        "HOME": str(writable_root / "home"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MKL_NUM_THREADS": "1",
        "MPLCONFIGDIR": str(writable_root / "tmp" / "mpl"),
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "1",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": python_path,
        "SENTENCE_TRANSFORMERS_HOME": str(writable_root / "hf"),
        "TMPDIR": str(writable_root / "tmp"),
        "TOKENIZERS_PARALLELISM": "false",
        "TORCH_HOME": str(writable_root / "hf"),
        "TRANSFORMERS_OFFLINE": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "XDG_CACHE_HOME": str(writable_root / "hf"),
    }


def _landlock_read_paths(
    *, runtime_root: Path, patched_import_root: Path
) -> tuple[Path, ...]:
    return tuple(
        dict.fromkeys(
            path
            for path in (
                *SYSTEM_READ_PATHS,
                runtime_root,
                patched_import_root,
            )
            if path.exists()
        )
    )


def _run_landlocked(
    *,
    arguments: Sequence[str],
    writable_root: Path,
    patched_import_root: Path,
    runtime_root: Path,
    runtime_python: Path,
    timeout: int,
) -> subprocess.CompletedProcess[bytes]:
    environment = _worker_environment(
        writable_root=writable_root,
        patched_import_root=patched_import_root,
        runtime_root=runtime_root,
    )
    spec = {
        "argv": [str(runtime_python), "-B", *arguments],
        "cwd": str(writable_root),
        "device_paths": [str(path) for path in DEVICE_PATHS],
        "environment": environment,
        "read_paths": [
            str(path)
            for path in _landlock_read_paths(
                runtime_root=runtime_root,
                patched_import_root=patched_import_root,
            )
        ],
        "schema": LANDLOCK_EXEC_SCHEMA,
        "write_paths": [str(writable_root)],
    }
    spec_path = writable_root / "landlock_exec.json"
    qualification._write_json(spec_path, spec)
    launcher = Path(__file__).with_name("landlock_exec.py").resolve(strict=True)
    command = [
        str(runtime_python),
        "-B",
        str(launcher),
        "--spec",
        str(spec_path),
    ]
    return subprocess.run(
        command,
        cwd=writable_root,
        env=environment,
        stdin=subprocess.DEVNULL,
        check=False,
        capture_output=True,
        timeout=timeout,
    )


def _run_synthetic_landlock(
    *,
    root: Path,
    patched_import_root: Path,
    runtime_root: Path,
    runtime_python: Path,
) -> dict[str, Any]:
    synthetic_root = root / "synthetic"
    _prepare_writable_root(synthetic_root)
    output = synthetic_root / "output.json"
    completed = _run_landlocked(
        arguments=("-m", SYNTHETIC_MODULE, "--output", str(output)),
        writable_root=synthetic_root,
        patched_import_root=patched_import_root,
        runtime_root=runtime_root,
        runtime_python=runtime_python,
        timeout=300,
    )
    qualification._write_exclusive(
        synthetic_root / "stdout.log", completed.stdout
    )
    qualification._write_exclusive(
        synthetic_root / "stderr.log", completed.stderr
    )
    if completed.returncode != 0:
        raise HippoRAGTotalityRepairQualificationError(
            "Landlock synthetic worker failed: "
            + hashlib.sha256(completed.stderr).hexdigest()
        )
    value = qualification._read_json(output, "synthetic totality output")
    if (
        value.get("schema")
        != "hipporag_zero_weight_totality_synthetic_fixture_v1"
        or value.get("source_sha256") != backport.PATCHED_SOURCE_SHA256
        or value.get("allowed_linking_key_count") != 2
        or value.get("allowed_nonzero_weight_count") != 1
        or value.get("allowed_values_unchanged") is not True
        or value.get("rejected_cases")
        != ["nonfinite", "unselected_nonzero"]
    ):
        raise HippoRAGTotalityRepairQualificationError(
            "Landlock synthetic totality result drifted"
        )
    return {
        "landlock_abi": landlock_abi_version(),
        "output_file_sha256": qualification.file_sha256(output),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        **value,
    }


def _run_cached_item_landlock(
    *,
    fixture_base: Path,
    root: Path,
    patched_import_root: Path,
    runtime_root: Path,
    llm_model: Path,
    embedding_model: Path,
    runtime_python: Path,
    source: qualification.FixtureSource,
    ordinal: int,
    counter: qualification._ConcurrencyCounter,
) -> dict[str, Any]:
    original = fixture_base / source.root_relative / f"item_{ordinal:03d}"
    item = root / "cached" / source.key / f"item_{ordinal:03d}"
    _prepare_writable_root(item)
    shutil.copy2(original / "input.json", item / "input.json")
    shutil.copytree(original / "index", item / "index")
    output = item / "output.json"
    counter.enter()
    try:
        completed = _run_landlocked(
            arguments=(
                "-m",
                CACHED_MODULE,
                "--input",
                str(item / "input.json"),
                "--output",
                str(output),
                "--index-root",
                str(item / "index"),
                "--llm-model",
                str(llm_model),
                "--embedding-model",
                str(embedding_model),
            ),
            writable_root=item,
            patched_import_root=patched_import_root,
            runtime_root=runtime_root,
            runtime_python=runtime_python,
            timeout=900,
        )
    finally:
        counter.leave()
    qualification._write_exclusive(item / "stdout.log", completed.stdout)
    qualification._write_exclusive(item / "stderr.log", completed.stderr)
    if completed.returncode != 0:
        raise HippoRAGTotalityRepairQualificationError(
            f"Landlock cached item {source.key}/{ordinal} failed: "
            + hashlib.sha256(completed.stderr).hexdigest()
        )
    value = qualification._parse_worker_output(output)
    was_failure = ordinal in source.failure_ordinals
    if was_failure:
        if value["graph_node_count"] <= 0 or value["graph_edge_count"] <= 0:
            raise HippoRAGTotalityRepairQualificationError(
                "repaired failure fixture graph is empty"
            )
        disposition = "previous_failure_now_structurally_valid"
    else:
        if output.read_bytes() != (original / "output.json").read_bytes():
            raise HippoRAGTotalityRepairQualificationError(
                f"cached parity item {source.key}/{ordinal} output changed"
            )
        disposition = "previous_success_byte_identical"
    raw = output.read_bytes()
    return {
        "disposition": disposition,
        "graph_edge_count": value["graph_edge_count"],
        "graph_node_count": value["graph_node_count"],
        "ordinal": ordinal,
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "output_size_bytes": len(raw),
        "source": source.key,
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
    }


def _run_cached_set_landlock(
    *,
    fixture_base: Path,
    root: Path,
    patched_import_root: Path,
    runtime_root: Path,
    llm_model: Path,
    embedding_model: Path,
    runtime_python: Path,
    process_concurrency: int,
) -> tuple[list[dict[str, Any]], int]:
    (root / "cached").mkdir(mode=0o700)
    counter = qualification._ConcurrencyCounter()
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    work = [
        (source, ordinal)
        for source in qualification.FIXTURE_SOURCES
        for ordinal in range(source.item_count)
    ]
    with ThreadPoolExecutor(max_workers=process_concurrency) as executor:
        futures: dict[Future[dict[str, Any]], tuple[str, int]] = {
            executor.submit(
                _run_cached_item_landlock,
                fixture_base=fixture_base,
                root=root,
                patched_import_root=patched_import_root,
                runtime_root=runtime_root,
                llm_model=llm_model,
                embedding_model=embedding_model,
                runtime_python=runtime_python,
                source=source,
                ordinal=ordinal,
                counter=counter,
            ): (source.key, ordinal)
            for source, ordinal in work
        }
        for future in as_completed(futures):
            key = futures[future]
            completed[key] = future.result()
    expected_keys = {(source.key, ordinal) for source, ordinal in work}
    if (
        counter.current != 0
        or not 0 < counter.peak <= process_concurrency
        or set(completed) != expected_keys
    ):
        raise HippoRAGTotalityRepairQualificationError(
            "Landlock cached-set completion or concurrency drifted"
        )
    rows = [
        completed[(source.key, ordinal)]
        for source in qualification.FIXTURE_SOURCES
        for ordinal in range(source.item_count)
    ]
    return rows, counter.peak


def _bind_external_runtime(
    *,
    runtime_root: Path,
    llm_model: Path,
    embedding_model: Path,
    runtime_python: Path,
) -> Path:
    source_repo = (
        runtime_root
        / "reconstruction_v2/reference/"
        "self_evo_continual_20260707/repos/HippoRAG"
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


def _public_source_binding(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in source.items()
        if key not in {"patched_import_root", "patched_path"}
    }


def _validate_fresh_paths(scratch_root: Path, output: Path) -> None:
    if (
        scratch_root.exists()
        or scratch_root.is_symlink()
        or output.exists()
        or output.is_symlink()
    ):
        raise HippoRAGTotalityRepairQualificationError(
            "repair qualification output has already been consumed"
        )


def run_canary(
    *,
    fixture_base: Path,
    scratch_root: Path,
    output: Path,
    runtime_root: Path,
    llm_model: Path,
    embedding_model: Path,
    runtime_python: Path,
) -> Mapping[str, Any]:
    fixture_base = fixture_base.resolve(strict=True)
    runtime_root = runtime_root.resolve(strict=True)
    llm_model = llm_model.resolve(strict=True)
    embedding_model = embedding_model.resolve(strict=True)
    runtime_python = runtime_python.absolute()
    _validate_fresh_paths(scratch_root, output)
    source_path = _bind_external_runtime(
        runtime_root=runtime_root,
        llm_model=llm_model,
        embedding_model=embedding_model,
        runtime_python=runtime_python,
    )
    artifact_binding = qualification.verify_frozen_artifact_sets(fixture_base)
    llm_alias, embedding_alias = _prepare_short_model_aliases(
        llm_model=llm_model, embedding_model=embedding_model
    )
    scratch_root.mkdir(mode=0o700)
    source = _materialize_source(source_path, scratch_root)
    synthetic = _run_synthetic_landlock(
        root=scratch_root,
        patched_import_root=source["patched_import_root"],
        runtime_root=runtime_root,
        runtime_python=runtime_python,
    )
    success_source = qualification.FIXTURE_SOURCES[0]
    failure_source = next(
        source_row
        for source_row in qualification.FIXTURE_SOURCES
        if source_row.failure_ordinals
    )
    counter = qualification._ConcurrencyCounter()
    rows = [
        _run_cached_item_landlock(
            fixture_base=fixture_base,
            root=scratch_root,
            patched_import_root=source["patched_import_root"],
            runtime_root=runtime_root,
            llm_model=llm_alias,
            embedding_model=embedding_alias,
            runtime_python=runtime_python,
            source=success_source,
            ordinal=0,
            counter=counter,
        ),
        _run_cached_item_landlock(
            fixture_base=fixture_base,
            root=scratch_root,
            patched_import_root=source["patched_import_root"],
            runtime_root=runtime_root,
            llm_model=llm_alias,
            embedding_model=embedding_alias,
            runtime_python=runtime_python,
            source=failure_source,
            ordinal=min(failure_source.failure_ordinals),
            counter=counter,
        ),
    ]
    dispositions = sorted(row["disposition"] for row in rows)
    if dispositions != [
        "previous_failure_now_structurally_valid",
        "previous_success_byte_identical",
    ] or counter.current != 0 or counter.peak != 1:
        raise HippoRAGTotalityRepairQualificationError(
            "Landlock canary disposition drifted"
        )
    result = qualification.self_hashed(
        {
            "artifact_binding": artifact_binding,
            "cached_retrieval_canary": {
                "item_count": len(rows),
                "previous_failure_structurally_valid_count": 1,
                "previous_success_byte_identical_count": 1,
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
            "recorded_date": "2026-07-29",
            "schema": CANARY_SCHEMA,
            "source_hardening": _public_source_binding(source),
            "status": "passed_landlock_source_and_cached_worker_canary",
            "synthetic": synthetic,
        }
    )
    qualification._write_json(output, result)
    return result


def _verify_canary(
    path: Path, artifact_binding: Mapping[str, Any]
) -> Mapping[str, Any]:
    resolved = path.resolve(strict=True)
    value = qualification._read_json(resolved, "repair canary")
    qualification.verify_self_hash(value)
    cached = value.get("cached_retrieval_canary")
    source = value.get("source_hardening")
    if (
        value.get("schema") != CANARY_SCHEMA
        or value.get("status")
        != "passed_landlock_source_and_cached_worker_canary"
        or value.get("artifact_binding") != artifact_binding
        or not isinstance(cached, Mapping)
        or cached.get("item_count") != 2
        or cached.get("previous_failure_structurally_valid_count") != 1
        or cached.get("previous_success_byte_identical_count") != 1
        or not isinstance(source, Mapping)
        or source.get("patched_source_sha256")
        != backport.PATCHED_SOURCE_SHA256
    ):
        raise HippoRAGTotalityRepairQualificationError(
            "Landlock canary binding drifted"
        )
    return value


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
    canary_result: Path,
) -> Mapping[str, Any]:
    fixture_base = fixture_base.resolve(strict=True)
    runtime_root = runtime_root.resolve(strict=True)
    llm_model = llm_model.resolve(strict=True)
    embedding_model = embedding_model.resolve(strict=True)
    runtime_python = runtime_python.absolute()
    _validate_fresh_paths(scratch_root, output)
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
    canary = _verify_canary(canary_result, artifact_binding)
    llm_alias, embedding_alias = _prepare_short_model_aliases(
        llm_model=llm_model, embedding_model=embedding_model
    )
    scratch_root.mkdir(mode=0o700)
    source = _materialize_source(source_path, scratch_root)
    synthetic = _run_synthetic_landlock(
        root=scratch_root,
        patched_import_root=source["patched_import_root"],
        runtime_root=runtime_root,
        runtime_python=runtime_python,
    )
    rows, peak = _run_cached_set_landlock(
        fixture_base=fixture_base,
        root=scratch_root,
        patched_import_root=source["patched_import_root"],
        runtime_root=runtime_root,
        llm_model=llm_alias,
        embedding_model=embedding_alias,
        runtime_python=runtime_python,
        process_concurrency=process_concurrency,
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
    result = qualification.self_hashed(
        {
            "artifact_binding": artifact_binding,
            "canary_binding": {
                "file_sha256": qualification.file_sha256(
                    canary_result.resolve(strict=True)
                ),
                "self_sha256": canary["self_sha256"],
            },
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
            "source_hardening": _public_source_binding(source),
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
    parser.add_argument("--mode", choices=("canary", "full"), required=True)
    parser.add_argument("--fixture-base", required=True, type=Path)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--llm-model", required=True, type=Path)
    parser.add_argument("--embedding-model", required=True, type=Path)
    parser.add_argument("--runtime-python", required=True, type=Path)
    parser.add_argument("--canary-result", type=Path)
    parser.add_argument(
        "--process-concurrency",
        required=True,
        type=int,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    common = {
        "fixture_base": arguments.fixture_base,
        "scratch_root": arguments.scratch_root,
        "output": arguments.output,
        "runtime_root": arguments.runtime_root,
        "llm_model": arguments.llm_model,
        "embedding_model": arguments.embedding_model,
        "runtime_python": arguments.runtime_python,
    }
    if arguments.mode == "canary":
        if arguments.canary_result is not None:
            raise HippoRAGTotalityRepairQualificationError(
                "canary mode cannot consume a canary result"
            )
        value = run_canary(**common)
    else:
        if arguments.canary_result is None:
            raise HippoRAGTotalityRepairQualificationError(
                "full mode requires the passed canary result"
            )
        value = run(
            **common,
            process_concurrency=arguments.process_concurrency,
            canary_result=arguments.canary_result,
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
