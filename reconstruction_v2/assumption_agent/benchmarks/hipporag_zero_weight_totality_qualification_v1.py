"""One-shot offline qualification of the HippoRAG zero-weight totality edit.

The qualification reuses only label-free cached comparator inputs and indexes.
Fifty-eight previously successful outputs must remain byte-identical, while the
two previously asserting indexes must terminate with structurally valid output.
No qrel, label, performance score, network call, or failed-cohort continuation
is part of this program.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any, Mapping, Sequence

from reconstruction_v2.replication_runtime.hipporag_upstream_hardening_v1 import (
    backport as upstream_hardening,
)
from reconstruction_v2.replication_runtime.hipporag_zero_weight_totality_v1 import (
    backport,
)


SCHEMA = "hipporag_zero_weight_totality_qualification_result_v1"
FREEZE_SCHEMA = "hipporag_zero_weight_totality_qualification_freeze_v1"
DESIGN_SCHEMA = "hipporag_zero_weight_totality_qualification_design_v1"
DESIGN_SELF_SHA256 = (
    "c1440565392b93fa85812539c63b8ca5df5f3fecdb81c0fc891aa6f93fa63f6e"
)
PROCESS_CONCURRENCY = 12

DESIGN_RELATIVE = Path(
    "manifests/hipporag_zero_weight_totality_qualification_design_v1.json"
)
FREEZE_RELATIVE = Path(
    "manifests/hipporag_zero_weight_totality_qualification_freeze_v1.json"
)
RESULT_RELATIVE = Path(
    "manifests/hipporag_zero_weight_totality_qualification_result_v1.json"
)
RUN_ROOT_RELATIVE = Path(
    "artifacts/hipporag_zero_weight_totality_qualification_v1"
)
BASELINE_REPO_RELATIVE = Path(
    "reference/self_evo_continual_20260707/repos/HippoRAG"
)
BASELINE_SOURCE_WITHIN_REPO = Path("src/hipporag/HippoRAG.py")
LLM_MODEL_RELATIVE = Path(
    "artifacts/bright_reasoning_retrieval_runtime_v1/smollm2_135m_instruct_exact"
)
EMBEDDING_MODEL_RELATIVE = Path("artifacts/qasper_minilm_runtime_v1/model")
RUNTIME_PYTHON_RELATIVE = Path(
    "artifacts/bright_reasoning_retrieval_runtime_v1/hipporag_venv/bin/python"
)
PRIOR_RESULT_RELATIVE = Path(
    "manifests/hipporag_upstream_hardening_qualification_result_v1.json"
)
FIQA_FAILURE_RELATIVE = Path(
    "manifests/fiqa_bridge_expansion_dev_runtime_failure_v1.json"
)
NANOBEIR_FAILURE_RELATIVE = Path(
    "manifests/nanobeir_p12_c_confirm_runtime_failure_v1.json"
)

IMPLEMENTATION_RELATIVES = (
    Path(
        "assumption_agent/benchmarks/"
        "hipporag_zero_weight_totality_qualification_v1.py"
    ),
    Path("replication_runtime/hipporag_zero_weight_totality_v1/__init__.py"),
    Path("replication_runtime/hipporag_zero_weight_totality_v1/backport.py"),
    Path("replication_runtime/hipporag_zero_weight_totality_v1/cached_worker.py"),
    Path("replication_runtime/hipporag_zero_weight_totality_v1/synthetic_worker.py"),
    Path("tests/test_hipporag_zero_weight_totality_qualification_v1.py"),
    Path("replication_runtime/hipporag_upstream_hardening_v1/backport.py"),
    Path("replication_runtime/hipporag_upstream_hardening_v1/cached_worker.py"),
    Path("replication_runtime/bright_official_hipporag_v1/contract.py"),
    Path("replication_runtime/bright_official_hipporag_v1/worker.py"),
)


@dataclass(frozen=True)
class FixtureSource:
    key: str
    public_key: str
    root_relative: Path
    item_count: int
    failure_ordinals: frozenset[int]


FIXTURE_SOURCES = (
    FixtureSource(
        key="fiqa_train",
        public_key="FiQA_TRAIN",
        root_relative=Path(
            "artifacts/fiqa_bridge_expansion_train_runtime_v2/hipporag"
        ),
        item_count=12,
        failure_ordinals=frozenset(),
    ),
    FixtureSource(
        key="fiqa_dev",
        public_key="FiQA_DEV",
        root_relative=Path(
            "artifacts/fiqa_bridge_expansion_dev_runtime_v1/hipporag"
        ),
        item_count=12,
        failure_ordinals=frozenset({2}),
    ),
    FixtureSource(
        key="nanobeir_p12",
        public_key="NanoBEIR_P12",
        root_relative=Path(
            "artifacts/nanobeir_p12_c_confirm_runtime_v1/hipporag"
        ),
        item_count=36,
        failure_ordinals=frozenset({15}),
    ),
)


class HippoRAGTotalityQualificationError(RuntimeError):
    """The frozen non-scoring qualification failed closed."""


class OneShotRefusal(HippoRAGTotalityQualificationError):
    """The formal qualification root or result has already been consumed."""


class _ConcurrencyCounter:
    def __init__(self) -> None:
        self.current = 0
        self.peak = 0
        self._lock = threading.Lock()

    def enter(self) -> None:
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def leave(self) -> None:
        with self._lock:
            self.current -= 1


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise HippoRAGTotalityQualificationError(
            "value is not canonical JSON"
        ) from exc


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_exclusive(path: Path, raw: bytes, mode: int = 0o600) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_exclusive(path, canonical_json_bytes(value))


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HippoRAGTotalityQualificationError(f"{name} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HippoRAGTotalityQualificationError(f"{name} is invalid") from exc
    if not isinstance(value, Mapping):
        raise HippoRAGTotalityQualificationError(f"{name} is not an object")
    return value


def verify_self_hash(value: Mapping[str, Any], field: str = "self_sha256") -> str:
    declared = value.get(field)
    if not isinstance(declared, str) or len(declared) != 64:
        raise HippoRAGTotalityQualificationError("self hash is absent")
    body = dict(value)
    body.pop(field, None)
    observed = stable_hash(body)
    if observed != declared:
        raise HippoRAGTotalityQualificationError("self hash drifted")
    return declared


def self_hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if "self_sha256" in result:
        raise HippoRAGTotalityQualificationError("self hash already exists")
    result["self_sha256"] = stable_hash(result)
    return result


def _git_output(arguments: Sequence[str], cwd: Path) -> bytes:
    completed = subprocess.run(
        ["git", *arguments], cwd=cwd, check=False, capture_output=True
    )
    if completed.returncode != 0:
        raise HippoRAGTotalityQualificationError(
            "git provenance command failed: "
            + hashlib.sha256(completed.stderr).hexdigest()
        )
    return completed.stdout


def _git_succeeds(arguments: Sequence[str], cwd: Path) -> bool:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def _verify_design_and_freeze(base: Path, project_root: Path) -> Mapping[str, Any]:
    design = _read_json(base / DESIGN_RELATIVE, "totality design")
    if (
        design.get("schema") != DESIGN_SCHEMA
        or verify_self_hash(design) != DESIGN_SELF_SHA256
    ):
        raise HippoRAGTotalityQualificationError("totality design identity drifted")
    freeze = _read_json(base / FREEZE_RELATIVE, "totality freeze")
    if freeze.get("schema") != FREEZE_SCHEMA:
        raise HippoRAGTotalityQualificationError("totality freeze schema drifted")
    verify_self_hash(freeze)
    if freeze.get("design_self_sha256") != DESIGN_SELF_SHA256:
        raise HippoRAGTotalityQualificationError("freeze design binding drifted")
    expected_commit = freeze.get("formal_implementation_commit")
    if (
        not isinstance(expected_commit, str)
        or not _git_succeeds(
            ["merge-base", "--is-ancestor", expected_commit, "HEAD"], project_root
        )
    ):
        raise HippoRAGTotalityQualificationError(
            "formal implementation commit drifted"
        )
    rows = freeze.get("implementation_bindings")
    if not isinstance(rows, list):
        raise HippoRAGTotalityQualificationError("freeze bindings are absent")
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in rows
        if isinstance(row, Mapping)
    }
    expected_paths = {path.as_posix() for path in IMPLEMENTATION_RELATIVES}
    if set(observed) != expected_paths:
        raise HippoRAGTotalityQualificationError("freeze file set drifted")
    for relative, expected in observed.items():
        path = base / str(relative)
        if not isinstance(expected, str) or file_sha256(path) != expected:
            raise HippoRAGTotalityQualificationError(
                "frozen implementation file drifted"
            )
    return freeze


def _verify_receipt(
    *, path: Path, expected_file: str, expected_self: str, name: str
) -> None:
    if file_sha256(path) != expected_file:
        raise HippoRAGTotalityQualificationError(f"{name} file drifted")
    value = _read_json(path, name)
    if verify_self_hash(value) != expected_self:
        raise HippoRAGTotalityQualificationError(f"{name} identity drifted")


def _artifact_rows(base: Path) -> tuple[list[dict[str, Any]], ...]:
    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    indexes: list[dict[str, Any]] = []
    for source in FIXTURE_SOURCES:
        root = base / source.root_relative
        for ordinal in range(source.item_count):
            item = root / f"item_{ordinal:03d}"
            input_path = item / "input.json"
            index = item / "index"
            output = item / "output.json"
            stderr = item / "stderr.log"
            if (
                input_path.is_symlink()
                or not input_path.is_file()
                or index.is_symlink()
                or not index.is_dir()
            ):
                raise HippoRAGTotalityQualificationError(
                    "cached fixture input or index is absent"
                )
            inputs.append(
                {
                    "source": source.key,
                    "ordinal": ordinal,
                    "sha256": file_sha256(input_path),
                    "size_bytes": input_path.stat().st_size,
                }
            )
            files = sorted(path for path in index.rglob("*") if path.is_file())
            if not files or any(path.is_symlink() for path in files):
                raise HippoRAGTotalityQualificationError(
                    "cached fixture index drifted"
                )
            indexes.extend(
                {
                    "source": source.key,
                    "ordinal": ordinal,
                    "relative_path": path.relative_to(index).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in files
            )
            expected_failure = ordinal in source.failure_ordinals
            if expected_failure:
                if output.exists() or output.is_symlink() or not stderr.is_file():
                    raise HippoRAGTotalityQualificationError(
                        "failure fixture terminal state drifted"
                    )
                failures.append(
                    {
                        "source": source.key,
                        "ordinal": ordinal,
                        "stderr_sha256": file_sha256(stderr),
                        "stderr_size_bytes": stderr.stat().st_size,
                    }
                )
            else:
                if output.is_symlink() or not output.is_file():
                    raise HippoRAGTotalityQualificationError(
                        "success fixture output is absent"
                    )
                outputs.append(
                    {
                        "source": source.key,
                        "ordinal": ordinal,
                        "sha256": file_sha256(output),
                        "size_bytes": output.stat().st_size,
                    }
                )
    return inputs, outputs, failures, indexes


def verify_frozen_artifact_sets(base: Path) -> dict[str, Any]:
    design = _read_json(base / DESIGN_RELATIVE, "totality design")
    if verify_self_hash(design) != DESIGN_SELF_SHA256:
        raise HippoRAGTotalityQualificationError("totality design drifted")
    binding = design.get("failure_evidence_binding")
    if not isinstance(binding, Mapping):
        raise HippoRAGTotalityQualificationError("failure binding is absent")
    _verify_receipt(
        path=base / FIQA_FAILURE_RELATIVE,
        expected_file=str(binding.get("FiQA_DEV_failure_file_sha256")),
        expected_self=str(binding.get("FiQA_DEV_failure_self_sha256")),
        name="FiQA DEV failure receipt",
    )
    _verify_receipt(
        path=base / NANOBEIR_FAILURE_RELATIVE,
        expected_file=str(binding.get("NanoBEIR_P12_failure_file_sha256")),
        expected_self=str(binding.get("NanoBEIR_P12_failure_self_sha256")),
        name="NanoBEIR P12 failure receipt",
    )
    prior = design.get("prior_hardening_binding")
    if not isinstance(prior, Mapping):
        raise HippoRAGTotalityQualificationError("prior hardening binding is absent")
    _verify_receipt(
        path=base / PRIOR_RESULT_RELATIVE,
        expected_file=str(prior.get("qualification_result_file_sha256")),
        expected_self=str(prior.get("qualification_result_self_sha256")),
        name="prior hardening qualification",
    )
    inputs, outputs, failures, indexes = _artifact_rows(base)
    source_rows: dict[str, dict[str, int]] = {}
    for source in FIXTURE_SOURCES:
        source_rows[source.public_key] = {
            "failure_count": sum(
                row["source"] == source.key for row in failures
            ),
            "item_count": sum(row["source"] == source.key for row in inputs),
            "success_count": sum(row["source"] == source.key for row in outputs),
        }
    observed = {
        "failure_fixture_count": len(failures),
        "failure_fixture_set_sha256": stable_hash(failures),
        "index_file_count": len(indexes),
        "index_set_sha256": stable_hash(indexes),
        "index_total_size_bytes": sum(row["size_bytes"] for row in indexes),
        "input_count": len(inputs),
        "input_set_sha256": stable_hash(inputs),
        "input_total_size_bytes": sum(row["size_bytes"] for row in inputs),
        "sources": source_rows,
        "success_output_count": len(outputs),
        "success_output_set_sha256": stable_hash(outputs),
        "success_output_total_size_bytes": sum(
            row["size_bytes"] for row in outputs
        ),
    }
    if observed != design.get("cached_fixture_contract"):
        raise HippoRAGTotalityQualificationError("cached fixture set drifted")
    return observed


def _verify_and_materialize_source(base: Path, root: Path) -> dict[str, Any]:
    repo = base / BASELINE_REPO_RELATIVE
    baseline_path = repo / BASELINE_SOURCE_WITHIN_REPO
    if (
        _git_output(["rev-parse", "HEAD"], repo).decode().strip()
        != upstream_hardening.BASELINE_COMMIT
        or _git_output(["status", "--porcelain", "--untracked-files=no"], repo)
        != b""
        or _git_output(["remote", "get-url", "origin"], repo).decode().strip()
        != "https://github.com/OSU-NLP-Group/HippoRAG.git"
    ):
        raise HippoRAGTotalityQualificationError(
            "baseline official repository drifted"
        )
    baseline = baseline_path.read_bytes()
    qualified_source = upstream_hardening.apply_fixed_backport(baseline)
    if hashlib.sha256(qualified_source).hexdigest() != backport.INPUT_SOURCE_SHA256:
        raise HippoRAGTotalityQualificationError("qualified input source drifted")
    patched = backport.apply_totality_hardening(qualified_source)
    patch = backport.unified_patch_bytes(qualified_source, patched)
    patched_root = root / "patched_source"
    patched_root.mkdir(mode=0o700)
    input_path = patched_root / "HippoRAG.qualified_nonfinite.py"
    patched_path = patched_root / "HippoRAG.py"
    patch_path = patched_root / "HippoRAG.zero_weight_totality.patch"
    _write_exclusive(input_path, qualified_source)
    _write_exclusive(patched_path, patched)
    _write_exclusive(patch_path, patch)
    return {
        "baseline_commit": upstream_hardening.BASELINE_COMMIT,
        "baseline_source_sha256": file_sha256(baseline_path),
        "input_source_sha256": file_sha256(input_path),
        "patched_path": patched_path,
        "patched_source_sha256": file_sha256(patched_path),
        "unified_patch_sha256": file_sha256(patch_path),
    }


def _bubblewrap_prefix(
    *, base: Path, writable_root: Path, patched_source: Path
) -> list[str]:
    baseline_source = (
        base / BASELINE_REPO_RELATIVE / BASELINE_SOURCE_WITHIN_REPO
    ).resolve(strict=True)
    return [
        "/usr/bin/bwrap",
        "--die-with-parent",
        "--unshare-all",
        "--new-session",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/tmp/models",
        "--ro-bind",
        str(base / LLM_MODEL_RELATIVE),
        "/tmp/models/llm",
        "--ro-bind",
        str(base / EMBEDDING_MODEL_RELATIVE),
        "/tmp/models/embed",
        "--ro-bind",
        str(patched_source),
        str(baseline_source),
        "--bind",
        str(writable_root),
        str(writable_root),
        "--chdir",
        str(base),
        "--setenv",
        "CUDA_VISIBLE_DEVICES",
        "",
        "--setenv",
        "HF_HOME",
        str(writable_root / "hf"),
        "--setenv",
        "HF_HUB_OFFLINE",
        "1",
        "--setenv",
        "HOME",
        str(writable_root / "home"),
        "--setenv",
        "MPLCONFIGDIR",
        str(writable_root / "tmp" / "mpl"),
        "--setenv",
        "OMP_NUM_THREADS",
        "2",
        "--setenv",
        "TOKENIZERS_PARALLELISM",
        "false",
        "--setenv",
        "TMPDIR",
        str(writable_root / "tmp"),
        "--setenv",
        "TRANSFORMERS_OFFLINE",
        "1",
        str(base / RUNTIME_PYTHON_RELATIVE),
        "-I",
        "-B",
    ]


def _prepare_writable_root(path: Path) -> None:
    path.mkdir(mode=0o700)
    for name in ("home", "hf", "tmp"):
        (path / name).mkdir(mode=0o700)


def _run_synthetic(base: Path, root: Path, patched_source: Path) -> dict[str, Any]:
    synthetic_root = root / "synthetic"
    _prepare_writable_root(synthetic_root)
    output = synthetic_root / "output.json"
    command = _bubblewrap_prefix(
        base=base,
        writable_root=synthetic_root,
        patched_source=patched_source,
    ) + [
        "-m",
        "replication_runtime.hipporag_zero_weight_totality_v1.synthetic_worker",
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command, cwd=base, check=False, capture_output=True, timeout=300
    )
    _write_exclusive(synthetic_root / "stdout.log", completed.stdout)
    _write_exclusive(synthetic_root / "stderr.log", completed.stderr)
    if completed.returncode != 0:
        raise HippoRAGTotalityQualificationError(
            "synthetic worker failed: "
            + hashlib.sha256(completed.stderr).hexdigest()
        )
    value = _read_json(output, "synthetic totality output")
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
        raise HippoRAGTotalityQualificationError(
            "synthetic totality result drifted"
        )
    return {
        "output_file_sha256": file_sha256(output),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        **value,
    }


def _parse_worker_output(path: Path) -> Mapping[str, Any]:
    value = _read_json(path, "cached worker output")
    top = value.get("top_ordinals")
    if (
        value.get("schema")
        != "bright_official_hipporag_candidate_retrieval_v1_output"
        or not isinstance(top, list)
        or len(top) != 10
        or len(set(top)) != 10
        or any(isinstance(row, bool) or not isinstance(row, int) for row in top)
        or not isinstance(value.get("graph_node_count"), int)
        or not isinstance(value.get("graph_edge_count"), int)
    ):
        raise HippoRAGTotalityQualificationError("cached worker output drifted")
    return value


def _run_cached_item(
    *,
    base: Path,
    root: Path,
    patched_source: Path,
    source: FixtureSource,
    ordinal: int,
    counter: _ConcurrencyCounter,
) -> dict[str, Any]:
    original = base / source.root_relative / f"item_{ordinal:03d}"
    item = root / "cached" / source.key / f"item_{ordinal:03d}"
    _prepare_writable_root(item)
    shutil.copy2(original / "input.json", item / "input.json")
    shutil.copytree(original / "index", item / "index")
    output = item / "output.json"
    command = _bubblewrap_prefix(
        base=base,
        writable_root=item,
        patched_source=patched_source,
    ) + [
        "-m",
        "replication_runtime.hipporag_zero_weight_totality_v1.cached_worker",
        "--input",
        str(item / "input.json"),
        "--output",
        str(output),
        "--index-root",
        str(item / "index"),
        "--llm-model",
        "/tmp/models/llm",
        "--embedding-model",
        "/tmp/models/embed",
    ]
    counter.enter()
    try:
        completed = subprocess.run(
            command, cwd=base, check=False, capture_output=True, timeout=900
        )
    finally:
        counter.leave()
    _write_exclusive(item / "stdout.log", completed.stdout)
    _write_exclusive(item / "stderr.log", completed.stderr)
    if completed.returncode != 0:
        raise HippoRAGTotalityQualificationError(
            f"cached item {source.key}/{ordinal} failed: "
            + hashlib.sha256(completed.stderr).hexdigest()
        )
    value = _parse_worker_output(output)
    was_failure = ordinal in source.failure_ordinals
    if was_failure:
        if value["graph_node_count"] <= 0 or value["graph_edge_count"] <= 0:
            raise HippoRAGTotalityQualificationError(
                "repaired failure fixture graph is empty"
            )
        disposition = "previous_failure_now_structurally_valid"
    else:
        expected = (original / "output.json").read_bytes()
        observed = output.read_bytes()
        if observed != expected:
            raise HippoRAGTotalityQualificationError(
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


def _run_cached_set(
    base: Path, root: Path, patched_source: Path
) -> tuple[list[dict[str, Any]], int]:
    (root / "cached").mkdir(mode=0o700)
    counter = _ConcurrencyCounter()
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    work = [
        (source, ordinal)
        for source in FIXTURE_SOURCES
        for ordinal in range(source.item_count)
    ]
    with ThreadPoolExecutor(max_workers=PROCESS_CONCURRENCY) as executor:
        futures: dict[Future[dict[str, Any]], tuple[str, int]] = {
            executor.submit(
                _run_cached_item,
                base=base,
                root=root,
                patched_source=patched_source,
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
        or not 0 < counter.peak <= PROCESS_CONCURRENCY
        or set(completed) != expected_keys
    ):
        raise HippoRAGTotalityQualificationError(
            "cached-set completion or concurrency drifted"
        )
    ordered = [
        completed[(source.key, ordinal)]
        for source in FIXTURE_SOURCES
        for ordinal in range(source.item_count)
    ]
    return ordered, counter.peak


def verify_offline_runtime_assets(base: Path) -> None:
    for directory in (base / LLM_MODEL_RELATIVE, base / EMBEDDING_MODEL_RELATIVE):
        if not directory.is_dir() or directory.is_symlink():
            raise HippoRAGTotalityQualificationError(
                "offline model asset is unavailable"
            )
    runtime = base / RUNTIME_PYTHON_RELATIVE
    if not runtime.is_file():
        raise HippoRAGTotalityQualificationError(
            "offline runtime Python is unavailable"
        )
    try:
        runtime.resolve(strict=True)
    except OSError as exc:
        raise HippoRAGTotalityQualificationError(
            "offline runtime Python is broken"
        ) from exc


def run_formal(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    result_path = base / RESULT_RELATIVE
    root = base / RUN_ROOT_RELATIVE
    if result_path.exists() or result_path.is_symlink():
        raise OneShotRefusal("totality qualification result already exists")
    if root.exists() or root.is_symlink():
        raise OneShotRefusal("totality qualification root already exists")
    freeze = _verify_design_and_freeze(base, project_root)
    artifact_binding = verify_frozen_artifact_sets(base)
    verify_offline_runtime_assets(base)
    root.mkdir(mode=0o700)
    source = _verify_and_materialize_source(base, root)
    synthetic = _run_synthetic(base, root, source["patched_path"])
    rows, peak = _run_cached_set(base, root, source["patched_path"])
    parity_count = sum(
        row["disposition"] == "previous_success_byte_identical" for row in rows
    )
    repair_count = sum(
        row["disposition"] == "previous_failure_now_structurally_valid"
        for row in rows
    )
    if parity_count != 58 or repair_count != 2:
        raise HippoRAGTotalityQualificationError(
            "cached-set disposition count drifted"
        )
    public_source = {
        key: value for key, value in source.items() if key != "patched_path"
    }
    result = self_hashed(
        {
            "artifact_binding": artifact_binding,
            "cached_retrieval": {
                "item_count": len(rows),
                "peak_process_concurrency": peak,
                "previous_failure_structurally_valid_count": repair_count,
                "previous_success_byte_identical_count": parity_count,
                "row_set_sha256": stable_hash(rows),
            },
            "claim_boundary": {
                "cached_fixture_input_and_index_read_count": 60,
                "current_failed_cohort_resumed_or_scored": False,
                "external_network_call_count": 0,
                "label_or_qrel_open_count": 0,
                "online_evaluator_call_count": 0,
                "performance_score_count": 0,
                "qualification_is_comparator_performance_evidence": False,
            },
            "formal_binding": {
                "design_self_sha256": DESIGN_SELF_SHA256,
                "formal_implementation_commit": freeze[
                    "formal_implementation_commit"
                ],
                "freeze_self_sha256": freeze["self_sha256"],
            },
            "recorded_date": "2026-07-21",
            "schema": SCHEMA,
            "source_hardening": public_source,
            "status": (
                "passed_totalized_official_comparator_qualified_for_"
                "future_separately_frozen_new_studies_only"
            ),
            "synthetic": synthetic,
        }
    )
    _write_json(result_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--verify-frozen-inputs", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.formal == arguments.verify_frozen_inputs:
        raise SystemExit("choose exactly one mode")
    if arguments.verify_frozen_inputs:
        value = verify_frozen_artifact_sets(
            arguments.project_root.resolve(strict=True) / "reconstruction_v2"
        )
    else:
        value = run_formal(arguments.project_root)
    print(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
