"""One-shot, non-scoring qualification of the upstream HippoRAG fix.

The current FiQA DEV cohort is never read.  The frozen two-edit official
backport is checked for byte provenance, exercised on a synthetic absent-node
fixture, and compared byte-for-byte on all twelve completed FiQA TRAIN cached
retrievals.
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
import threading
from typing import Any, Iterable, Mapping, Sequence

from reconstruction_v2.replication_runtime.hipporag_upstream_hardening_v1 import (
    backport,
)


SCHEMA = "hipporag_upstream_hardening_qualification_result_v1"
FREEZE_SCHEMA = "hipporag_upstream_hardening_qualification_freeze_v1_r1"
DESIGN_SCHEMA = "hipporag_upstream_hardening_qualification_design_v1"
DESIGN_SELF_SHA256 = (
    "ce2814f09297bc5b07dfa1657086f363a93fab213ed2b6e7ee761ea343482a39"
)
ITEM_COUNT = 12
PROCESS_CONCURRENCY = 2

DESIGN_RELATIVE = Path(
    "manifests/hipporag_upstream_hardening_qualification_design_v1.json"
)
FREEZE_RELATIVE = Path(
    "manifests/hipporag_upstream_hardening_qualification_freeze_v1_r1.json"
)
RESULT_RELATIVE = Path(
    "manifests/hipporag_upstream_hardening_qualification_result_v1.json"
)
RUN_ROOT_RELATIVE = Path("artifacts/hipporag_upstream_hardening_qualification_v1")
BASELINE_REPO_RELATIVE = Path(
    "reference/self_evo_continual_20260707/repos/HippoRAG"
)
BASELINE_SOURCE_WITHIN_REPO = Path("src/hipporag/HippoRAG.py")
TRAIN_HIPPO_ROOT_RELATIVE = Path(
    "artifacts/fiqa_bridge_expansion_train_runtime_v2/hipporag"
)
LLM_MODEL_RELATIVE = Path(
    "artifacts/bright_reasoning_retrieval_runtime_v1/smollm2_135m_instruct_exact"
)
EMBEDDING_MODEL_RELATIVE = Path("artifacts/qasper_minilm_runtime_v1/model")
RUNTIME_PYTHON_RELATIVE = Path(
    "artifacts/bright_reasoning_retrieval_runtime_v1/hipporag_venv/bin/python"
)

IMPLEMENTATION_RELATIVES = (
    Path("assumption_agent/benchmarks/hipporag_upstream_hardening_qualification_v1.py"),
    Path("replication_runtime/hipporag_upstream_hardening_v1/__init__.py"),
    Path("replication_runtime/hipporag_upstream_hardening_v1/backport.py"),
    Path("replication_runtime/hipporag_upstream_hardening_v1/cached_worker.py"),
    Path("replication_runtime/hipporag_upstream_hardening_v1/synthetic_worker.py"),
    Path("tests/test_hipporag_upstream_hardening_qualification_v1.py"),
)

FROZEN_INPUT_SET_SHA256 = (
    "703894336939887295663726163b5b0cfc3b8241ad64f051e4f3190ca4eb52af"
)
FROZEN_OUTPUT_SET_SHA256 = (
    "2924363c103b3dd1899c7d5948d21428a0ef797d05f6d772e7767eca6a423b72"
)
FROZEN_INDEX_SET_SHA256 = (
    "da0ef9bd91e1f196c8cead4dd7fc8f85a78fd6c6cc9457261f0afea81d6e8e9c"
)


class HippoRAGQualificationError(RuntimeError):
    """The frozen non-scoring qualification failed closed."""


class OneShotRefusal(HippoRAGQualificationError):
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
        raise HippoRAGQualificationError("value is not canonical JSON") from exc


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
        raise HippoRAGQualificationError(f"{name} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HippoRAGQualificationError(f"{name} is invalid") from exc
    if not isinstance(value, Mapping):
        raise HippoRAGQualificationError(f"{name} is not an object")
    return value


def verify_self_hash(value: Mapping[str, Any], field: str = "self_sha256") -> str:
    declared = value.get(field)
    if not isinstance(declared, str) or len(declared) != 64:
        raise HippoRAGQualificationError("self hash is absent")
    body = dict(value)
    body.pop(field, None)
    observed = stable_hash(body)
    if observed != declared:
        raise HippoRAGQualificationError("self hash drifted")
    return declared


def self_hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if "self_sha256" in result:
        raise HippoRAGQualificationError("self hash field already exists")
    result["self_sha256"] = stable_hash(result)
    return result


def _git_output(arguments: Sequence[str], cwd: Path) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise HippoRAGQualificationError(
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
    design = _read_json(base / DESIGN_RELATIVE, "hardening design")
    if (
        design.get("schema") != DESIGN_SCHEMA
        or verify_self_hash(design) != DESIGN_SELF_SHA256
    ):
        raise HippoRAGQualificationError("hardening design identity drifted")
    freeze = _read_json(base / FREEZE_RELATIVE, "hardening freeze")
    if freeze.get("schema") != FREEZE_SCHEMA:
        raise HippoRAGQualificationError("hardening freeze schema drifted")
    verify_self_hash(freeze)
    if freeze.get("design_self_sha256") != DESIGN_SELF_SHA256:
        raise HippoRAGQualificationError("hardening freeze design binding drifted")
    expected_commit = freeze.get("formal_implementation_commit")
    if (
        not isinstance(expected_commit, str)
        or not _git_succeeds(
            ["merge-base", "--is-ancestor", expected_commit, "HEAD"], project_root
        )
    ):
        raise HippoRAGQualificationError("formal implementation commit drifted")
    rows = freeze.get("implementation_bindings")
    if not isinstance(rows, list):
        raise HippoRAGQualificationError("hardening freeze bindings are absent")
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in rows
        if isinstance(row, Mapping)
    }
    expected_paths = {path.as_posix() for path in IMPLEMENTATION_RELATIVES}
    if set(observed) != expected_paths:
        raise HippoRAGQualificationError("hardening freeze file set drifted")
    for relative, expected in observed.items():
        path = base / str(relative)
        if not isinstance(expected, str) or file_sha256(path) != expected:
            raise HippoRAGQualificationError("hardening implementation drifted")
    return freeze


def _artifact_rows(base: Path) -> tuple[list[dict[str, Any]], ...]:
    root = base / TRAIN_HIPPO_ROOT_RELATIVE
    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    indexes: list[dict[str, Any]] = []
    for ordinal in range(ITEM_COUNT):
        item = root / f"item_{ordinal:03d}"
        for name, destination in (("input.json", inputs), ("output.json", outputs)):
            path = item / name
            if path.is_symlink() or not path.is_file():
                raise HippoRAGQualificationError("frozen TRAIN parity file is absent")
            destination.append(
                {
                    "ordinal": ordinal,
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        index = item / "index"
        files = sorted(path for path in index.rglob("*") if path.is_file())
        if not files or any(path.is_symlink() for path in files):
            raise HippoRAGQualificationError("frozen TRAIN cached index drifted")
        for path in files:
            indexes.append(
                {
                    "ordinal": ordinal,
                    "relative_path": path.relative_to(index).as_posix(),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return inputs, outputs, indexes


def verify_frozen_artifact_sets(base: Path) -> dict[str, Any]:
    inputs, outputs, indexes = _artifact_rows(base)
    observed = {
        "cached_index_file_count": len(indexes),
        "cached_index_set_sha256": stable_hash(indexes),
        "cached_index_total_size_bytes": sum(row["size_bytes"] for row in indexes),
        "input_count": len(inputs),
        "input_set_sha256": stable_hash(inputs),
        "input_total_size_bytes": sum(row["size_bytes"] for row in inputs),
        "output_count": len(outputs),
        "output_set_sha256": stable_hash(outputs),
        "output_total_size_bytes": sum(row["size_bytes"] for row in outputs),
    }
    expected = {
        "cached_index_file_count": 72,
        "cached_index_set_sha256": FROZEN_INDEX_SET_SHA256,
        "cached_index_total_size_bytes": 3537837,
        "input_count": 12,
        "input_set_sha256": FROZEN_INPUT_SET_SHA256,
        "input_total_size_bytes": 316896,
        "output_count": 12,
        "output_set_sha256": FROZEN_OUTPUT_SET_SHA256,
        "output_total_size_bytes": 1884,
    }
    if observed != expected:
        raise HippoRAGQualificationError("frozen TRAIN artifact set drifted")
    return observed


def _verify_and_materialize_source(base: Path, root: Path) -> dict[str, Any]:
    repo = base / BASELINE_REPO_RELATIVE
    baseline_path = repo / BASELINE_SOURCE_WITHIN_REPO
    if (
        _git_output(["rev-parse", "HEAD"], repo).decode().strip()
        != backport.BASELINE_COMMIT
        or _git_output(["status", "--porcelain", "--untracked-files=no"], repo)
        != b""
        or _git_output(["remote", "get-url", "origin"], repo).decode().strip()
        != "https://github.com/OSU-NLP-Group/HippoRAG.git"
    ):
        raise HippoRAGQualificationError("baseline official repository drifted")
    baseline = baseline_path.read_bytes()
    upstream = _git_output(
        ["show", f"{backport.UPSTREAM_COMMIT}:src/hipporag/HippoRAG.py"], repo
    )
    backport.verify_upstream_contains_backport(upstream)
    patched = backport.apply_fixed_backport(baseline)
    patch = backport.unified_patch_bytes(baseline, patched)
    patched_root = root / "patched_source"
    patched_root.mkdir(mode=0o700)
    patched_path = patched_root / "HippoRAG.py"
    patch_path = patched_root / "HippoRAG.upstream_backport.patch"
    _write_exclusive(patched_path, patched)
    _write_exclusive(patch_path, patch)
    return {
        "baseline_commit": backport.BASELINE_COMMIT,
        "baseline_source_sha256": file_sha256(baseline_path),
        "patched_path": patched_path,
        "patched_source_sha256": file_sha256(patched_path),
        "unified_patch_sha256": file_sha256(patch_path),
        "upstream_commit": backport.UPSTREAM_COMMIT,
        "upstream_source_sha256": hashlib.sha256(upstream).hexdigest(),
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
        base=base, writable_root=synthetic_root, patched_source=patched_source
    ) + [
        "-m",
        "replication_runtime.hipporag_upstream_hardening_v1.synthetic_worker",
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, cwd=base, check=False, capture_output=True, timeout=300)
    _write_exclusive(synthetic_root / "stdout.log", completed.stdout)
    _write_exclusive(synthetic_root / "stderr.log", completed.stderr)
    if completed.returncode != 0:
        raise HippoRAGQualificationError(
            "synthetic worker failed: " + hashlib.sha256(completed.stderr).hexdigest()
        )
    value = _read_json(output, "synthetic hardening output")
    if (
        value.get("schema")
        != "hipporag_upstream_hardening_synthetic_fixture_v1"
        or value.get("source_sha256") != backport.PATCHED_SOURCE_SHA256
        or value.get("baseline_nonfinite_count") != 2
        or value.get("hardened_nonfinite_count") != 0
        or value.get("ranked_passage_rows") != [0, 1]
    ):
        raise HippoRAGQualificationError("synthetic hardening result drifted")
    return {
        "baseline_nonfinite_count": 2,
        "hardened_nonfinite_count": 0,
        "output_file_sha256": file_sha256(output),
        "ranked_passage_rows": [0, 1],
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
    }


def _run_parity_item(
    *,
    base: Path,
    root: Path,
    patched_source: Path,
    ordinal: int,
    semaphore: threading.Semaphore,
    counter: _ConcurrencyCounter,
) -> dict[str, Any]:
    source = base / TRAIN_HIPPO_ROOT_RELATIVE / f"item_{ordinal:03d}"
    item = root / "parity" / f"item_{ordinal:03d}"
    _prepare_writable_root(item)
    shutil.copy2(source / "input.json", item / "input.json")
    shutil.copytree(source / "index", item / "index")
    with semaphore:
        counter.enter()
        try:
            output = item / "output.json"
            command = _bubblewrap_prefix(
                base=base, writable_root=item, patched_source=patched_source
            ) + [
                "-m",
                "replication_runtime.hipporag_upstream_hardening_v1.cached_worker",
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
            completed = subprocess.run(
                command, cwd=base, check=False, capture_output=True, timeout=900
            )
        finally:
            counter.leave()
    _write_exclusive(item / "stdout.log", completed.stdout)
    _write_exclusive(item / "stderr.log", completed.stderr)
    if completed.returncode != 0:
        raise HippoRAGQualificationError(
            f"cached parity item {ordinal} failed: "
            + hashlib.sha256(completed.stderr).hexdigest()
        )
    expected = (source / "output.json").read_bytes()
    observed = output.read_bytes()
    if observed != expected:
        raise HippoRAGQualificationError(
            f"cached parity item {ordinal} output changed"
        )
    return {
        "ordinal": ordinal,
        "output_sha256": hashlib.sha256(observed).hexdigest(),
        "output_size_bytes": len(observed),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
    }


def _run_parity(
    base: Path, root: Path, patched_source: Path
) -> tuple[list[dict[str, Any]], int]:
    (root / "parity").mkdir(mode=0o700)
    semaphore = threading.Semaphore(PROCESS_CONCURRENCY)
    counter = _ConcurrencyCounter()
    completed: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=PROCESS_CONCURRENCY) as executor:
        futures: dict[Future[dict[str, Any]], int] = {
            executor.submit(
                _run_parity_item,
                base=base,
                root=root,
                patched_source=patched_source,
                ordinal=ordinal,
                semaphore=semaphore,
                counter=counter,
            ): ordinal
            for ordinal in range(ITEM_COUNT)
        }
        for future in as_completed(futures):
            ordinal = futures[future]
            completed[ordinal] = future.result()
    if counter.current != 0 or counter.peak > PROCESS_CONCURRENCY:
        raise HippoRAGQualificationError("cached parity concurrency drifted")
    if set(completed) != set(range(ITEM_COUNT)):
        raise HippoRAGQualificationError("cached parity completion drifted")
    return [completed[index] for index in range(ITEM_COUNT)], counter.peak


def verify_offline_runtime_assets(base: Path) -> None:
    for directory in (base / LLM_MODEL_RELATIVE, base / EMBEDDING_MODEL_RELATIVE):
        if not directory.is_dir() or directory.is_symlink():
            raise HippoRAGQualificationError("offline model asset is unavailable")
    runtime = base / RUNTIME_PYTHON_RELATIVE
    if not runtime.is_file():
        raise HippoRAGQualificationError("offline runtime Python is unavailable")
    try:
        runtime.resolve(strict=True)
    except OSError as exc:
        raise HippoRAGQualificationError("offline runtime Python is broken") from exc


def run_formal(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    result_path = base / RESULT_RELATIVE
    root = base / RUN_ROOT_RELATIVE
    if result_path.exists() or result_path.is_symlink():
        raise OneShotRefusal("hardening qualification result already exists")
    if root.exists() or root.is_symlink():
        raise OneShotRefusal("hardening qualification root already exists")
    freeze = _verify_design_and_freeze(base, project_root)
    artifact_binding = verify_frozen_artifact_sets(base)
    verify_offline_runtime_assets(base)
    root.mkdir(mode=0o700)
    source = _verify_and_materialize_source(base, root)
    synthetic = _run_synthetic(base, root, source["patched_path"])
    parity_rows, peak = _run_parity(base, root, source["patched_path"])
    public_source = {key: value for key, value in source.items() if key != "patched_path"}
    result = self_hashed(
        {
            "artifact_binding": artifact_binding,
            "claim_boundary": {
                "current_FiQA_DEV_input_index_or_query_open_count": 0,
                "FiQA_DEV_label_open_count": 0,
                "FiQA_TEST_label_open_count": 0,
                "online_evaluator_call_count": 0,
                "performance_score_count": 0,
                "qualification_is_non_scoring": True,
            },
            "formal_binding": {
                "design_self_sha256": DESIGN_SELF_SHA256,
                "formal_implementation_commit": freeze["formal_implementation_commit"],
                "freeze_self_sha256": freeze["self_sha256"],
            },
            "parity": {
                "byte_identical_count": len(parity_rows),
                "item_count": ITEM_COUNT,
                "peak_process_concurrency": peak,
                "row_set_sha256": stable_hash(parity_rows),
            },
            "recorded_date": "2026-07-20",
            "schema": SCHEMA,
            "source_backport": public_source,
            "status": "passed_upstream_fixed_comparator_qualified_for_future_new_studies_only",
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
