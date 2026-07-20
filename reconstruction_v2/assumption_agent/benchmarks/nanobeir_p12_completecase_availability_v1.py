"""Single label-free HippoRAG availability screen for the final P12 sources.

Every public query is embedded once against its family corpus to form the
frozen RAW top-32 pool.  The already-qualified upstream HippoRAG source then
runs once per query.  Failures are retained as availability failures, never
retried, and never used to change P12.  This program does not read qrels.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
from typing import Any, Mapping, Sequence

import numpy as np

from reconstruction_v2.assumption_agent.benchmarks import (
    bright_reasoning_retrieval_core_v1 as core,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    fiqa_bridge_expansion_train_runtime_v1 as train,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    hipporag_upstream_hardening_qualification_v1 as hardening_qualification,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p11_c_confirm_runtime_v1 as p11_runtime,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    nanobeir_p12_acquisition_v1 as utilities,
)
from reconstruction_v2.replication_runtime.hipporag_upstream_hardening_v1 import (
    backport,
)


SCHEMA = "nanobeir_p12_completecase_availability_result_v1"
ATTEMPT_SCHEMA = "nanobeir_p12_completecase_availability_attempt_v1"
PACK_SCHEMA = "nanobeir_p12_completecase_availability_private_pack_v1"
FREEZE_SCHEMA = "nanobeir_p12_completecase_availability_freeze_v1"
FAMILIES = ("NanoArguAna", "NanoFEVER", "NanoSciFact")
ITEMS_PER_FAMILY = 50
ITEM_COUNT = 150
MINIMUM_ELIGIBLE_PER_FAMILY = 36
PROCESS_CONCURRENCY = 12
DOCUMENT_PROJECTION_CHARACTERS = 3000

SOURCE_ROOT_RELATIVE = Path(
    "artifacts/nanobeir_p12_completecase_source_v1/dataset"
)
RUN_ROOT_RELATIVE = Path(
    "artifacts/nanobeir_p12_completecase_availability_v1"
)
RESULT_RELATIVE = Path(
    "manifests/nanobeir_p12_completecase_availability_result_v1.json"
)
FREEZE_RELATIVE = Path(
    "manifests/nanobeir_p12_completecase_availability_freeze_v1.json"
)
IMPLEMENTATION_RELATIVE = Path(
    "assumption_agent/benchmarks/nanobeir_p12_completecase_availability_v1.py"
)
TEST_RELATIVE = Path(
    "tests/test_nanobeir_p12_completecase_availability_v1.py"
)

PRECONDITIONS = {
    "candidate": {
        "relative": "manifests/nanobeir_p12_candidate_freeze_v1.json",
        "file_sha256": "156343887ddda849f24e3d29fc8e577585e46e86b9f24473d2b953c22773f519",
        "self_sha256": "2421b8c9fec755f6a7087621771b376dd77a4a726ef23ee8c248268044a5bd9e",
    },
    "design": {
        "relative": "manifests/nanobeir_p12_completecase_study_design_v2.json",
        "file_sha256": "d6a4082eff394c8ec34c7dec782a637f48723ac8673c59c46f3c4a4877c8f132",
        "self_sha256": "758563170e7c51a1ee503f4da53ef7d63f452711f7c85b7c74188b356ba4ad8f",
    },
    "hardening": {
        "relative": "manifests/hipporag_upstream_hardening_qualification_result_v1.json",
        "file_sha256": "55f8295539d2e6b1d6c776cf8c7c7e2b7ac6bccbda89aebf5d447f55de854da5",
        "self_sha256": "2c9571cf7437d47d6e0dad3317841f77ebc0b782132b945185514f106c0ed8a3",
    },
    "source_access": {
        "relative": "manifests/nanobeir_p12_completecase_source_access_v2.json",
        "file_sha256": "d1bb90f29d1bf2933200f1da8724eb10cb1d850911f68df2ec1d85142639d598",
        "self_sha256": "95566f6be6785b6c12208575925920937295a2aa2457a5fb62f3689545dd60fc",
    },
}

SOURCE_FILES = {
    "corpus/NanoArguAna-00000-of-00001.parquet": "9ae34836d485a4b1f83fd0605f444e8f3f0677a375cb6c8b542904a7b4c904a9",
    "corpus/NanoFEVER-00000-of-00001.parquet": "44ba16cc807bab801c3e45df054c9b711a6b58260e3a18d76e91f3ef160d1c4e",
    "corpus/NanoSciFact-00000-of-00001.parquet": "ece958f761f10a813a3ca17bfb63455629b110116abf00a3f51396b69664d358",
    "queries/NanoArguAna-00000-of-00001.parquet": "e0dfec4d3c90355e0d11bc8e50a54622a533ee1c8feeb89ece700a7134d8b237",
    "queries/NanoFEVER-00000-of-00001.parquet": "bdaba371eb97e4ed2a8d7a59446197d708993402abef0a584791d2717c9b2a51",
    "queries/NanoSciFact-00000-of-00001.parquet": "0b621d5ea9ba032b88a52c4ea894f98114cc6eb5d6a999520ada0c6bee9c6f27",
}

REQUIRED_IMPLEMENTATION_RELATIVES = (
    IMPLEMENTATION_RELATIVE,
    TEST_RELATIVE,
    Path("assumption_agent/benchmarks/nanobeir_p11_c_confirm_runtime_v1.py"),
    Path("replication_runtime/hipporag_upstream_hardening_v1/backport.py"),
    Path("replication_runtime/bright_official_hipporag_v1/contract.py"),
    Path("replication_runtime/bright_official_hipporag_v1/worker.py"),
)


class NanoBEIRAvailabilityError(RuntimeError):
    """The frozen label-free availability screen failed closed."""


class OneShotRefusal(NanoBEIRAvailabilityError):
    """The formal screen root or result is already consumed."""


@dataclass(frozen=True)
class SourceFamily:
    ids: tuple[str, ...]
    contents: tuple[str, ...]
    query_ids: tuple[str, ...]
    queries: tuple[str, ...]


@dataclass(frozen=True)
class ScreenItem:
    ordinal: int
    family: str
    family_ordinal: int
    query_id: str
    query: str
    base_pool: tuple[int, ...]
    raw_top10: tuple[int, ...]


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise NanoBEIRAvailabilityError(f"{name} is unavailable")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NanoBEIRAvailabilityError(f"{name} is invalid") from exc
    if not isinstance(value, Mapping):
        raise NanoBEIRAvailabilityError(f"{name} is not an object")
    return value


def _verify_self(value: Mapping[str, Any], expected: str) -> None:
    body = dict(value)
    declared = body.pop("self_sha256", None)
    if declared != expected or utilities.stable_hash(body) != expected:
        raise NanoBEIRAvailabilityError("manifest self hash drifted")


def _verify_preconditions(base: Path) -> Mapping[str, Any]:
    loaded: dict[str, Any] = {}
    for name, binding in PRECONDITIONS.items():
        path = base / binding["relative"]
        if utilities.file_sha256(path) != binding["file_sha256"]:
            raise NanoBEIRAvailabilityError(f"{name} manifest file drifted")
        value = _read_json(path, name)
        _verify_self(value, binding["self_sha256"])
        loaded[name] = value
    if loaded["source_access"].get("qualification", {}).get("source_passed") is not True:
        raise NanoBEIRAvailabilityError("source qualification did not pass")
    if loaded["hardening"].get("status") != (
        "passed_upstream_fixed_comparator_qualified_for_future_new_studies_only"
    ):
        raise NanoBEIRAvailabilityError("upstream hardening is not qualified")
    for relative, expected in SOURCE_FILES.items():
        path = base / SOURCE_ROOT_RELATIVE / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or utilities.file_sha256(path) != expected
        ):
            raise NanoBEIRAvailabilityError("pinned source file drifted")
    return loaded


def _git_succeeds(arguments: Sequence[str], cwd: Path) -> bool:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def _verify_freeze(base: Path, project_root: Path) -> Mapping[str, Any]:
    value = _read_json(base / FREEZE_RELATIVE, "availability freeze")
    if value.get("schema") != FREEZE_SCHEMA:
        raise NanoBEIRAvailabilityError("availability freeze schema drifted")
    declared = value.get("self_sha256")
    if not isinstance(declared, str):
        raise NanoBEIRAvailabilityError("availability freeze hash is absent")
    _verify_self(value, declared)
    if value.get("study_design_self_sha256") != PRECONDITIONS["design"]["self_sha256"]:
        raise NanoBEIRAvailabilityError("freeze design binding drifted")
    commit = value.get("formal_implementation_commit")
    if (
        not isinstance(commit, str)
        or not _git_succeeds(["merge-base", "--is-ancestor", commit, "HEAD"], project_root)
    ):
        raise NanoBEIRAvailabilityError("formal implementation commit drifted")
    rows = value.get("implementation_bindings")
    if not isinstance(rows, list):
        raise NanoBEIRAvailabilityError("freeze bindings are absent")
    observed = {
        row.get("relative_path"): row.get("sha256")
        for row in rows
        if isinstance(row, Mapping)
    }
    expected_paths = {path.as_posix() for path in REQUIRED_IMPLEMENTATION_RELATIVES}
    if set(observed) != expected_paths:
        raise NanoBEIRAvailabilityError("freeze file set drifted")
    for relative, expected in observed.items():
        if not isinstance(expected, str) or utilities.file_sha256(
            base / str(relative)
        ) != expected:
            raise NanoBEIRAvailabilityError("frozen implementation drifted")
    return value


def _required_text(value: object, name: str, maximum: int | None = None) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or (maximum is not None and len(value) > maximum)
    ):
        raise NanoBEIRAvailabilityError(f"{name} is invalid")
    return value


def load_sources(base: Path) -> Mapping[str, SourceFamily]:
    """Read only corpus and query members; qrel members are outside this function."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise NanoBEIRAvailabilityError("pyarrow is unavailable") from exc
    root = base / SOURCE_ROOT_RELATIVE
    output: dict[str, SourceFamily] = {}
    for family in FAMILIES:
        corpus_path = root / "corpus" / f"{family}-00000-of-00001.parquet"
        query_path = root / "queries" / f"{family}-00000-of-00001.parquet"
        corpus_table = pq.read_table(corpus_path)
        query_table = pq.read_table(query_path)
        if corpus_table.column_names != ["_id", "text"]:
            raise NanoBEIRAvailabilityError("corpus schema drifted")
        if query_table.column_names != ["_id", "text"]:
            raise NanoBEIRAvailabilityError("query schema drifted")
        ids: list[str] = []
        contents: list[str] = []
        for row in corpus_table.to_pylist():
            identifier = _required_text(row.get("_id"), "corpus ID")
            content = _required_text(row.get("text"), "corpus text")[
                :DOCUMENT_PROJECTION_CHARACTERS
            ]
            if identifier in ids:
                raise NanoBEIRAvailabilityError("duplicate corpus ID")
            ids.append(identifier)
            contents.append(content)
        query_ids: list[str] = []
        queries: list[str] = []
        for row in query_table.to_pylist():
            identifier = _required_text(row.get("_id"), "query ID")
            query = _required_text(row.get("text"), "query", 24_000)
            if identifier in query_ids or query in queries:
                raise NanoBEIRAvailabilityError("duplicate query ID or text")
            query_ids.append(identifier)
            queries.append(query)
        if len(query_ids) != ITEMS_PER_FAMILY or len(ids) < 64:
            raise NanoBEIRAvailabilityError("source capacity drifted")
        output[family] = SourceFamily(
            ids=tuple(ids),
            contents=tuple(contents),
            query_ids=tuple(query_ids),
            queries=tuple(queries),
        )
    return output


def build_screen_items(
    sources: Mapping[str, SourceFamily],
    corpus_embeddings: Mapping[str, np.ndarray],
    query_embeddings: Mapping[str, np.ndarray],
) -> tuple[ScreenItem, ...]:
    items: list[ScreenItem] = []
    ordinal = 0
    for family in FAMILIES:
        source = sources[family]
        corpus_matrix = np.asarray(corpus_embeddings[family], dtype=np.float32)
        query_matrix = np.asarray(query_embeddings[family], dtype=np.float32)
        if corpus_matrix.shape != (len(source.ids), 384):
            raise NanoBEIRAvailabilityError("corpus embedding shape drifted")
        if query_matrix.shape != (ITEMS_PER_FAMILY, 384):
            raise NanoBEIRAvailabilityError("query embedding shape drifted")
        if not np.isfinite(corpus_matrix).all() or not np.isfinite(query_matrix).all():
            raise NanoBEIRAvailabilityError("embedding contains a nonfinite value")
        for family_ordinal, (query_id, query) in enumerate(
            zip(source.query_ids, source.queries)
        ):
            scores = train.quantized_scores(
                corpus_matrix, query_matrix[family_ordinal]
            )
            try:
                local = core.build_local_retrieval([scores])
            except core.BrightStudyCoreError as exc:
                raise NanoBEIRAvailabilityError(str(exc)) from exc
            items.append(
                ScreenItem(
                    ordinal=ordinal,
                    family=family,
                    family_ordinal=family_ordinal,
                    query_id=query_id,
                    query=query,
                    base_pool=local.candidate_rows,
                    raw_top10=local.raw_rows,
                )
            )
            ordinal += 1
    if len(items) != ITEM_COUNT:
        raise NanoBEIRAvailabilityError("screen item count drifted")
    return tuple(items)


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _materialize_hardened_source(base: Path, root: Path) -> Path:
    baseline = (
        base
        / hardening_qualification.BASELINE_REPO_RELATIVE
        / hardening_qualification.BASELINE_SOURCE_WITHIN_REPO
    )
    patched = backport.apply_fixed_backport(baseline.read_bytes())
    patched_root = root / "patched_source"
    patched_root.mkdir(mode=0o700)
    path = patched_root / "HippoRAG.py"
    utilities._write_exclusive(path, patched, mode=0o600)
    if utilities.file_sha256(path) != backport.PATCHED_SOURCE_SHA256:
        raise NanoBEIRAvailabilityError("materialized hardening drifted")
    return path


def _prepare_item_roots(
    *, root: Path, items: Sequence[ScreenItem], sources: Mapping[str, SourceFamily]
) -> Mapping[int, Path]:
    hippo_root = root / "hipporag"
    hippo_root.mkdir(mode=0o700)
    result: dict[int, Path] = {}
    for item in items:
        item_root = hippo_root / f"item_{item.ordinal:03d}"
        item_root.mkdir(mode=0o700)
        for name in ("home", "hf", "tmp"):
            (item_root / name).mkdir(mode=0o700)
        contents = sources[item.family].contents
        payload = {
            "documents": [
                {"content": contents[row], "ordinal": position}
                for position, row in enumerate(item.base_pool)
            ],
            "query": item.query,
            "schema": train.hippo_contract.INPUT_SCHEMA,
        }
        train.hippo_contract.validate_input(payload["query"], payload["documents"])
        train.bright_runtime._write_json(item_root / "input.json", payload)
        result[item.ordinal] = item_root
    expected = {item.ordinal for item in items}
    if len(expected) != len(items) or set(result) != expected:
        raise NanoBEIRAvailabilityError("HippoRAG item-root set drifted")
    return result


def _failure_row(item: ScreenItem, item_root: Path, exc: Exception) -> dict[str, Any]:
    stderr = item_root / "stderr.log"
    stdout = item_root / "stdout.log"
    return {
        "availability": "failed",
        "base_pool": list(item.base_pool),
        "exception_class": type(exc).__name__,
        "family": item.family,
        "family_ordinal": item.family_ordinal,
        "ordinal": item.ordinal,
        "query": item.query,
        "query_id": item.query_id,
        "raw_top10": list(item.raw_top10),
        "stderr_sha256": (
            utilities.file_sha256(stderr) if stderr.is_file() else None
        ),
        "stdout_sha256": (
            utilities.file_sha256(stdout) if stdout.is_file() else None
        ),
    }


def _success_row(
    item: ScreenItem, value: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "availability": "terminal",
        "base_pool": list(item.base_pool),
        "family": item.family,
        "family_ordinal": item.family_ordinal,
        "graph_edge_count": value["graph_edge_count"],
        "graph_node_count": value["graph_node_count"],
        "HippoRAG_output_file_sha256": value["output_file_sha256"],
        "HippoRAG_top_rows": list(value["top_rows"]),
        "ordinal": item.ordinal,
        "query": item.query,
        "query_id": item.query_id,
        "raw_top10": list(item.raw_top10),
        "stderr_sha256": value["stderr_sha256"],
        "stdout_sha256": value["stdout_sha256"],
    }


def _run_all(
    *,
    base: Path,
    items: Sequence[ScreenItem],
    item_roots: Mapping[int, Path],
    patched_source: Path,
) -> tuple[list[dict[str, Any]], int]:
    semaphore = threading.Semaphore(PROCESS_CONCURRENCY)
    counter = train.bright_runtime._ConcurrencyCounter()
    rows: dict[int, dict[str, Any]] = {}
    updates = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        with ThreadPoolExecutor(max_workers=PROCESS_CONCURRENCY) as executor:
            futures: dict[Future[Mapping[str, Any]], ScreenItem] = {
                executor.submit(
                    p11_runtime._run_hardened_hipporag_item,
                    base=base,
                    item_root=item_roots[item.ordinal],
                    candidate_rows=item.base_pool,
                    patched_source=patched_source,
                    semaphore=semaphore,
                    counter=counter,
                ): item
                for item in items
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    rows[item.ordinal] = _success_row(item, future.result())
                except Exception as exc:
                    rows[item.ordinal] = _failure_row(
                        item, item_roots[item.ordinal], exc
                    )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if (
        counter.current != 0
        or not 0 < counter.peak <= PROCESS_CONCURRENCY
        or set(rows) != set(range(ITEM_COUNT))
    ):
        raise NanoBEIRAvailabilityError("screen completion drifted")
    return [rows[index] for index in range(ITEM_COUNT)], counter.peak


def _git_head(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_formal(project_root: Path) -> Mapping[str, Any]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    root = base / RUN_ROOT_RELATIVE
    result_path = base / RESULT_RELATIVE
    if root.exists() or root.is_symlink():
        raise OneShotRefusal("availability root already exists")
    if result_path.exists() or result_path.is_symlink():
        raise OneShotRefusal("availability result already exists")
    _verify_preconditions(base)
    freeze = _verify_freeze(base, project_root)
    sources = load_sources(base)
    root.mkdir(mode=0o700)
    marker = {
        "candidate_freeze_self_sha256": PRECONDITIONS["candidate"]["self_sha256"],
        "implementation_freeze_self_sha256": freeze["self_sha256"],
        "schema": ATTEMPT_SCHEMA,
        "source_access_self_sha256": PRECONDITIONS["source_access"]["self_sha256"],
        "study_design_self_sha256": PRECONDITIONS["design"]["self_sha256"],
    }
    marker_path = root / "attempt.marker"
    utilities._write_json(marker_path, marker)
    patched_source = _materialize_hardened_source(base, root)

    try:
        encoder = train.bright_runtime._new_minilm(base)
        corpus_embeddings = {
            family: train.bright_runtime._encode_chunks(
                encoder, sources[family].contents
            )
            for family in FAMILIES
        }
        query_embeddings = {
            family: train.bright_runtime._encode_chunks(
                encoder, sources[family].queries
            )
            for family in FAMILIES
        }
    except train.bright_runtime.BrightStudyError as exc:
        raise NanoBEIRAvailabilityError(str(exc)) from exc
    tensor_root = root / "tensors"
    tensor_root.mkdir(mode=0o700)
    tensor_bindings: dict[str, Any] = {}
    for family in FAMILIES:
        corpus_matrix = np.asarray(corpus_embeddings[family], dtype=np.float32)
        query_matrix = np.asarray(query_embeddings[family], dtype=np.float32)
        corpus_path = tensor_root / f"{family}.corpus.npy"
        query_path = tensor_root / f"{family}.queries.npy"
        train.bright_runtime._save_npy_exclusive(corpus_path, corpus_matrix)
        train.bright_runtime._save_npy_exclusive(query_path, query_matrix)
        tensor_bindings[family] = {
            "corpus_float32_sha256": train.float32_matrix_sha256(corpus_matrix),
            "corpus_tensor_file_sha256": utilities.file_sha256(corpus_path),
            "query_float32_sha256": train.float32_matrix_sha256(query_matrix),
            "query_tensor_file_sha256": utilities.file_sha256(query_path),
        }
    items = build_screen_items(sources, corpus_embeddings, query_embeddings)
    del encoder, corpus_embeddings, query_embeddings
    _release_cuda()

    item_roots = _prepare_item_roots(root=root, items=items, sources=sources)
    rows, peak = _run_all(
        base=base,
        items=items,
        item_roots=item_roots,
        patched_source=patched_source,
    )
    private_pack = utilities.self_hashed(
        {
            "candidate_action_count": 0,
            "items": rows,
            "label_or_qrel_open_count": 0,
            "schema": PACK_SCHEMA,
        },
        field="pack_sha256",
    )
    private_root = root / "private"
    private_root.mkdir(mode=0o700)
    private_path = private_root / "availability.pack.json"
    utilities._write_json(private_path, private_pack)

    family_aggregates: dict[str, Any] = {}
    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        successes = [
            row for row in family_rows if row["availability"] == "terminal"
        ]
        failures = [row for row in family_rows if row["availability"] == "failed"]
        family_aggregates[family] = {
            "eligible_terminal_count": len(successes),
            "failure_count": len(failures),
            "failure_exception_class_counts": {
                name: sum(row.get("exception_class") == name for row in failures)
                for name in sorted(
                    {str(row.get("exception_class")) for row in failures}
                )
            },
            "minimum_graph_edge_count": min(
                (int(row["graph_edge_count"]) for row in successes), default=None
            ),
            "minimum_graph_node_count": min(
                (int(row["graph_node_count"]) for row in successes), default=None
            ),
            "query_count": len(family_rows),
        }
    passed = all(
        row["eligible_terminal_count"] >= MINIMUM_ELIGIBLE_PER_FAMILY
        for row in family_aggregates.values()
    )
    status = (
        "passed_complete_case_eligible_set_ready_for_private_HMAC_acquisition"
        if passed
        else "terminal_complete_case_capacity_failure_no_selection_action_or_score"
    )
    result = utilities.self_hashed(
        {
            "claim_boundary": {
                "candidate_action_count": 0,
                "external_network_call_count": 0,
                "label_or_qrel_open_count": 0,
                "online_evaluator_call_count": 0,
                "performance_score_count": 0,
                "selection_secret_count": 0,
            },
            "eligibility_passed": passed,
            "family_aggregates": family_aggregates,
            "formal_binding": {
                "attempt_marker_sha256": utilities.file_sha256(marker_path),
                "candidate_freeze_self_sha256": PRECONDITIONS["candidate"]["self_sha256"],
                "formal_execution_commit": _git_head(project_root),
                "implementation_freeze_self_sha256": freeze["self_sha256"],
                "source_access_self_sha256": PRECONDITIONS["source_access"]["self_sha256"],
                "study_design_self_sha256": PRECONDITIONS["design"]["self_sha256"],
            },
            "private_pack_binding": {
                "file_sha256": utilities.file_sha256(private_path),
                "pack_sha256": private_pack["pack_sha256"],
                "relative_path": private_path.relative_to(base).as_posix(),
                "size_bytes": private_path.stat().st_size,
            },
            "recorded_date": "2026-07-21",
            "runtime": {
                "HippoRAG_input_count": ITEM_COUNT,
                "HippoRAG_peak_process_concurrency": peak,
                "HippoRAG_single_launch_per_query": True,
                "qualified_HippoRAG_source_sha256": backport.PATCHED_SOURCE_SHA256,
                "tensor_bindings": tensor_bindings,
            },
            "schema": SCHEMA,
            "status": status,
        }
    )
    utilities._write_exclusive(
        result_path, utilities.canonical_json_bytes(result), mode=0o644
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--formal", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.formal:
        raise SystemExit("--formal is required")
    result = run_formal(arguments.project_root)
    print(
        json.dumps(
            {
                "eligibility_passed": result["eligibility_passed"],
                "self_sha256": result["self_sha256"],
                "status": result["status"],
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
