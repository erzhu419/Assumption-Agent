"""Public build-once/reopen adapter for the 609-article official HippoRAG index."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from replication_runtime.musique_official_hipporag_v1.contract import (
    MuSiQueOfficialHippoRAGError,
)
from replication_runtime.musique_official_hipporag_v1.runtime_attestation_v3 import (
    verify_formal_runtime_attestation_v3,
)

from .contract import (
    CORPUS_INPUT_SCHEMA,
    CORPUS_SIZE,
    IndexTreeSnapshot,
    QUERY_INPUT_SCHEMA,
    MultiHopRAGOfficialHippoRAGError,
    RetrievalBatch,
    canonical_json_bytes,
    corpus_sha256,
    parse_retrieval_output,
    serialize_corpus,
    snapshot_index_tree,
    validate_build_receipt,
    validate_corpus,
    validate_queries,
)


CORPUS_INPUT_FILENAME = "global_corpus.input.json"
BUILD_RECEIPT_FILENAME = "global_index.build_receipt.json"
RUNTIME_RECEIPT_FILENAME = "runtime.attestation_receipt.json"
INDEX_DIRECTORY_NAME = "official_global_index"
QUERY_LOCK_FILENAME = ".retrieve.lock"


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _assert_no_symlink_components(path: Path, field: str) -> None:
    absolute = path.absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise MultiHopRAGOfficialHippoRAGError(
                f"{field} contains a symlink component"
            )


def _write_exclusive(path: Path, payload: object, *, mode: int = 0o600) -> None:
    raw = canonical_json_bytes(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _load_canonical_object(path: Path, field: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise MultiHopRAGOfficialHippoRAGError(f"{field} is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MultiHopRAGOfficialHippoRAGError(f"{field} is invalid") from exc
    if raw != canonical_json_bytes(value) or not isinstance(value, dict):
        raise MultiHopRAGOfficialHippoRAGError(f"{field} is not canonical JSON")
    return value


def _validated_runtime(
    *,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    base_binding_receipt_path: Path,
    attestation_receipt_path: Path,
) -> tuple[Path, Path, Path, Path, Path, dict[str, Any]]:
    runtime_python = runtime_python.absolute()
    local_llm_model = local_llm_model.resolve(strict=True)
    local_embedding_model = local_embedding_model.resolve(strict=True)
    base_binding_receipt_path = base_binding_receipt_path.absolute()
    attestation_receipt_path = attestation_receipt_path.absolute()
    _assert_no_symlink_components(base_binding_receipt_path, "base binding receipt")
    _assert_no_symlink_components(attestation_receipt_path, "attestation receipt")
    project_root = base_binding_receipt_path.parent.parent
    try:
        safe_receipt = verify_formal_runtime_attestation_v3(
            project_root=project_root,
            attestation_receipt_path=attestation_receipt_path,
            base_binding_receipt_path=base_binding_receipt_path,
            runtime_python=runtime_python,
            local_llm_model=local_llm_model,
            local_embedding_model=local_embedding_model,
        )
    except MuSiQueOfficialHippoRAGError as exc:
        raise MultiHopRAGOfficialHippoRAGError(
            "inherited official HippoRAG runtime attestation failed"
        ) from exc
    return (
        project_root,
        runtime_python,
        local_llm_model,
        local_embedding_model,
        base_binding_receipt_path,
        safe_receipt,
    )


def _validate_timeout(timeout_seconds: int) -> int:
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
        raise MultiHopRAGOfficialHippoRAGError("timeout must be an integer")
    if not 1 <= timeout_seconds <= 14_400:
        raise MultiHopRAGOfficialHippoRAGError("timeout is outside the frozen bound")
    return timeout_seconds


def _launch_worker(
    *,
    stage: str,
    project_root: Path,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    corpus_input: Path,
    output_path: Path,
    index_root: Path,
    stage_root: Path,
    writable_root: Path,
    timeout_seconds: int,
    runtime_attestation_receipt_sha256: str,
    query_input: Path | None = None,
    build_receipt: Path | None = None,
) -> None:
    if stage not in {"build", "retrieve"}:
        raise MultiHopRAGOfficialHippoRAGError("worker stage is invalid")
    bwrap = Path("/usr/bin/bwrap")
    if not bwrap.is_file():
        raise MultiHopRAGOfficialHippoRAGError(
            "network-isolating runtime is unavailable"
        )
    environment = {
        "PATH": f"{runtime_python.parent}:/usr/bin:/bin",
        "HOME": str(writable_root / "home"),
        "HF_HOME": str(writable_root / "cache"),
        "TMPDIR": str(writable_root / "tmp"),
        "TMP": str(writable_root / "tmp"),
        "TEMP": str(writable_root / "tmp"),
        "PYTHONPATH": str(project_root),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(writable_root / "pycache"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "TOKENIZERS_PARALLELISM": "false",
    }
    command = [
        str(bwrap),
        "--unshare-net",
        "--die-with-parent",
        "--new-session",
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
    ]
    command.extend(
        [
            "--bind" if stage == "build" else "--ro-bind",
            str(stage_root),
            str(stage_root),
        ]
    )
    if writable_root != stage_root:
        command.extend(["--bind", str(writable_root), str(writable_root)])
    command.extend(
        [
            str(runtime_python),
            "-B",
            "-m",
            "replication_runtime.multihoprag_official_hipporag_v1.worker",
            "--stage",
            stage,
            "--corpus-input",
            str(corpus_input),
            "--output",
            str(output_path),
            "--index-root",
            str(index_root),
            "--llm-model",
            str(local_llm_model),
            "--embedding-model",
            str(local_embedding_model),
            "--runtime-attestation-receipt-sha256",
            runtime_attestation_receipt_sha256,
        ]
    )
    if stage == "retrieve":
        if query_input is None or build_receipt is None:
            raise MultiHopRAGOfficialHippoRAGError(
                "retrieve worker inputs are incomplete"
            )
        command.extend(
            [
                "--query-input",
                str(query_input),
                "--build-receipt",
                str(build_receipt),
            ]
        )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        env=environment,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise MultiHopRAGOfficialHippoRAGError(
            "official global HippoRAG worker failed; "
            f"returncode={completed.returncode}; "
            f"stdout_sha256={_sha256_bytes(completed.stdout)}; "
            f"stderr_sha256={_sha256_bytes(completed.stderr)}"
        )
    try:
        status = json.loads(completed.stdout.decode("utf-8").strip().splitlines()[-1])
    except (IndexError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MultiHopRAGOfficialHippoRAGError(
            "official worker emitted no safe terminal receipt"
        ) from exc
    if not isinstance(status, dict) or status.get("stage") != stage or status.get(
        "status"
    ) != "passed":
        raise MultiHopRAGOfficialHippoRAGError(
            "official worker terminal receipt drifted"
        )
    if stage == "build" and status != {
        "corpus_count": CORPUS_SIZE,
        "index_call_count": 1,
        "stage": "build",
        "status": "passed",
    }:
        raise MultiHopRAGOfficialHippoRAGError("build terminal receipt drifted")
    if stage == "retrieve" and (
        set(status)
        != {"batch_count", "index_call_count", "query_count", "stage", "status"}
        or status.get("index_call_count") != 0
    ):
        raise MultiHopRAGOfficialHippoRAGError("retrieve terminal receipt drifted")


def build_official_hipporag_global_index_v1(
    *,
    articles: Sequence[Mapping[str, object]],
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    base_binding_receipt_path: Path,
    attestation_receipt_path: Path,
    stage_root: Path,
    timeout_seconds: int = 7200,
) -> dict[str, Any]:
    """Build and retain one official index for the complete 609-article corpus."""

    validated_articles = validate_corpus(articles)
    documents = serialize_corpus(validated_articles)
    timeout_seconds = _validate_timeout(timeout_seconds)
    (
        project_root,
        runtime_python,
        local_llm_model,
        local_embedding_model,
        _base_receipt,
        runtime_receipt,
    ) = _validated_runtime(
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    stage_root = stage_root.absolute()
    _assert_no_symlink_components(stage_root.parent, "stage root parent")
    if stage_root.exists():
        raise MultiHopRAGOfficialHippoRAGError(
            "persistent global index stage must not already exist"
        )
    stage_root.mkdir(mode=0o700)
    try:
        for name in ("home", "cache", "tmp", "pycache"):
            (stage_root / name).mkdir(mode=0o700)
        corpus_input = stage_root / CORPUS_INPUT_FILENAME
        build_receipt_path = stage_root / BUILD_RECEIPT_FILENAME
        runtime_receipt_path = stage_root / RUNTIME_RECEIPT_FILENAME
        index_root = stage_root / INDEX_DIRECTORY_NAME
        _write_exclusive(
            corpus_input,
            {
                "articles": [
                    {"body": row.body, "idx": row.idx, "title": row.title}
                    for row in validated_articles
                ],
                "schema": CORPUS_INPUT_SCHEMA,
            },
        )
        _write_exclusive(runtime_receipt_path, runtime_receipt)
        runtime_receipt_file_sha256 = _sha256_bytes(
            runtime_receipt_path.read_bytes()
        )
        _launch_worker(
            stage="build",
            project_root=project_root,
            runtime_python=runtime_python,
            local_llm_model=local_llm_model,
            local_embedding_model=local_embedding_model,
            corpus_input=corpus_input,
            output_path=build_receipt_path,
            index_root=index_root,
            stage_root=stage_root,
            writable_root=stage_root,
            timeout_seconds=timeout_seconds,
            runtime_attestation_receipt_sha256=runtime_receipt_file_sha256,
        )
        payload = _load_canonical_object(build_receipt_path, "build receipt")
        index_snapshot = snapshot_index_tree(index_root)
        return validate_build_receipt(
            payload,
            expected_corpus_sha256=corpus_sha256(documents),
            expected_index_snapshot=index_snapshot,
            expected_runtime_attestation_receipt_sha256=(
                runtime_receipt_file_sha256
            ),
        )
    except BaseException:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


def retrieve_official_hipporag_global_index_v1(
    *,
    queries: Sequence[str],
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    base_binding_receipt_path: Path,
    attestation_receipt_path: Path,
    stage_root: Path,
    work_root: Path,
    timeout_seconds: int = 3600,
) -> RetrievalBatch:
    """Reopen the persistent index and retrieve query batches without reindexing."""

    validated_queries = validate_queries(queries)
    timeout_seconds = _validate_timeout(timeout_seconds)
    (
        project_root,
        runtime_python,
        local_llm_model,
        local_embedding_model,
        _base_receipt,
        runtime_receipt,
    ) = _validated_runtime(
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
        base_binding_receipt_path=base_binding_receipt_path,
        attestation_receipt_path=attestation_receipt_path,
    )
    stage_root = stage_root.absolute()
    work_root = work_root.absolute()
    _assert_no_symlink_components(stage_root, "persistent index stage")
    _assert_no_symlink_components(work_root.parent, "query work root parent")
    if not stage_root.is_dir():
        raise MultiHopRAGOfficialHippoRAGError(
            "persistent global index stage is unavailable"
        )
    if work_root.exists():
        raise MultiHopRAGOfficialHippoRAGError("query work root must not already exist")
    if stage_root == work_root or stage_root in work_root.parents or work_root in stage_root.parents:
        raise MultiHopRAGOfficialHippoRAGError(
            "query work root must be disjoint from the persistent index stage"
        )

    corpus_input = stage_root / CORPUS_INPUT_FILENAME
    build_receipt_path = stage_root / BUILD_RECEIPT_FILENAME
    runtime_receipt_path = stage_root / RUNTIME_RECEIPT_FILENAME
    index_root = stage_root / INDEX_DIRECTORY_NAME
    persisted_runtime = _load_canonical_object(runtime_receipt_path, "runtime receipt")
    if persisted_runtime != runtime_receipt:
        raise MultiHopRAGOfficialHippoRAGError(
            "reopen runtime differs from the build-stage attestation"
        )
    corpus_payload = _load_canonical_object(corpus_input, "corpus input")
    if set(corpus_payload) != {"articles", "schema"} or corpus_payload.get(
        "schema"
    ) != CORPUS_INPUT_SCHEMA or not isinstance(corpus_payload.get("articles"), list):
        raise MultiHopRAGOfficialHippoRAGError("persisted corpus envelope drifted")
    articles = validate_corpus(corpus_payload["articles"])
    documents = serialize_corpus(articles)
    runtime_receipt_file_sha256 = _sha256_bytes(runtime_receipt_path.read_bytes())
    canonical_snapshot_before = snapshot_index_tree(index_root)
    build_receipt = validate_build_receipt(
        _load_canonical_object(build_receipt_path, "build receipt"),
        expected_corpus_sha256=corpus_sha256(documents),
        expected_index_snapshot=canonical_snapshot_before,
        expected_runtime_attestation_receipt_sha256=(
            runtime_receipt_file_sha256
        ),
    )
    if index_root.is_symlink() or not index_root.is_dir():
        raise MultiHopRAGOfficialHippoRAGError(
            "persistent official global index is unavailable"
        )

    lock_path = stage_root / QUERY_LOCK_FILENAME
    try:
        lock_descriptor = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise MultiHopRAGOfficialHippoRAGError(
            "a global-index retrieval worker is already active"
        ) from exc
    os.close(lock_descriptor)
    try:
        work_root.mkdir(mode=0o700)
        for name in ("home", "cache", "tmp", "pycache"):
            (work_root / name).mkdir(mode=0o700)
        working_index_root = work_root / "official_global_index.read_clone"
        shutil.copytree(index_root, working_index_root, copy_function=shutil.copy2)
        working_snapshot = snapshot_index_tree(working_index_root)
        if working_snapshot != canonical_snapshot_before:
            raise MultiHopRAGOfficialHippoRAGError(
                "query index clone differs from the persisted build index"
            )
        query_input = work_root / "queries.input.json"
        output_path = work_root / "retrieved_idx.receipt.json"
        _write_exclusive(
            query_input,
            {"queries": list(validated_queries), "schema": QUERY_INPUT_SCHEMA},
        )
        _launch_worker(
            stage="retrieve",
            project_root=project_root,
            runtime_python=runtime_python,
            local_llm_model=local_llm_model,
            local_embedding_model=local_embedding_model,
            corpus_input=corpus_input,
            query_input=query_input,
            build_receipt=build_receipt_path,
            output_path=output_path,
            index_root=working_index_root,
            stage_root=stage_root,
            writable_root=work_root,
            timeout_seconds=timeout_seconds,
            runtime_attestation_receipt_sha256=(
                runtime_receipt_file_sha256
            ),
        )
        if output_path.is_symlink() or not output_path.is_file():
            raise MultiHopRAGOfficialHippoRAGError(
                "retrieval idx/receipt output is unavailable"
            )
        working_snapshot_after = snapshot_index_tree(working_index_root)
        canonical_snapshot_after = snapshot_index_tree(index_root)
        if canonical_snapshot_after != canonical_snapshot_before:
            raise MultiHopRAGOfficialHippoRAGError(
                "persisted build index changed during retrieve"
            )
        result = parse_retrieval_output(
            output_path.read_bytes(),
            queries=validated_queries,
            expected_build_receipt=build_receipt,
            expected_index_snapshot_after=working_snapshot_after,
        )
        if result.receipt.get("corpus_sha256") != build_receipt["corpus_sha256"]:
            raise MultiHopRAGOfficialHippoRAGError(
                "retrieval receipt is not bound to the persistent index corpus"
            )
        if (
            result.receipt.get("index_tree_sha256")
            != canonical_snapshot_before.tree_sha256
            or result.receipt.get("runtime_attestation_receipt_sha256")
            != runtime_receipt_file_sha256
        ):
            raise MultiHopRAGOfficialHippoRAGError(
                "retrieval receipt is not bound to the persisted index/runtime"
            )
        return result
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "build_official_hipporag_global_index_v1",
    "retrieve_official_hipporag_global_index_v1",
]
