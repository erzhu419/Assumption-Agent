"""Public build-once/reopen adapter for the exact FEVEROUS 8,192-unit index."""

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
    CORPUS_SIZE,
    CORPUS_INPUT_SCHEMA,
    IndexTreeSnapshot,
    MAX_QUERY_COUNT,
    QUERY_INPUT_SCHEMA,
    FeverousOfficialHippoRAGError,
    RetrievalBatch,
    SYSTEMD_NETWORK_PROPERTIES,
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
SYSTEMD_RUN = Path("/usr/bin/systemd-run")
SYSTEMD_RUN_FLAGS = ("--user", "--wait", "--pipe", "--collect", "--quiet")
SYSTEMD_PREFLIGHT_TIMEOUT_SECONDS = 30
SYSTEMD_PREFLIGHT_SCRIPT = (
    "import socket\n"
    "probe=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);probe.close()\n"
    "for family in (socket.AF_INET,socket.AF_INET6):\n"
    " try: probe=socket.socket(family,socket.SOCK_STREAM)\n"
    " except OSError: continue\n"
    " probe.close();raise SystemExit(41)\n"
)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _assert_no_symlink_components(path: Path, field: str) -> None:
    absolute = path.absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise FeverousOfficialHippoRAGError(
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
        raise FeverousOfficialHippoRAGError(f"{field} is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeverousOfficialHippoRAGError(f"{field} is invalid") from exc
    if raw != canonical_json_bytes(value) or not isinstance(value, dict):
        raise FeverousOfficialHippoRAGError(f"{field} is not canonical JSON")
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
        raise FeverousOfficialHippoRAGError(
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
        raise FeverousOfficialHippoRAGError("timeout must be an integer")
    if not 1 <= timeout_seconds <= 14_400:
        raise FeverousOfficialHippoRAGError("timeout is outside the frozen bound")
    return timeout_seconds


def _launcher_environment() -> dict[str, str]:
    """Return only the variables needed to reach the user systemd manager."""

    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": os.environ.get("HOME", "/"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }
    for key in ("DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR"):
        value = os.environ.get(key)
        if value:
            environment[key] = value
    return environment


def _systemd_command_prefix(child_environment: Mapping[str, str]) -> list[str]:
    """Freeze the GPU-preserving, transport-denying transient-unit prefix."""

    if not SYSTEMD_RUN.is_file():
        raise FeverousOfficialHippoRAGError(
            "systemd network-isolating runtime is unavailable"
        )
    if any(
        not isinstance(key, str)
        or not key
        or "=" in key
        or "\x00" in key
        or not isinstance(value, str)
        or "\x00" in value
        or "\n" in value
        for key, value in child_environment.items()
    ):
        raise FeverousOfficialHippoRAGError(
            "systemd child environment is malformed"
        )
    command = [str(SYSTEMD_RUN), *SYSTEMD_RUN_FLAGS]
    for property_value in SYSTEMD_NETWORK_PROPERTIES:
        command.extend(("--property", property_value))
    for key in sorted(child_environment):
        command.append(f"--setenv={key}={child_environment[key]}")
    command.append("--")
    return command


def _preflight_systemd_transport() -> None:
    """Prove that the user manager accepts both frozen network properties."""

    command = _systemd_command_prefix({}) + [
        "/usr/bin/python3",
        "-I",
        "-c",
        SYSTEMD_PREFLIGHT_SCRIPT,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=_launcher_environment(),
            timeout=SYSTEMD_PREFLIGHT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FeverousOfficialHippoRAGError(
            "systemd network-isolation capability preflight failed"
        ) from exc
    if completed.returncode != 0:
        raise FeverousOfficialHippoRAGError(
            "systemd network-isolation capability preflight failed; "
            f"returncode={completed.returncode}; "
            f"stdout_sha256={_sha256_bytes(completed.stdout)}; "
            f"stderr_sha256={_sha256_bytes(completed.stderr)}"
        )


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
    expected_corpus_count: int,
    expected_query_count: int | None = None,
    query_input: Path | None = None,
    build_receipt: Path | None = None,
) -> None:
    if stage not in {"build", "retrieve"}:
        raise FeverousOfficialHippoRAGError("worker stage is invalid")
    if expected_corpus_count != CORPUS_SIZE:
        raise FeverousOfficialHippoRAGError(
            "worker corpus count must equal the frozen FEVEROUS corpus size"
        )
    _preflight_systemd_transport()
    child_environment = {
        "PATH": f"{runtime_python.parent}:/usr/bin:/bin",
        "HOME": str(writable_root / "home"),
        "HF_HOME": str(writable_root / "cache"),
        "TMPDIR": str(writable_root / "tmp"),
        "TMP": str(writable_root / "tmp"),
        "TEMP": str(writable_root / "tmp"),
        "PYTHONPATH": str(project_root),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    command = _systemd_command_prefix(child_environment)
    command.extend(
        [
            str(runtime_python),
            "-B",
            "-m",
            "replication_runtime.feverous_official_hipporag_v1.worker",
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
        if (
            query_input is None
            or build_receipt is None
            or isinstance(expected_query_count, bool)
            or not isinstance(expected_query_count, int)
            or not 1 <= expected_query_count <= MAX_QUERY_COUNT
        ):
            raise FeverousOfficialHippoRAGError(
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
        env=_launcher_environment(),
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise FeverousOfficialHippoRAGError(
            "official global HippoRAG worker failed; "
            f"returncode={completed.returncode}; "
            f"stdout_sha256={_sha256_bytes(completed.stdout)}; "
            f"stderr_sha256={_sha256_bytes(completed.stderr)}"
        )
    try:
        status = json.loads(completed.stdout.decode("utf-8").strip().splitlines()[-1])
    except (IndexError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeverousOfficialHippoRAGError(
            "official worker emitted no safe terminal receipt"
        ) from exc
    if not isinstance(status, dict) or status.get("stage") != stage or status.get(
        "status"
    ) != "passed":
        raise FeverousOfficialHippoRAGError(
            "official worker terminal receipt drifted"
        )
    if stage == "build" and status != {
        "corpus_count": expected_corpus_count,
        "index_call_count": 1,
        "stage": "build",
        "status": "passed",
    }:
        raise FeverousOfficialHippoRAGError("build terminal receipt drifted")
    if stage == "retrieve" and status != {
        "batch_count": (expected_query_count + 7) // 8,  # type: ignore[operator]
        "index_call_count": 0,
        "query_count": expected_query_count,
        "stage": "retrieve",
        "status": "passed",
    }:
        raise FeverousOfficialHippoRAGError("retrieve terminal receipt drifted")


def build_feverous_official_hipporag_global_index_v1(
    *,
    units: Sequence[Mapping[str, object]],
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    base_binding_receipt_path: Path,
    attestation_receipt_path: Path,
    stage_root: Path,
    timeout_seconds: int = 7200,
) -> dict[str, Any]:
    """Build and retain one official index for the complete cohort corpus."""

    validated_units = validate_corpus(units)
    documents = serialize_corpus(validated_units)
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
        raise FeverousOfficialHippoRAGError(
            "persistent global index stage must not already exist"
        )
    stage_root.mkdir(mode=0o700)
    try:
        for name in ("home", "cache", "tmp"):
            (stage_root / name).mkdir(mode=0o700)
        corpus_input = stage_root / CORPUS_INPUT_FILENAME
        build_receipt_path = stage_root / BUILD_RECEIPT_FILENAME
        runtime_receipt_path = stage_root / RUNTIME_RECEIPT_FILENAME
        index_root = stage_root / INDEX_DIRECTORY_NAME
        _write_exclusive(
            corpus_input,
            {
                "units": [
                    {"idx": row.idx, "text": row.text}
                    for row in validated_units
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
            expected_corpus_count=len(validated_units),
        )
        payload = _load_canonical_object(build_receipt_path, "build receipt")
        index_snapshot = snapshot_index_tree(index_root)
        return validate_build_receipt(
            payload,
            expected_documents=documents,
            expected_index_snapshot=index_snapshot,
            expected_runtime_attestation_receipt_sha256=(
                runtime_receipt_file_sha256
            ),
        )
    except BaseException:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


def retrieve_feverous_official_hipporag_global_index_v1(
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
        raise FeverousOfficialHippoRAGError(
            "persistent global index stage is unavailable"
        )
    if work_root.exists():
        raise FeverousOfficialHippoRAGError("query work root must not already exist")
    if (
        stage_root == work_root
        or stage_root in work_root.parents
        or work_root in stage_root.parents
    ):
        raise FeverousOfficialHippoRAGError(
            "query work root must be disjoint from the persistent index stage"
        )

    corpus_input = stage_root / CORPUS_INPUT_FILENAME
    build_receipt_path = stage_root / BUILD_RECEIPT_FILENAME
    runtime_receipt_path = stage_root / RUNTIME_RECEIPT_FILENAME
    index_root = stage_root / INDEX_DIRECTORY_NAME
    persisted_runtime = _load_canonical_object(runtime_receipt_path, "runtime receipt")
    if persisted_runtime != runtime_receipt:
        raise FeverousOfficialHippoRAGError(
            "reopen runtime differs from the build-stage attestation"
        )
    corpus_payload = _load_canonical_object(corpus_input, "corpus input")
    if set(corpus_payload) != {"units", "schema"} or corpus_payload.get(
        "schema"
    ) != CORPUS_INPUT_SCHEMA or not isinstance(corpus_payload.get("units"), list):
        raise FeverousOfficialHippoRAGError("persisted corpus envelope drifted")
    units = validate_corpus(corpus_payload["units"])
    documents = serialize_corpus(units)
    runtime_receipt_file_sha256 = _sha256_bytes(runtime_receipt_path.read_bytes())
    canonical_snapshot_before = snapshot_index_tree(index_root)
    build_receipt = validate_build_receipt(
        _load_canonical_object(build_receipt_path, "build receipt"),
        expected_documents=documents,
        expected_index_snapshot=canonical_snapshot_before,
        expected_runtime_attestation_receipt_sha256=(
            runtime_receipt_file_sha256
        ),
    )
    if index_root.is_symlink() or not index_root.is_dir():
        raise FeverousOfficialHippoRAGError(
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
        raise FeverousOfficialHippoRAGError(
            "a global-index retrieval worker is already active"
        ) from exc
    os.close(lock_descriptor)
    try:
        work_root.mkdir(mode=0o700)
        for name in ("home", "cache", "tmp"):
            (work_root / name).mkdir(mode=0o700)
        working_index_root = work_root / "official_global_index.read_clone"
        shutil.copytree(index_root, working_index_root, copy_function=shutil.copy2)
        working_snapshot = snapshot_index_tree(working_index_root)
        if working_snapshot != canonical_snapshot_before:
            raise FeverousOfficialHippoRAGError(
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
            expected_corpus_count=len(documents),
            expected_query_count=len(validated_queries),
        )
        if output_path.is_symlink() or not output_path.is_file():
            raise FeverousOfficialHippoRAGError(
                "retrieval idx/receipt output is unavailable"
            )
        working_snapshot_after = snapshot_index_tree(working_index_root)
        canonical_snapshot_after = snapshot_index_tree(index_root)
        if canonical_snapshot_after != canonical_snapshot_before:
            raise FeverousOfficialHippoRAGError(
                "persisted build index changed during retrieve"
            )
        result = parse_retrieval_output(
            output_path.read_bytes(),
            queries=validated_queries,
            expected_build_receipt=build_receipt,
            expected_index_snapshot_after=working_snapshot_after,
        )
        if result.receipt.get("corpus_sha256") != build_receipt["corpus_sha256"]:
            raise FeverousOfficialHippoRAGError(
                "retrieval receipt is not bound to the persistent index corpus"
            )
        if (
            result.receipt.get("index_tree_sha256")
            != canonical_snapshot_before.tree_sha256
            or result.receipt.get("runtime_attestation_receipt_sha256")
            != runtime_receipt_file_sha256
        ):
            raise FeverousOfficialHippoRAGError(
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
    "build_feverous_official_hipporag_global_index_v1",
    "retrieve_feverous_official_hipporag_global_index_v1",
]
