"""Build-once/reopen adapter for the exact 2,900-unit DSTC9 corpus."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

from replication_runtime.musique_official_hipporag_v1.contract import (
    MuSiQueOfficialHippoRAGError,
)
from replication_runtime.musique_official_hipporag_v1.runtime_attestation_v3 import (
    verify_formal_runtime_attestation_v3,
)

from .contract import (
    CORPUS_SIZE,
    CUDA_VISIBLE_DEVICES,
    MAX_QUERY_COUNT,
    SYSTEMD_NETWORK_PROPERTIES,
    WORKER_ENVIRONMENT_KEYS,
    WORKER_FIXED_ENVIRONMENT_VALUES,
    Dstc9OfficialHippoRAGError,
    RetrievalBatch,
    canonical_json_bytes,
    corpus_input_projection,
    parse_retrieval_output,
    query_input_projection,
    serialize_corpus,
    snapshot_index_tree,
    validate_build_receipt,
    validate_corpus_input,
    validate_query_input,
)


CORPUS_INPUT_FILENAME = "global_corpus.input.json"
BUILD_RECEIPT_FILENAME = "global_index.build_receipt.json"
RUNTIME_RECEIPT_FILENAME = "runtime.attestation_receipt.json"
INDEX_DIRECTORY_NAME = "official_global_index"
QUERY_LOCK_FILENAME = ".retrieve.lock"
SYSTEMD_RUN = Path("/usr/bin/systemd-run")
ENV_EXECUTABLE = Path("/usr/bin/env")
SYSTEMD_RUN_FLAGS = ("--user", "--wait", "--pipe", "--collect", "--quiet")
SYSTEMD_PREFLIGHT_TIMEOUT_SECONDS = 30
SYSTEMD_PREFLIGHT_SCRIPT = (
    "import os,socket\n"
    "if set(os.environ)!={'LANG'} or os.environ.get('LANG')!='C.UTF-8':"
    " raise SystemExit(40)\n"
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
            raise Dstc9OfficialHippoRAGError(
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
        raise Dstc9OfficialHippoRAGError(f"{field} is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Dstc9OfficialHippoRAGError(f"{field} is invalid") from exc
    if raw != canonical_json_bytes(value) or not isinstance(value, dict):
        raise Dstc9OfficialHippoRAGError(f"{field} is not canonical JSON")
    return value


def _validated_runtime(
    *,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    base_binding_receipt_path: Path,
    attestation_receipt_path: Path,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
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
        raise Dstc9OfficialHippoRAGError(
            "inherited official HippoRAG runtime attestation failed"
        ) from exc
    return (
        project_root,
        runtime_python,
        local_llm_model,
        local_embedding_model,
        safe_receipt,
    )


def _validate_timeout(timeout_seconds: int) -> int:
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 14_400:
        raise Dstc9OfficialHippoRAGError(
            "timeout is outside the frozen integer bound"
        )
    return timeout_seconds


def _launcher_environment() -> dict[str, str]:
    """Return only variables needed to contact the user systemd manager."""

    environment = {
        "HOME": os.environ.get("HOME", "/"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PATH": "/usr/bin:/bin",
    }
    for key in ("DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR"):
        value = os.environ.get(key)
        if value:
            environment[key] = value
    return environment


def _systemd_command_prefix() -> list[str]:
    if not SYSTEMD_RUN.is_file():
        raise Dstc9OfficialHippoRAGError(
            "systemd network-isolating runtime is unavailable"
        )
    command = [str(SYSTEMD_RUN), *SYSTEMD_RUN_FLAGS]
    for property_value in SYSTEMD_NETWORK_PROPERTIES:
        command.extend(("--property", property_value))
    command.append("--")
    return command


def _clean_environment_exec_prefix(
    environment: Mapping[str, str],
) -> list[str]:
    if not ENV_EXECUTABLE.is_file():
        raise Dstc9OfficialHippoRAGError(
            "environment-clearing runtime is unavailable"
        )
    if any(
        not isinstance(key, str)
        or not key
        or "=" in key
        or "\x00" in key
        or not isinstance(value, str)
        or "\x00" in value
        or "\n" in value
        for key, value in environment.items()
    ):
        raise Dstc9OfficialHippoRAGError(
            "systemd child environment is malformed"
        )
    return [
        str(ENV_EXECUTABLE),
        "--ignore-environment",
        "--",
        *(f"{key}={environment[key]}" for key in sorted(environment)),
    ]


def _preflight_systemd_transport() -> None:
    command = (
        _systemd_command_prefix()
        + _clean_environment_exec_prefix({"LANG": "C.UTF-8"})
        + ["/usr/bin/python3", "-I", "-c", SYSTEMD_PREFLIGHT_SCRIPT]
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=_launcher_environment(),
            timeout=SYSTEMD_PREFLIGHT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Dstc9OfficialHippoRAGError(
            "systemd network-isolation capability preflight failed"
        ) from exc
    if completed.returncode != 0:
        raise Dstc9OfficialHippoRAGError(
            "systemd network-isolation capability preflight failed; "
            f"returncode={completed.returncode}; "
            f"stdout_sha256={_sha256_bytes(completed.stdout)}; "
            f"stderr_sha256={_sha256_bytes(completed.stderr)}"
        )


def _worker_environment(
    *,
    runtime_python: Path,
    project_root: Path,
    writable_root: Path,
) -> dict[str, str]:
    environment = {
        "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
        "HOME": str(writable_root / "home"),
        "HF_HOME": str(writable_root / "cache"),
        "HF_HUB_OFFLINE": "1",
        "LANG": "C.UTF-8",
        "PATH": f"{runtime_python.parent}:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(project_root),
        "TEMP": str(writable_root / "tmp"),
        "TMP": str(writable_root / "tmp"),
        "TMPDIR": str(writable_root / "tmp"),
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if frozenset(environment) != WORKER_ENVIRONMENT_KEYS or any(
        environment.get(key) != value
        for key, value in WORKER_FIXED_ENVIRONMENT_VALUES.items()
    ):
        raise Dstc9OfficialHippoRAGError("worker environment contract drifted")
    return environment


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
    writable_root: Path,
    timeout_seconds: int,
    runtime_attestation_receipt_sha256: str,
    expected_corpus_count: int,
    expected_query_count: int | None = None,
    query_input: Path | None = None,
    build_receipt: Path | None = None,
) -> None:
    if stage not in {"build", "retrieve"}:
        raise Dstc9OfficialHippoRAGError("worker stage is invalid")
    if expected_corpus_count != CORPUS_SIZE:
        raise Dstc9OfficialHippoRAGError(
            "worker corpus count must remain exactly 2900"
        )
    _preflight_systemd_transport()
    child_environment = _worker_environment(
        runtime_python=runtime_python,
        project_root=project_root,
        writable_root=writable_root,
    )
    command = _systemd_command_prefix()
    command.extend(_clean_environment_exec_prefix(child_environment))
    command.extend(
        [
            str(runtime_python),
            "-B",
            "-m",
            "replication_runtime.dstc9_official_hipporag_v1.worker",
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
            or type(expected_query_count) is not int
            or not 1 <= expected_query_count <= MAX_QUERY_COUNT
        ):
            raise Dstc9OfficialHippoRAGError(
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
        raise Dstc9OfficialHippoRAGError(
            "official global HippoRAG worker failed; "
            f"returncode={completed.returncode}; "
            f"stdout_sha256={_sha256_bytes(completed.stdout)}; "
            f"stderr_sha256={_sha256_bytes(completed.stderr)}"
        )
    try:
        status = json.loads(
            completed.stdout.decode("utf-8").strip().splitlines()[-1]
        )
    except (IndexError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Dstc9OfficialHippoRAGError(
            "official worker emitted no safe terminal receipt"
        ) from exc
    if stage == "build":
        expected = {
            "corpus_count": CORPUS_SIZE,
            "index_call_count": 1,
            "stage": "build",
            "status": "passed",
        }
    else:
        expected = {
            "batch_count": (expected_query_count + 7) // 8,  # type: ignore[operator]
            "index_call_count": 0,
            "query_count": expected_query_count,
            "stage": "retrieve",
            "status": "passed",
        }
    if status != expected:
        raise Dstc9OfficialHippoRAGError(
            f"{stage} terminal receipt drifted"
        )


def build_dstc9_official_hipporag_global_index_v1(
    *,
    corpus_input: object,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    base_binding_receipt_path: Path,
    attestation_receipt_path: Path,
    stage_root: Path,
    timeout_seconds: int = 7200,
) -> dict[str, Any]:
    """Build one persistent index from a self-hashed source-free envelope."""

    corpus = validate_corpus_input(corpus_input)
    timeout_seconds = _validate_timeout(timeout_seconds)
    (
        project_root,
        runtime_python,
        local_llm_model,
        local_embedding_model,
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
        raise Dstc9OfficialHippoRAGError(
            "persistent global index stage must not already exist"
        )
    stage_root.mkdir(mode=0o700)
    try:
        for name in ("home", "cache", "tmp"):
            (stage_root / name).mkdir(mode=0o700)
        corpus_path = stage_root / CORPUS_INPUT_FILENAME
        build_receipt_path = stage_root / BUILD_RECEIPT_FILENAME
        runtime_receipt_path = stage_root / RUNTIME_RECEIPT_FILENAME
        index_root = stage_root / INDEX_DIRECTORY_NAME
        _write_exclusive(corpus_path, corpus_input_projection(corpus))
        _write_exclusive(runtime_receipt_path, runtime_receipt)
        runtime_file_hash = _sha256_bytes(runtime_receipt_path.read_bytes())
        _launch_worker(
            stage="build",
            project_root=project_root,
            runtime_python=runtime_python,
            local_llm_model=local_llm_model,
            local_embedding_model=local_embedding_model,
            corpus_input=corpus_path,
            output_path=build_receipt_path,
            index_root=index_root,
            writable_root=stage_root,
            timeout_seconds=timeout_seconds,
            runtime_attestation_receipt_sha256=runtime_file_hash,
            expected_corpus_count=CORPUS_SIZE,
        )
        return validate_build_receipt(
            _load_canonical_object(build_receipt_path, "build receipt"),
            expected_corpus_input=corpus,
            expected_index_snapshot=snapshot_index_tree(index_root),
            expected_runtime_attestation_receipt_sha256=runtime_file_hash,
        )
    except BaseException:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


def retrieve_dstc9_official_hipporag_global_index_v1(
    *,
    query_input: object,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    base_binding_receipt_path: Path,
    attestation_receipt_path: Path,
    stage_root: Path,
    work_root: Path,
    timeout_seconds: int = 3600,
) -> RetrievalBatch:
    """Reopen the build index and retrieve once without indexing or retry."""

    timeout_seconds = _validate_timeout(timeout_seconds)
    (
        project_root,
        runtime_python,
        local_llm_model,
        local_embedding_model,
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
        raise Dstc9OfficialHippoRAGError(
            "persistent global index stage is unavailable"
        )
    if work_root.exists():
        raise Dstc9OfficialHippoRAGError(
            "query work root must not already exist"
        )
    if (
        stage_root == work_root
        or stage_root in work_root.parents
        or work_root in stage_root.parents
    ):
        raise Dstc9OfficialHippoRAGError(
            "query work root must be disjoint from the persistent index"
        )

    corpus_path = stage_root / CORPUS_INPUT_FILENAME
    build_receipt_path = stage_root / BUILD_RECEIPT_FILENAME
    runtime_receipt_path = stage_root / RUNTIME_RECEIPT_FILENAME
    index_root = stage_root / INDEX_DIRECTORY_NAME
    persisted_runtime = _load_canonical_object(
        runtime_receipt_path, "runtime receipt"
    )
    if persisted_runtime != runtime_receipt:
        raise Dstc9OfficialHippoRAGError(
            "reopen runtime differs from build attestation"
        )
    corpus = validate_corpus_input(
        _load_canonical_object(corpus_path, "corpus input")
    )
    queries = validate_query_input(
        query_input, expected_study_id=corpus.study_id
    )
    runtime_file_hash = _sha256_bytes(runtime_receipt_path.read_bytes())
    canonical_before = snapshot_index_tree(index_root)
    build_receipt = validate_build_receipt(
        _load_canonical_object(build_receipt_path, "build receipt"),
        expected_corpus_input=corpus,
        expected_index_snapshot=canonical_before,
        expected_runtime_attestation_receipt_sha256=runtime_file_hash,
    )

    lock_path = stage_root / QUERY_LOCK_FILENAME
    try:
        lock_descriptor = os.open(
            lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError as exc:
        raise Dstc9OfficialHippoRAGError(
            "a global-index retrieval worker is already active"
        ) from exc
    os.close(lock_descriptor)
    try:
        work_root.mkdir(mode=0o700)
        for name in ("home", "cache", "tmp"):
            (work_root / name).mkdir(mode=0o700)
        working_index = work_root / "official_global_index.read_clone"
        shutil.copytree(index_root, working_index, copy_function=shutil.copy2)
        if snapshot_index_tree(working_index) != canonical_before:
            raise Dstc9OfficialHippoRAGError(
                "query index clone differs from persisted build index"
            )
        query_path = work_root / "queries.input.json"
        output_path = work_root / "retrieved_ordinals.receipt.json"
        _write_exclusive(query_path, query_input_projection(queries))
        _launch_worker(
            stage="retrieve",
            project_root=project_root,
            runtime_python=runtime_python,
            local_llm_model=local_llm_model,
            local_embedding_model=local_embedding_model,
            corpus_input=corpus_path,
            query_input=query_path,
            build_receipt=build_receipt_path,
            output_path=output_path,
            index_root=working_index,
            writable_root=work_root,
            timeout_seconds=timeout_seconds,
            runtime_attestation_receipt_sha256=runtime_file_hash,
            expected_corpus_count=CORPUS_SIZE,
            expected_query_count=len(queries.queries),
        )
        if output_path.is_symlink() or not output_path.is_file():
            raise Dstc9OfficialHippoRAGError(
                "retrieval ordinal/receipt output is unavailable"
            )
        working_after = snapshot_index_tree(working_index)
        if snapshot_index_tree(index_root) != canonical_before:
            raise Dstc9OfficialHippoRAGError(
                "persisted build index changed during retrieve"
            )
        result = parse_retrieval_output(
            output_path.read_bytes(),
            query_input=queries,
            expected_build_receipt=build_receipt,
            expected_index_snapshot_after=working_after,
        )
        if (
            result.receipt.get("corpus_sha256")
            != build_receipt["corpus_sha256"]
            or result.receipt.get("corpus_input_self_sha256")
            != build_receipt["corpus_input_self_sha256"]
            or result.receipt.get("index_tree_sha256")
            != canonical_before.tree_sha256
            or result.receipt.get("runtime_attestation_receipt_sha256")
            != runtime_file_hash
        ):
            raise Dstc9OfficialHippoRAGError(
                "retrieval receipt is not bound to build/runtime"
            )
        return result
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "build_dstc9_official_hipporag_global_index_v1",
    "retrieve_dstc9_official_hipporag_global_index_v1",
]
