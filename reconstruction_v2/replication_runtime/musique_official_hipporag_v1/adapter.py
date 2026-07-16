"""Public fail-closed launcher for one-item official HippoRAG retrieval."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Mapping, Sequence

from .binding import verify_live_binding
from .contract import (
    INPUT_SCHEMA,
    MuSiQueOfficialHippoRAGError,
    parse_idx_only_output,
    validate_single_item,
)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _assert_no_symlink_components(path: Path, field: str) -> None:
    absolute = path.absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise MuSiQueOfficialHippoRAGError(f"{field} contains a symlink component")


def _write_private_input(
    path: Path, *, question: str, paragraphs: Sequence[Mapping[str, object]]
) -> None:
    payload = {
        "schema": INPUT_SCHEMA,
        "question": question,
        "paragraphs": [dict(row) for row in paragraphs],
    }
    raw = (json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _launch_worker(
    *,
    project_root: Path,
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    input_path: Path,
    output_path: Path,
    index_root: Path,
    writable_root: Path,
    timeout_seconds: int,
) -> None:
    bwrap = Path("/usr/bin/bwrap")
    if not bwrap.is_file():
        raise MuSiQueOfficialHippoRAGError("network-isolating runtime is unavailable")
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
        "--bind",
        str(writable_root),
        str(writable_root),
        str(runtime_python),
        "-B",
        "-m",
        "replication_runtime.musique_official_hipporag_v1.worker",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--index-root",
        str(index_root),
        "--llm-model",
        str(local_llm_model),
        "--embedding-model",
        str(local_embedding_model),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        env=environment,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise MuSiQueOfficialHippoRAGError(
            "official retrieve-only worker failed; "
            f"returncode={completed.returncode}; "
            f"stdout_sha256={_sha256_bytes(completed.stdout)}; "
            f"stderr_sha256={_sha256_bytes(completed.stderr)}"
        )
    try:
        safe_status = json.loads(completed.stdout.decode("utf-8").strip().splitlines()[-1])
    except (IndexError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MuSiQueOfficialHippoRAGError("worker emitted no safe terminal status") from exc
    if safe_status != {"retrieval_count": 5, "status": "passed"}:
        raise MuSiQueOfficialHippoRAGError("worker safe terminal status drifted")


def run_official_hipporag_retrieve_only(
    *,
    question: str,
    paragraphs: Sequence[Mapping[str, object]],
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    binding_receipt_path: Path,
    work_root: Path,
    timeout_seconds: int = 900,
) -> tuple[int, ...]:
    """Return exactly five idx values from an isolated per-item official index.

    The work root must not exist.  It is private and ephemeral: the launcher
    removes the question, corpus, official index, and output file after the
    idx-only return value has been parsed.
    """

    validated_question, validated_paragraphs = validate_single_item(question, paragraphs)
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
        raise MuSiQueOfficialHippoRAGError("timeout must be an integer")
    if not 1 <= timeout_seconds <= 3600:
        raise MuSiQueOfficialHippoRAGError("timeout is outside the frozen bound")
    binding_receipt_path = binding_receipt_path.absolute()
    runtime_python = runtime_python.absolute()
    local_llm_model = local_llm_model.resolve(strict=True)
    local_embedding_model = local_embedding_model.resolve(strict=True)
    work_root = work_root.absolute()
    _assert_no_symlink_components(binding_receipt_path, "binding receipt")
    _assert_no_symlink_components(work_root.parent, "work root parent")
    if work_root.exists():
        raise MuSiQueOfficialHippoRAGError("per-item work root must not already exist")

    verify_live_binding(
        binding_receipt_path=binding_receipt_path,
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
    )
    project_root = binding_receipt_path.parent.parent
    work_root.mkdir(mode=0o700)
    try:
        for name in ("home", "cache", "tmp"):
            (work_root / name).mkdir(mode=0o700)
        input_path = work_root / "single_item.input.json"
        output_path = work_root / "retrieved_idx.json"
        index_root = work_root / "official_item_index"
        _write_private_input(
            input_path,
            question=validated_question,
            paragraphs=[
                {
                    "idx": row.idx,
                    "title": row.title,
                    "paragraph_text": row.paragraph_text,
                }
                for row in validated_paragraphs
            ],
        )
        _launch_worker(
            project_root=project_root,
            runtime_python=runtime_python,
            local_llm_model=local_llm_model,
            local_embedding_model=local_embedding_model,
            input_path=input_path,
            output_path=output_path,
            index_root=index_root,
            writable_root=work_root,
            timeout_seconds=timeout_seconds,
        )
        if output_path.is_symlink() or not output_path.is_file():
            raise MuSiQueOfficialHippoRAGError("worker idx-only output is unavailable")
        return parse_idx_only_output(
            output_path.read_bytes(), candidate_count=len(validated_paragraphs)
        )
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
