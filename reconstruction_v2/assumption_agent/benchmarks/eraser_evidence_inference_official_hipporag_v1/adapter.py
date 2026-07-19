"""Fail-closed launcher for one fresh, isolated Evidence Inference item."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Sequence

from replication_runtime.musique_official_hipporag_v1.runtime_attestation_v3 import (
    verify_formal_runtime_attestation_v3,
)

from .contract import (
    INPUT_SCHEMA,
    EraserEvidenceInferenceOfficialHippoRAGError,
    canonical_json_bytes,
    parse_ordinals_only_output,
    validate_single_item,
)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _assert_no_symlink_components(path: Path, field: str) -> None:
    absolute = path.absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise EraserEvidenceInferenceOfficialHippoRAGError(
                f"{field} contains a symlink component"
            )


def _write_private_input(
    path: Path,
    *,
    query: str,
    sentence_texts: Sequence[str],
) -> None:
    raw = canonical_json_bytes(
        {
            "query": query,
            "schema": INPUT_SCHEMA,
            "sentence_texts": list(sentence_texts),
        }
    )
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except OSError as exc:
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "private item input cannot be created exclusively"
        ) from exc
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
        raise EraserEvidenceInferenceOfficialHippoRAGError(
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
        (
            "assumption_agent.benchmarks."
            "eraser_evidence_inference_official_hipporag_v1.worker"
        ),
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
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "official item-local worker failed; "
            f"returncode={completed.returncode}; "
            f"stdout_sha256={_sha256_bytes(completed.stdout)}; "
            f"stderr_sha256={_sha256_bytes(completed.stderr)}"
        )
    try:
        terminal = json.loads(
            completed.stdout.decode("utf-8").strip().splitlines()[-1]
        )
    except (IndexError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "worker emitted no safe terminal status"
        ) from exc
    if terminal != {"retrieval_count": 5, "status": "passed"}:
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "worker safe terminal status drifted"
        )


def run_item_local_official_hipporag_v1(
    *,
    query: str,
    sentence_texts: Sequence[str],
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    base_binding_receipt_path: Path,
    attestation_receipt_path: Path,
    work_root: Path,
    timeout_seconds: int = 900,
) -> tuple[int, ...]:
    """Return five ordinals from a fresh index destroyed before return."""

    validated_query, validated_sentences = validate_single_item(
        query, sentence_texts
    )
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "timeout must be an integer"
        )
    if not 1 <= timeout_seconds <= 3600:
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "timeout is outside the frozen bound"
        )
    base_binding_receipt_path = base_binding_receipt_path.absolute()
    attestation_receipt_path = attestation_receipt_path.absolute()
    runtime_python = runtime_python.absolute()
    local_llm_model = local_llm_model.resolve(strict=True)
    local_embedding_model = local_embedding_model.resolve(strict=True)
    work_root = work_root.absolute()
    _assert_no_symlink_components(
        base_binding_receipt_path, "base binding receipt"
    )
    _assert_no_symlink_components(
        attestation_receipt_path, "attestation receipt"
    )
    _assert_no_symlink_components(work_root.parent, "work root parent")
    if work_root.exists() or work_root.is_symlink():
        raise EraserEvidenceInferenceOfficialHippoRAGError(
            "per-item work root must not already exist"
        )

    project_root = base_binding_receipt_path.parent.parent
    verify_formal_runtime_attestation_v3(
        project_root=project_root,
        attestation_receipt_path=attestation_receipt_path,
        base_binding_receipt_path=base_binding_receipt_path,
        runtime_python=runtime_python,
        local_llm_model=local_llm_model,
        local_embedding_model=local_embedding_model,
    )
    work_root.mkdir(mode=0o700)
    try:
        for name in ("home", "cache", "tmp"):
            (work_root / name).mkdir(mode=0o700)
        input_path = work_root / "single_item.input.json"
        output_path = work_root / "retrieved_ordinals.json"
        index_root = work_root / "official_item_index"
        _write_private_input(
            input_path,
            query=validated_query,
            sentence_texts=validated_sentences,
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
            raise EraserEvidenceInferenceOfficialHippoRAGError(
                "worker ordinal-only output is unavailable"
            )
        return parse_ordinals_only_output(
            output_path.read_bytes(),
            logical_sentence_count=len(validated_sentences),
        )
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


__all__ = ["run_item_local_official_hipporag_v1"]
