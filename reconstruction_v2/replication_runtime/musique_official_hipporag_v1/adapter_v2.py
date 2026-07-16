"""Prospective v2 official-HippoRAG retrieve-only launcher.

Unlike the frozen v1 launcher, this interface requires the separately frozen
v2 filesystem-attestation receipt.  Its pre-item check performs no executable
identity probe; the only subprocess launched here is the actual isolated
retrieve-only worker.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Mapping, Sequence

from .adapter import (
    _assert_no_symlink_components,
    _launch_worker,
    _write_private_input,
)
from .contract import (
    MuSiQueOfficialHippoRAGError,
    parse_idx_only_output,
    validate_single_item,
)
from .runtime_attestation_v2 import verify_formal_runtime_attestation_v2


def run_official_hipporag_retrieve_only_v2(
    *,
    question: str,
    paragraphs: Sequence[Mapping[str, object]],
    runtime_python: Path,
    local_llm_model: Path,
    local_embedding_model: Path,
    base_binding_receipt_path: Path,
    attestation_receipt_path: Path,
    work_root: Path,
    timeout_seconds: int = 900,
) -> tuple[int, ...]:
    """Return five paragraph indices under the prospective v2 trust root."""

    validated_question, validated_paragraphs = validate_single_item(question, paragraphs)
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
        raise MuSiQueOfficialHippoRAGError("timeout must be an integer")
    if not 1 <= timeout_seconds <= 3600:
        raise MuSiQueOfficialHippoRAGError("timeout is outside the frozen bound")
    base_binding_receipt_path = base_binding_receipt_path.absolute()
    attestation_receipt_path = attestation_receipt_path.absolute()
    runtime_python = runtime_python.absolute()  # retain lexical venv/bin/python
    local_llm_model = local_llm_model.resolve(strict=True)
    local_embedding_model = local_embedding_model.resolve(strict=True)
    work_root = work_root.absolute()
    _assert_no_symlink_components(base_binding_receipt_path, "base binding receipt")
    _assert_no_symlink_components(attestation_receipt_path, "attestation receipt")
    _assert_no_symlink_components(work_root.parent, "work root parent")
    if work_root.exists():
        raise MuSiQueOfficialHippoRAGError("per-item work root must not already exist")
    project_root = base_binding_receipt_path.parent.parent
    verify_formal_runtime_attestation_v2(
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


__all__ = ["run_official_hipporag_retrieve_only_v2"]
